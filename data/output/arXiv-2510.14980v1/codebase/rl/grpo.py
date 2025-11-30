# rl/grpo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import logging
from peft import LoraConfig, get_peft_model
from bitsandbytes.optim import AdamW8bit
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.config import Config


class GRPO:
    """
    Group Relative Policy Optimization for LLM fine-tuning with verifiable rewards.
    Implements the exact GRPO algorithm as specified in the paper, with Pass@k advantage estimator,
    KL regularization, LoRA parameter efficiency, and 8-bit AdamW optimizer.
    Designed to work with verl framework's rollout batch format.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 5e-6,
        kl_coef: float = 0.001,
        clip_ratio: float = 0.2,
        lora_rank: int = 64,
        use_mixed_precision: bool = True,
        gradient_clipping: float = 0.5,
        config: Optional[Config] = None
    ):
        """
        Initialize the GRPO optimizer.
        
        Args:
            model (nn.Module): Base LLM model (e.g., Qwen2.5-14B-Instruct)
            lr (float): Learning rate for optimizer (default: 5e-6 from config)
            kl_coef (float): KL divergence penalty coefficient (default: 0.001 from config)
            clip_ratio (float): Advantage clipping ratio (default: 0.2 from config)
            lora_rank (int): Rank of LoRA adapters (default: 64 from config)
            use_mixed_precision (bool): Whether to use mixed precision training (default: True from config)
            gradient_clipping (float): Gradient clipping threshold (default: 0.5 from config)
            config (Config, optional): Configuration loader. If None, loads from "config.yaml"
        """
        self.config = config or Config("config.yaml")
        self.logger = logging.getLogger(__name__)
        
        # Extract hyperparameters from config with defaults
        self.lr = lr
        self.kl_coef = kl_coef
        self.clip_ratio = clip_ratio
        self.lora_rank = lora_rank
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clipping = gradient_clipping
        
        # Validate parameters
        if not isinstance(self.lr, float) or self.lr <= 0:
            raise ValueError("lr must be a positive float")
        if not isinstance(self.kl_coef, float) or self.kl_coef < 0:
            raise ValueError("kl_coef must be a non-negative float")
        if not isinstance(self.clip_ratio, float) or self.clip_ratio <= 0:
            raise ValueError("clip_ratio must be a positive float")
        if not isinstance(self.lora_rank, int) or self.lora_rank <= 0:
            raise ValueError("lora_rank must be a positive integer")
        if not isinstance(self.gradient_clipping, float) or self.gradient_clipping <= 0:
            raise ValueError("gradient_clipping must be a positive float")
        
        # Store model reference
        self.model = model
        
        # Initialize LoRA adapters on ALL linear layers
        # For Qwen2.5-14B-Instruct, target modules include attention and MLP layers
        # Based on Qwen architecture, linear layers are: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        self.target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Create LoRA configuration
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,  # Standard: alpha = 2*r
            target_modules=self.target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(model, lora_config)
        
        # Only train LoRA parameters
        for name, param in self.model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # Initialize 8-bit AdamW optimizer on trainable parameters (LoRA only)
        self.optimizer = AdamW8bit(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01  # Standard AdamW weight decay
        )
        
        # Initialize gradient scaler for mixed precision
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Initialize tokenizer (for reference logprobs computation)
        # Note: This assumes the model is a Hugging Face model with tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            self.logger.warning(f"Failed to initialize tokenizer: {str(e)}. "
                              "Tokenization will be handled externally by verl_wrapper.")
            self.tokenizer = None
            
        # Track training statistics
        self.step_count = 0
        self.total_loss = 0.0
        self.total_kl = 0.0
        self.total_advantage = 0.0
        
        self.logger.info(f"GRPO initialized with: "
                         f"lr={self.lr}, kl_coef={self.kl_coef}, "
                         f"clip_ratio={self.clip_ratio}, lora_rank={self.lora_rank}, "
                         f"use_mixed_precision={self.use_mixed_precision}, "
                         f"gradient_clipping={self.gradient_clipping}")
        self.logger.info(f"LoRA target modules: {self.target_modules}")
        self.logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform a single GRPO update step using a batch of rollouts.
        
        Batch structure (from verl_wrapper.rollout()):
        {
            "prompts": List[str],  # e.g., ["Build a car to drive far"]
            "responses": List[str],  # generated JSON strings (construction trees)
            "rewards": List[float],  # R = R_valid * R_task (float)
            "logprobs": List[float],  # log probability of generated sequence under current policy
            "ref_logprobs": List[float],  # log probability under reference (pretrained) model
            "prompt_ids": List[int],  # indices to group responses by prompt (e.g., 0,0,0,1,1,1...)
        }
        
        Algorithm:
        1. Group responses by prompt_id
        2. For each group (one prompt's k rollouts):
           - Compute group mean reward
           - Compute advantage: A_j = R_j - mean_R
           - Clip advantages: A_j = clamp(A_j, -clip_ratio, +clip_ratio)
           - Compute importance ratio: ratio = exp(logprob_j - ref_logprob_j)
           - Compute surrogate loss: min(ratio * A_j, clip(ratio, 1-clip_ratio, 1+clip_ratio) * A_j)
        3. Compute KL divergence: KL = mean(logprob_j - ref_logprob_j)
        4. Total loss = -mean(surrogate) + kl_coef * KL
        5. Backpropagate and update only LoRA parameters
        6. Apply gradient clipping
        
        Args:
            batch (Dict[str, Any]): Batch of rollouts from verl_wrapper
            
        Returns:
            Dict[str, float]: Training statistics (loss, kl, advantage, etc.)
        """
        # Extract batch components
        prompts = batch["prompts"]
        responses = batch["responses"]
        rewards = batch["rewards"]
        logprobs = batch["logprobs"]
        ref_logprobs = batch["ref_logprobs"]
        prompt_ids = batch["prompt_ids"]
        
        # Convert to tensors
        rewards = torch.tensor(rewards, dtype=torch.float32)
        logprobs = torch.tensor(logprobs, dtype=torch.float32)
        ref_logprobs = torch.tensor(ref_logprobs, dtype=torch.float32)
        
        # Group by prompt_id
        grouped_rewards = defaultdict(list)
        grouped_logprobs = defaultdict(list)
        grouped_ref_logprobs = defaultdict(list)
        
        for i, pid in enumerate(prompt_ids):
            grouped_rewards[pid].append(rewards[i])
            grouped_logprobs[pid].append(logprobs[i])
            grouped_ref_logprobs[pid].append(ref_logprobs[i])
        
        # Convert lists to tensors
        for pid in grouped_rewards:
            grouped_rewards[pid] = torch.stack(grouped_rewards[pid])
            grouped_logprobs[pid] = torch.stack(grouped_logprobs[pid])
            grouped_ref_logprobs[pid] = torch.stack(grouped_ref_logprobs[pid])
        
        # Compute group-relative advantages and surrogate loss
        surrogate_losses = []
        kl_divergences = []
        
        for pid in grouped_rewards:
            # Get group data
            group_rewards = grouped_rewards[pid]
            group_logprobs = grouped_logprobs[pid]
            group_ref_logprobs = grouped_ref_logprobs[pid]
            
            # Compute group mean reward
            group_mean_reward = torch.mean(group_rewards)
            
            # Compute advantages: A_j = R_j - mean_R
            advantages = group_rewards - group_mean_reward
            
            # Clip advantages: A_j = clamp(A_j, -clip_ratio, +clip_ratio)
            advantages = torch.clamp(advantages, -self.clip_ratio, self.clip_ratio)
            
            # Compute importance sampling ratio: ratio = exp(logprob_j - ref_logprob_j)
            # Add small epsilon to avoid log(0) in numerical stability
            log_ratio = group_logprobs - group_ref_logprobs
            ratios = torch.exp(log_ratio)
            
            # Compute surrogate loss: min(ratio * A, clip(ratio, 1-clip_ratio, 1+clip_ratio) * A)
            # First term: ratio * advantage
            surr1 = ratios * advantages
            
            # Second term: clipped ratio * advantage
            clipped_ratios = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surr2 = clipped_ratios * advantages
            
            # Take minimum
            surrogate_loss = -torch.min(surr1, surr2)
            surrogate_losses.append(surrogate_loss)
            
            # Compute KL divergence: KL = mean(logprob_j - ref_logprob_j)
            # Note: This is the reverse KL: D_KL(pi || pi_ref)
            kl_div = torch.mean(log_ratio)
            kl_divergences.append(kl_div)
        
        # Combine all surrogate losses and KL divergences
        if len(surrogate_losses) == 0:
            self.logger.warning("No valid groups in batch")
            return {"loss": 0.0, "kl": 0.0, "advantage": 0.0, "num_groups": 0}
            
        all_surr_losses = torch.cat(surrogate_losses)
        all_kl_divs = torch.stack(kl_divergences)
        
        # Compute mean surrogate loss
        mean_surr_loss = torch.mean(all_surr_losses)
        
        # Compute mean KL divergence
        mean_kl_div = torch.mean(all_kl_divs)
        
        # Compute total loss: -mean(surrogate) + kl_coef * KL
        # Note: We use negative because we want to maximize the surrogate
        total_loss = mean_surr_loss + self.kl_coef * mean_kl_div
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Use mixed precision if enabled
        if self.use_mixed_precision:
            with autocast():
                # Compute loss (already computed above)
                pass
            # Scale loss and backpropagate
            self.scaler.scale(total_loss).backward()
            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)
        else:
            total_loss.backward()
        
        # Apply gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        
        # Update optimizer
        if self.use_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Update step counter
        self.step_count += 1
        
        # Compute statistics
        mean_advantage = torch.mean(torch.cat([advantages for advantages in grouped_rewards.values()])).item()
        mean_reward = torch.mean(rewards).item()
        
        # Store statistics for logging
        self.total_loss += total_loss.item()
        self.total_kl += mean_kl_div.item()
        self.total_advantage += mean_advantage
        
        # Return training statistics
        stats = {
            "loss": total_loss.item(),
            "kl": mean_kl_div.item(),
            "advantage": mean_advantage,
            "mean_reward": mean_reward,
            "grad_norm": grad_norm.item(),
            "num_groups": len(grouped_rewards),
            "total_rollouts": len(rewards)
        }
        
        return stats
