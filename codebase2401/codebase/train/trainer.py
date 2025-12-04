## train/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
import logging
import time
import os
from contextlib import nullcontext
from dataclasses import dataclass
import warnings

# Suppress non-critical warnings during training
warnings.filterwarnings("ignore", category=UserWarning)

from config import CONFIG
from model.apt_adapter import APTAdapter
from pruning.pruning_engine import PruningEngine
from tuning.tuning_controller import TuningController
from distillation.self_distiller import SelfDistiller
from data.data_loader import DatasetLoader
from utils.mask_scheduler import cubic_schedule, linear_weight_schedule

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Container for tracking training metrics."""
    step: int
    loss: float
    sft_loss: float
    kd_loss: float
    lambda_t: float
    current_sparsity: float
    target_sparsity: float
    tuning_params: int
    tuning_ratio: float
    learning_rate: float
    grad_norm: float
    gpu_memory_mb: int
    step_time_ms: float

class APTrainer:
    """
    Main trainer class for Adaptive Pruning and Tuning (APT) framework.
    
    Orchestrates the entire training pipeline including:
    - Model initialization with APT adapters
    - Data loading and preprocessing
    - Adaptive pruning and tuning updates
    - Self-distillation with EMA teacher
    - Two-phase training schedule
    - Comprehensive logging and checkpointing
    
    Based on paper's formulation in sec:apt and alg:epa.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the APT trainer with configuration.
        
        Args:
            config: Configuration dictionary (defaults to global CONFIG)
            
        Raises:
            RuntimeError: If CUDA is not available but required
            ValueError: If configuration is invalid
        """
        # Use provided config or fall back to global config
        self.config = config or CONFIG
        
        # Validate device availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for APT training")
        self.device = torch.device("cuda")
        
        # Setup output directory
        os.makedirs(self.config.output.output_dir, exist_ok=True)
        
        # Enable mixed precision if configured
        self.use_amp = (self.config.training.mixed_precision == "fp16")
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Internal state
        self.global_step = 0
        self.epoch = 0
        self.total_steps = 0
        self.best_metric = float('-inf')
        self.training_start_time = None
        
        # Components will be initialized in setup()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.eval_dataloader = None
        self.pruning_engine = None
        self.tuning_controller = None
        self.self_distiller = None
        self.dataset_loader = None
        
        # Training phase control
        self.phase = "adaptive"  # 'adaptive' or 'recovery'
        self.adaptive_phase_steps = None
        
        logger.info(f"APTrainer initialized with config: {self.config}")
    
    def setup(self) -> None:
        """
        Setup all components: model, data, optimizer, and auxiliary modules.
        
        This method must be called before training starts.
        """
        logger.info("Starting trainer setup...")
        
        # Set seed for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Load tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.tokenizer if self.config.model.tokenizer != "auto" 
            else self.config.model.name
        )
        
        # Load and wrap model
        self._setup_model()
        
        # Setup data loader
        self.dataset_loader = DatasetLoader(
            tokenizer=tokenizer,
            max_length=self.config.data.max_length,
            batch_size=self.config.data.batch_size,
            eval_batch_size=self.config.data.eval_batch_size
        )
        
        # Create dataloaders
        self.train_dataloader = self.dataset_loader.load_split("train")
        if "validation" in self.dataset_loader.raw_datasets:
            self.eval_dataloader = self.dataset_loader.load_split("validation")
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup auxiliary modules
        total_training_steps = len(self.train_dataloader) * self.config.training.epochs
        self.adaptive_phase_steps = total_training_steps  # All steps use adaptive strategy initially
        
        self.pruning_engine = PruningEngine(self.model, total_steps=total_training_steps)
        self.tuning_controller = TuningController(self.model, total_steps=total_training_steps)
        
        if self.config.distillation.enabled:
            self.self_distiller = SelfDistiller(self.model)
        else:
            self.self_distiller = None
        
        # Prepare model for training
        self.model.to(self.device)
        self.model.train()
        
        # Log setup completion
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Setup completed:")
        logger.info(f"  Model: {self.config.model.name}")
        logger.info(f"  Task: {self.config.data.task}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        logger.info(f"  Batch size: {self.config.data.batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Mixed precision: {self.use_amp}")
        logger.info(f"  Distillation: {'enabled' if self.self_distiller else 'disabled'}")
    
    def _setup_model(self) -> None:
        """
        Load pretrained model and inject APT adapters into specified layers.
        
        Handles different model architectures:
        - Encoder-only (RoBERTa/BERT): classification tasks
        - Encoder-decoder (T5): summarization/instruction tuning
        - Decoder-only (LLaMA): causal language modeling
        """
        from transformers import (
            AutoModelForSequenceClassification,
            AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM,
            AutoModelForCausalLM,
            AutoConfig
        )
        
        model_name = self.config.model.name
        task = self.config.data.task
        
        # Determine model type based on task
        if task in ["sst2", "mnli"]:
            model_class = AutoModelForSequenceClassification
            num_labels = 2 if task == "sst2" else 3
            config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
            self.model = model_class.from_pretrained(model_name, config=config)
            
        elif task == "squad_v2":
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
            
        elif task in ["cnn_dm", "alpaca"]:
            if "t5" in model_name:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Inject APT adapters into specified layers
        self._inject_apt_adapters()
        
        logger.debug(f"Model loaded: {model_name}, task: {task}")
    
    def _inject_apt_adapters(self) -> None:
        """
        Inject APT adapters into Transformer layers according to config.pruning.apply_to.
        
        Modifies:
        - MHA query and value projections
        - FFN intermediate projections
        
        Uses APTAdapter wrapper to add low-rank adaptation with pruning masks.
        """
        from collections import defaultdict
        
        # Track number of adapters injected
        adapter_counts = defaultdict(int)
        
        def inject_into_module(parent_module: nn.Module, parent_name: str = ''):
            """Recursively inject APT adapters into eligible linear layers."""
            for child_name, child_module in parent_module.named_children():
                current_name = f"{parent_name}.{child_name}" if parent_name else child_name
                
                # Skip certain modules
                if isinstance(child_module, (APTAdapter, nn.LayerNorm, nn.Dropout)):
                    continue
                
                # Check if this module should have APT adapter
                apply_targets = [tgt.lower() for tgt in self.config.pruning.apply_to]
                current_lower = current_name.lower()
                
                needs_adapter = any(target in current_lower for target in apply_targets)
                
                if needs_adapter and isinstance(child_module, nn.Linear):
                    # Replace linear layer with APT-wrapped version
                    in_features = child_module.in_features
                    out_features = child_module.out_features
                    
                    # Create APT adapter
                    apt_adapter = APTAdapter(
                        in_features=in_features,
                        out_features=out_features,
                        rank=self.config.apt.initial_rank,
                        alpha=self.config.apt.alpha_scaling
                    )
                    
                    # Wrap original weight with residual connection
                    # Original computation: W @ x
                    # New computation: W @ x + apt_adapter(x)
                    # So we keep W frozen and only tune apt_adapter
                    child_module.weight.requires_grad = False
                    if hasattr(child_module, 'bias') and child_module.bias is not None:
                        child_module.bias.requires_grad = False
                    
                    # Store original forward method
                    original_forward = child_module.forward
                    
                    # Define new forward that combines both
                    def make_forward(orig_forward, adapter):
                        def wrapped_forward(x):
                            return orig_forward(x) + adapter(x)
                        return wrapped_forward
                    
                    child_module.forward = make_forward(original_forward, apt_adapter)
                    
                    # Add adapter as submodule so it gets registered
                    setattr(parent_module, f"{child_name}_adapter", apt_adapter)
                    
                    # Update count
                    if 'query' in current_lower or 'key' in current_lower or 'value' in current_lower:
                        adapter_counts['mha'] += 1
                    elif 'ffn' in current_lower or 'intermediate' in current_lower:
                        adapter_counts['ffn'] += 1
                
                # Recurse into children
                inject_into_module(child_module, current_name)
        
        # Start injection from model root
        inject_into_module(self.model)
        
        logger.info(f"Injected APT adapters: {dict(adapter_counts)}")
    
    def _setup_optimizer(self) -> None:
        """
        Setup optimizer for training only tunable parameters.
        
        Tunable parameters include:
        - APT adapter matrices A and B
        - Transformation modules (if used in distillation)
        - Not the base model weights (frozen)
        """
        # Collect all trainable parameters
        trainable_params = []
        
        def collect_trainable_params(module: nn.Module, prefix: str = ''):
            """Recursively collect parameters that require gradients."""
            for name, param in module.named_parameters(recurse=False):
                full_name = f"{prefix}.{name}" if prefix else name
                if param.requires_grad:
                    trainable_params.append(param)
            
            # Recurse into children
            for child_name, child_module in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                collect_trainable_params(child_module, child_prefix)
        
        collect_trainable_params(self.model)
        
        # Create optimizer
        if self.config.training.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                eps=self.config.training.adam_epsilon
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
        
        # Setup scheduler if needed
        # For now, use constant LR; could extend to support warmup/cosine
        self.scheduler = None
        
        logger.info(f"Optimizer setup complete with {len(trainable_params)} trainable parameters")
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the full training loop with adaptive pruning and tuning.
        
        Implements two-phase training:
        Phase 1: Adaptive pruning/tuning with self-distillation
        Phase 2: Recovery fine-tuning (optional)
        
        Returns:
            Dictionary with final training statistics
        """
        logger.info("Starting training...")
        self.training_start_time = time.time()
        
        # Calculate total steps
        steps_per_epoch = len(self.train_dataloader)
        self.total_steps = steps_per_epoch * self.config.training.epochs
        
        # Training loop
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # Set progress bar description
            epoch_desc = f"Epoch {epoch+1}/{self.config.training.epochs}"
            
            # Training phase
            self.model.train()
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Perform training step
                metrics = self.step(batch)
                
                # Perform adaptive update periodically
                if self.global_step % self.config.apt.update_frequency == 0:
                    self.adaptive_update(self.global_step)
                
                # Log metrics periodically
                if batch_idx % 10 == 0:
                    self._log_metrics(metrics, epoch_desc, batch_idx, steps_per_epoch)
                
                # Increment global step
                self.global_step += 1
                
                # Break if reached total steps
                if self.global_step >= self.total_steps:
                    break
            
            # End of epoch processing
            epoch_time = time.time() - epoch_start_time
            logger.info(f"{epoch_desc} completed in {epoch_time:.2f}s")
            
            # Evaluate at end of epoch if possible
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"{epoch_desc} Evaluation - Accuracy: {eval_metrics.get('accuracy', 0):.4f}")
                
                # Save best model
                metric_value = eval_metrics.get('accuracy', eval_metrics.get('f1', 0))
                if metric_value > self.best_metric:
                    self.best_metric = metric_value
                    self._save_checkpoint(f"best")
            
            # Break if reached total steps
            if self.global_step >= self.total_steps:
                break
        
        # Final evaluation
        final_eval = {}
        if self.eval_dataloader is not None:
            final_eval = self.evaluate()
            logger.info(f"Final Evaluation - Accuracy: {final_eval.get('accuracy', 0):.4f}")
        
        # Save final model
        self._save_checkpoint("final")
        
        # Training summary
        total_training_time = time.time() - self.training_start_time
        summary = {
            "total_steps": self.global_step,
            "total_epochs": self.epoch + 1,
            "total_training_time_s": total_training_time,
            "final_accuracy": final_eval.get("accuracy", 0),
            "final_f1": final_eval.get("f1", 0),
            "final_loss": metrics.loss if 'metrics' in locals() else 0,
            "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
            "config": self.config
        }
        
        logger.info(f"Training completed. Summary: {summary}")
        return summary
    
    def _prepare_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for training by moving to device and converting types.
        
        Args:
            batch: Input batch dictionary
            
        Returns:
            Prepared batch on proper device
        """
        prepared = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                prepared[k] = v.to(self.device)
            else:
                prepared[k] = v
        return prepared
    
    def step(self, batch: Dict[str, torch.Tensor]) -> TrainingMetrics:
        """
        Perform a single training step: forward, loss, backward, update.
        
        Args:
            batch: Input batch containing input_ids, labels, etc.
            
        Returns:
            TrainingMetrics object with step statistics
        """
        step_start_time = time.time()
        
        # Prepare context managers
        ctx_manager = torch.cuda.amp.autocast() if self.use_amp else nullcontext()
        
        # Forward pass
        with ctx_manager:
            # Compute supervised fine-tuning loss
            outputs = self.model(**batch)
            sft_loss = outputs.loss
            
            # Initialize KD loss
            kd_loss = torch.tensor(0.0, device=self.device)
            lambda_t = 0.0
            
            # Apply knowledge distillation if enabled
            if self.self_distiller is not None and self.self_distiller.enabled:
                # Get current lambda_t based on schedule
                lambda_t = self.self_distiller.get_current_lambda(
                    self.global_step, self.total_steps
                )
                
                # Update teacher before computing KD loss
                self.self_distiller.update_teacher()
                
                # Prepare teacher outputs (this would involve hooks in practice)
                # Simplified implementation here
                with torch.no_grad():
                    teacher_outputs = self._get_teacher_outputs(batch)
                
                # Compute KD loss between student and teacher
                student_hidden_states = getattr(outputs, 'hidden_states', None)
                if student_hidden_states is not None and teacher_outputs:
                    layer_mapping = self.self_distiller.compute_layer_mapping({})
                    kd_loss = self.self_distiller.compute_kd_loss(
                        student_outputs={i: h for i, h in enumerate(student_hidden_states)},
                        teacher_outputs=teacher_outputs,
                        layer_mapping=layer_mapping
                    )
            
            # Combine losses
            if self.self_distiller is not None and self.self_distiller.enabled:
                total_loss = (1 - lambda_t) * sft_loss + lambda_t * kd_loss
            else:
                total_loss = sft_loss
                kd_loss = torch.tensor(0.0, device=self.device)
                lambda_t = 0.0
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total_loss.backward()
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Compute gradient norm
        grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # Record metrics
        step_time_ms = (time.time() - step_start_time) * 1000
        
        metrics = TrainingMetrics(
            step=self.global_step,
            loss=total_loss.item(),
            sft_loss=sft_loss.item(),
            kd_loss=kd_loss.item(),
            lambda_t=lambda_t,
            current_sparsity=self.pruning_engine.get_current_sparsity() if self.pruning_engine else 0.0,
            target_sparsity=self.config.apt.target_sparsity,
            tuning_params=self.tuning_controller.current_total_tuning_params if self.tuning_controller else 0,
            tuning_ratio=self.tuning_controller.get_current_tuning_ratio() if self.tuning_controller else 0.0,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            grad_norm=grad_norm,
            gpu_memory_mb=int(torch.cuda.memory_allocated() / 1024 / 1024),
            step_time_ms=step_time_ms
        )
        
        return metrics
    
    def _get_teacher_outputs(self, batch: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Get intermediate outputs from teacher model for KD loss.
        
        In practice, this would use hooks to capture hidden states.
        This is a simplified placeholder implementation.
        
        Args:
            batch: Input batch
            
        Returns:
            Dictionary of teacher outputs by layer index
        """
        # This would be implemented with proper hook system
        # Returning empty dict for now
        return {}
    
    def adaptive_update(self, global_step: int) -> Dict[str, Any]:
        """
        Perform adaptive pruning and tuning update.
        
        Called periodically during training to:
        1. Update pruning masks based on salience
        2. Increase ranks in important adapters
        3. Reset optimizer if architecture changed
        
        Args:
            global_step: Current global training step
            
        Returns:
            Dictionary with update statistics
        """
        update_stats = {}
        
        # Only perform update if pruning engine exists
        if self.pruning_engine is None:
            return {"update_performed": False, "reason": "no_pruning_engine"}
        
        # Pruning update
        pruning_stats = self.pruning_engine.adaptive_update(global_step)
        update_stats.update({"pruning": pruning_stats})
        
        # Tuning update
        if self.tuning_controller is not None:
            tuning_stats = self.tuning_controller.adaptive_update(global_step)
            update_stats.update({"tuning": tuning_stats})
        
        # After structural changes, reset optimizer state for stability
        # As mentioned in paper's limitation section
        if pruning_stats.get("update_performed") or tuning_stats.get("update_performed"):
            # Reinitialize optimizer state
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        state = self.optimizer.state[p]
                        # Reset momentum and variance
                        if 'exp_avg' in state:
                            state['exp_avg'].zero_()
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'].zero_()
        
        return update_stats
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on validation set.
        
        Returns:
            Dictionary with evaluation metrics (accuracy, F1, etc.)
        """
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Extract predictions and references
                if hasattr(outputs, 'logits'):
                    preds = outputs.logits.argmax(dim=-1)
                    labels = batch['labels']
                    
                    # Handle padding
                    active_mask = labels != -100
                    predictions.extend(preds[active_mask].cpu().numpy())
                    references.extend(labels[active_mask].cpu().numpy())
        
        # Compute metrics based on task
        task = self.config.data.task
        
        if task in ["sst2", "mnli"]:
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(references, predictions)
            return {"accuracy": acc}
            
        elif task == "squad_v2":
            # Would use official SQuAD script
            # Simplified here
            from sklearn.metrics import f1_score
            f1 = f1_score(references, predictions, average='macro')
            return {"f1": f1}
            
        else:
            # Default to accuracy
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(references, predictions)
            return {"accuracy": acc}
    
    def _log_metrics(self, metrics: TrainingMetrics, epoch_desc: str, 
                   batch_idx: int, steps_per_epoch: int) -> None:
        """
        Log training metrics to console.
        
        Args:
            metrics: TrainingMetrics object
            epoch_desc: Epoch description string
            batch_idx: Current batch index
            steps_per_epoch: Total steps per epoch
        """
        logger.info(
            f"{epoch_desc} Step {batch_idx}/{steps_per_epoch} | "
            f"Loss: {metrics.loss:.4f} | "
            f"SFT: {metrics.sft_loss:.4f} | "
            f"KD: {metrics.kd_loss:.4f} | "
            f"λ: {metrics.lambda_t:.3f} | "
            f"Sparsity: {metrics.current_sparsity:.3f} | "
            f"Tuning: {metrics.tuning_params:,} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Mem: {metrics.gpu_memory_mb}MB"
        )
    
    def _save_checkpoint(self, suffix: str = "") -> None:
        """
        Save model checkpoint.
        
        Args:
            suffix: String to append to checkpoint name
        """
        if not self.config.output.save_model:
            return
            
        save_path = os.path.join(
            self.config.output.output_dir,
            f"checkpoint_{suffix}.pt"
        )
        
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'config': self.config
        }, save_path)
        
        logger.info(f"Checkpoint saved to {save_path}")

__all__ = ['APTrainer']
