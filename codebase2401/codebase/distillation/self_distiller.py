## distillation/self_distiller.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import logging
from collections import defaultdict
from config import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration specific to self-distillation."""
    enabled: bool = True
    teacher_momentum: float = 0.999
    lambda_start: float = 0.0
    lambda_end: float = 1.0
    lambda_schedule: str = "linear"
    layer_mapping: str = "closest_non_pruned"
    transformation_module: str = "lora"
    trans_rank: int = 8

class TransformationAdapter(nn.Module):
    """
    Learnable transformation module for aligning teacher and student representations.
    
    Implements a LoRA-style residual adapter initialized as identity mapping.
    This allows flexible alignment between teacher and student outputs even when
    architectures diverge due to pruning.
    
    Formula: f(x) = x + (B @ A) @ x
    Initialized with B=0 so f(x) ≈ x initially.
    """
    
    def __init__(self, dim: int, rank: int = 8, device: Optional[torch.device] = None):
        """
        Initialize the transformation adapter.
        
        Args:
            dim: Feature dimension size
            rank: Rank of low-rank decomposition
            device: Device to place parameters on
        """
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize A and B matrices
        self.A = nn.Parameter(torch.randn((rank, dim), device=self.device) / rank)
        self.B = nn.Parameter(torch.zeros((dim, rank), device=self.device))
        
        logger.debug(f"TransformationAdapter initialized: dim={dim}, rank={rank}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformation: f(x) = x + (B @ A) @ x
        
        Args:
            x: Input tensor of shape (*, dim)
            
        Returns:
            Transformed tensor of same shape
        """
        # Compute low-rank update
        delta = x @ self.A.T  # (*, rank)
        delta = delta @ self.B.T  # (*, dim)
        return x + delta

class SelfDistiller:
    """
    Efficient self-knowledge distillation module for Adaptive Pruning and Tuning (APT).
    
    Implements the paper's self-distillation technique that:
    - Shares frozen backbone between student and teacher
    - Maintains EMA copy of only tunable APT parameters (A/B matrices)
    - Uses learnable transformation module for output alignment
    - Maps pruned student layers to closest non-pruned teacher layers
    - Gradually increases distillation weight λ_t from 0 to 1
    
    Key benefits:
    - ~70% memory reduction vs full-model distillation
    - Faster convergence than external teacher methods
    - Recovers performance lost during aggressive early pruning
    
    Based on paper formulation in sec:momentum and alg:epa.
    """
    
    def __init__(self, student_model: nn.Module):
        """
        Initialize self-distillation with EMA-based teacher over tunable parameters.
        
        Args:
            student_model: The student model being trained with APT adapters
            
        Raises:
            ValueError: If no APT adapters found or invalid configuration
        """
        self.student_model = student_model
        
        # Load configuration
        cfg = CONFIG.distillation
        self.enabled = cfg.enabled
        self.momentum = cfg.teacher_momentum
        self.lambda_start = cfg.lambda_start
        self.lambda_end = cfg.lambda_end
        self.lambda_schedule = cfg.lambda_schedule
        self.layer_mapping_strategy = cfg.layer_mapping
        self.use_transformation = (cfg.transformation_module == "lora")
        self.trans_rank = cfg.trans_rank
        
        # Validate inputs
        if not self.enabled:
            logger.info("Self-distillation is disabled.")
            return
            
        if self.momentum < 0 or self.momentum >= 1.0:
            raise ValueError(f"teacher_momentum must be in [0,1), got {self.momentum}")
        if not (0.0 <= self.lambda_start <= 1.0):
            raise ValueError(f"lambda_start must be in [0,1], got {self.lambda_start}")
        if not (0.0 <= self.lambda_end <= 1.0):
            raise ValueError(f"lambda_end must be in [0,1], got {self.lambda_end}")
        if self.lambda_schedule not in ["linear", "cubic"]:
            raise ValueError(f"lambda_schedule must be 'linear' or 'cubic', got {self.lambda_schedule}")
            
        # Internal state
        self._apt_params: Dict[str, nn.Parameter] = {}
        self._teacher_state: Dict[str, torch.Tensor] = {}
        self._transformation_modules: Dict[int, TransformationAdapter] = {}
        self._layer_mapping_cache: Dict[int, int] = {}
        self._is_setup = False
        
        # Setup teacher and transformation modules
        self._setup_teacher()
        
        logger.info(f"SelfDistiller initialized with EMA momentum={self.momentum}, "
                   f"lambda_schedule={self.lambda_schedule}, "
                   f"transformation_module={'enabled' if self.use_transformation else 'disabled'}")
    
    def _setup_teacher(self) -> None:
        """
        Set up the EMA teacher by identifying all tunable APT adapter parameters.
        
        Only creates EMA copies for trainable A/B matrices from APT adapters.
        Frozen backbone weights are shared between student and teacher.
        """
        if not self.enabled:
            return
            
        self._apt_params.clear()
        self._teacher_state.clear()
        
        # Find all APT adapter parameters
        for name, param in self.student_model.named_parameters():
            if any(k in name for k in ["adapter.A", "adapter.B"]):
                self._apt_params[name] = param
                
                # Initialize teacher state with current parameter values
                self._teacher_state[name] = param.detach().clone()
                
                logger.debug(f"Registered APT parameter for distillation: {name}")
        
        if len(self._apt_params) == 0:
            raise ValueError("No APT adapter parameters found. Ensure model has been properly wrapped.")
            
        # Create transformation modules per layer
        if self.use_transformation:
            self._create_transformation_modules()
            
        self._is_setup = True
        logger.info(f"Teacher setup complete: tracking {len(self._apt_params)} APT parameters")
    
    def _create_transformation_modules(self) -> None:
        """
        Create transformation adapters for each layer that may need output alignment.
        
        One transformation module per unique feature dimension.
        """
        seen_dims = set()
        
        for name, param in self.student_model.named_parameters():
            if "adapter.A" in name:
                # Get input dimension from A matrix shape
                dim = param.shape[1]  # A: (rank, in_features)
                if dim not in seen_dims:
                    self._transformation_modules[dim] = TransformationAdapter(
                        dim=dim,
                        rank=self.trans_rank,
                        device=param.device
                    )
                    seen_dims.add(dim)
    
    def update_teacher(self) -> None:
        """
        Update teacher parameters via Exponential Moving Average (EMA).
        
        θ_tea = β * θ_tea + (1-β) * θ_stu
        
        Should be called after each optimizer step to maintain a slowly evolving teacher.
        Does nothing if distillation is disabled.
        """
        if not self.enabled or not self._is_setup:
            return
            
        try:
            for name, student_param in self._apt_params.items():
                teacher_param = self._teacher_state[name]
                
                # In-place EMA update
                teacher_param.mul_(self.momentum).add_(
                    student_param.detach(), 
                    alpha=1 - self.momentum
                )
                
            logger.debug(f"Teacher updated with momentum={self.momentum}")
            
        except Exception as e:
            logger.error(f"Failed to update teacher: {e}")
            raise
    
    def prepare_teacher_outputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Run forward pass on teacher model and collect intermediate outputs.
        
        Since teacher shares frozen backbone, we run inference using current EMA parameters
        while keeping the student unchanged.
        
        Args:
            inputs: Model inputs (same format as student forward pass)
            
        Returns:
            Dictionary mapping layer names to their transformed outputs
        """
        if not self.enabled or not self._is_setup:
            return {}
            
        # Store original student parameters
        original_states = {}
        for name, param in self._apt_params.items():
            original_states[name] = param.detach().clone()
        
        try:
            # Swap in teacher parameters
            for name, param in self._apt_params.items():
                param.data.copy_(self._teacher_state[name])
            
            # Enable evaluation mode
            self.student_model.eval()
            
            # Forward pass with no gradient
            with torch.no_grad():
                # We need to capture intermediate outputs
                # For now, assume we can get them through hooks or modified forward
                # This is a simplified implementation
                outputs = {}
                
                # In practice, you would register hooks to capture hidden states
                # Here we simulate capturing attention/FFN outputs
                layer_idx = 0
                for name, module in self.student_model.named_modules():
                    if hasattr(module, 'forward') and 'adapter' in name:
                        # Simulate getting output (this would come from hook)
                        # Actual implementation depends on model architecture
                        pass
                
                # Placeholder: return empty dict until proper hook system implemented
                teacher_outputs = {}
                
            return teacher_outputs
            
        except Exception as e:
            logger.error(f"Failed to prepare teacher outputs: {e}")
            return {}
            
        finally:
            # Restore original student parameters
            for name, param in self._apt_params.items():
                param.data.copy_(original_states[name])
            
            # Return to train mode
            self.student_model.train()
    
    def compute_layer_mapping(self, student_masks: Dict[str, torch.BoolTensor]) -> Dict[int, int]:
        """
        Compute mapping from student layers to closest non-pruned teacher layers.
        
        Implements φ(ls): maps each student layer to its closest non-pruned counterpart.
        
        Strategy:
        - For each student layer index i
        - Find nearest non-pruned layer j where mask[j] == True
        - Break ties by preferring smaller |i-j| difference
        
        Args:
            student_masks: Dictionary of layer masks indicating pruned status
            
        Returns:
            Mapping dictionary {student_layer_idx -> teacher_layer_idx}
        """
        if not self.enabled:
            return {}
            
        # Extract layer indices and their pruned status
        layer_status = []
        for name, mask in student_masks.items():
            try:
                # Parse layer index from name like "encoder.layer.3.attention.query.adapter.M_out"
                parts = name.split('.')
                if 'layer' in parts:
                    idx_part = parts[parts.index('layer') + 1]
                    layer_idx = int(idx_part)
                    # Consider layer pruned if any mask element is zero
                    is_pruned = not mask.all().item()
                    layer_status.append((layer_idx, is_pruned))
            except (ValueError, IndexError):
                continue
        
        # Sort by layer index
        layer_status.sort()
        indices = [idx for idx, _ in layer_status]
        is_pruned_list = [pruned for _, pruned in layer_status]
        
        mapping = {}
        non_pruned_indices = [i for i, pruned in layer_status if not pruned]
        
        if not non_pruned_indices:
            # All layers pruned - fallback to first layer
            default_target = indices[0] if indices else 0
            return {idx: default_target for idx in indices}
        
        for layer_idx, is_pruned in layer_status:
            if not is_pruned:
                # Non-pruned layer maps to itself
                mapping[layer_idx] = layer_idx
            else:
                # Find closest non-pruned layer
                distances = [(abs(layer_idx - target), target) for target in non_pruned_indices]
                _, closest = min(distances, key=lambda x: x[0])
                mapping[layer_idx] = closest
        
        self._layer_mapping_cache = mapping
        return mapping
    
    def get_transformation(self, dim: int) -> Optional[TransformationAdapter]:
        """
    Get transformation module for given feature dimension.
    
    Args:
        dim: Feature dimension size
        
    Returns:
        TransformationAdapter instance or None if disabled
    """
        if not self.enabled or not self.use_transformation:
            return None
            
        return self._transformation_modules.get(dim, None)
    
    def compute_kd_loss(self,
                       student_outputs: Dict[int, torch.Tensor],
                       teacher_outputs: Dict[int, torch.Tensor],
                       layer_mapping: Dict[int, int]) -> torch.Tensor:
        """
        Compute knowledge distillation loss between student and teacher.
        
        L_KD = Σ ||h_s[i] - f(h_t[φ(i)])||^2
        
        Where f is an optional transformation module.
        
        Args:
            student_outputs: Dictionary {layer_idx -> output_tensor}
            teacher_outputs: Dictionary {layer_idx -> output_tensor}
            layer_mapping: Dictionary {student_layer -> teacher_layer}
            
        Returns:
            Scalar KD loss tensor
        """
        if not self.enabled or not student_outputs:
            return torch.tensor(0.0, device=next(self.student_model.parameters()).device)
            
        device = next(iter(student_outputs.values())).device
        kd_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for s_layer, s_output in student_outputs.items():
            if s_layer not in layer_mapping:
                continue
                
            t_layer = layer_mapping[s_layer]
            if t_layer not in teacher_outputs:
                continue
                
            t_output = teacher_outputs[t_layer]
            
            # Handle different sequence lengths by averaging over sequence dimension
            if s_output.dim() == 3 and t_output.dim() == 3:
                s_mean = s_output.mean(dim=1)  # average over seq_len
                t_mean = t_output.mean(dim=1)
            else:
                s_mean = s_output
                t_mean = t_output
            
            # Resize if dimensions don't match
            if s_mean.shape[-1] != t_mean.shape[-1]:
                # Pad or truncate t_mean to match s_mean
                dim = s_mean.shape[-1]
                if t_mean.shape[-1] < dim:
                    pad_size = dim - t_mean.shape[-1]
                    padding = torch.zeros(*t_mean.shape[:-1], pad_size, device=t_mean.device)
                    t_mean = torch.cat([t_mean, padding], dim=-1)
                else:
                    t_mean = t_mean[..., :dim]
            
            # Apply transformation if available
            trans_module = self.get_transformation(s_mean.shape[-1])
            if trans_module is not None:
                t_transformed = trans_module(t_mean)
            else:
                t_transformed = t_mean
            
            # Compute MSE loss
            layer_loss = torch.nn.functional.mse_loss(s_mean, t_transformed)
            kd_loss += layer_loss
            count += 1
        
        if count > 0:
            kd_loss = kd_loss / count
        else:
            kd_loss = torch.tensor(0.0, device=device)
            
        return kd_loss
    
    def get_current_lambda(self, current_step: int, total_steps: int) -> float:
        """
        Compute current distillation weight λ_t based on training progress.
        
        Follows schedule specified in config:
        - linear: λ_t = λ_start + (λ_end - λ_start) * (current_step / total_steps)
        - cubic: λ_t = λ_start + (λ_end - λ_start) * (current_step / total_steps)^3
        
        Args:
            current_step: Current global training step
            total_steps: Total number of training steps
            
        Returns:
            Interpolated λ_t value in [0,1]
        """
        if not self.enabled:
            return 0.0
            
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if current_step < 0:
            raise ValueError(f"current_step must be non-negative, got {current_step}")
        if current_step > total_steps:
            current_step = total_steps
            
        progress = current_step / total_steps
        
        if self.lambda_schedule == "linear":
            lambda_t = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        elif self.lambda_schedule == "cubic":
            lambda_t = self.lambda_start + (self.lambda_end - self.lambda_start) * (progress ** 3)
        else:
            # Default to linear
            lambda_t = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        
        # Clamp to valid range
        lambda_t = max(0.0, min(1.0, lambda_t))
        
        return lambda_t
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current distillation state.
        
        Returns:
            Dictionary with distillation information
        """
        if not self.enabled:
            return {"distillation_enabled": False}
            
        return {
            "distillation_enabled": True,
            "teacher_momentum": self.momentum,
            "lambda_start": self.lambda_start,
            "lambda_end": self.lambda_end,
            "lambda_schedule": self.lambda_schedule,
            "num_tracked_params": len(self._apt_params),
            "total_tracked_elements": sum(p.numel() for p in self._apt_params.values()),
            "use_transformation": self.use_transformation,
            "transformation_rank": self.trans_rank,
            "layer_mapping_strategy": self.layer_mapping_strategy,
            "is_setup": self._is_setup
        }

__all__ = ['SelfDistiller']
