## model/apt_adapter.py
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
from config import CONFIG

@dataclass
class AdapterConfig:
    """Configuration specific to a single APT adapter instance."""
    in_features: int
    out_features: int
    rank: int = 8
    alpha: float = 2.0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

class APTAdapter(nn.Module):
    """
    Adaptive Pruning and Tuning (APT) Adapter module.
    
    This class implements the core APT mechanism that combines structured pruning
    with dynamic low-rank adaptation. It extends LoRA by adding binary masks for
    input/output dimensions and supporting runtime rank adjustment.
    
    Key features:
    - Structured pruning via M_in and M_out binary masks
    - Dynamic rank increase with stable initialization
    - Configurable scaling factor (alpha)
    - Efficient parameter counting for budget management
    
    Based on paper formulation:
    h = Wx + α/r * (M_out ⊙ (B @ A)) @ (M_in ⊙ x)
    """
    
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 rank: int = None,
                 alpha: float = None,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize the APT adapter with specified dimensions and hyperparameters.
        
        Args:
            in_features: Input dimension size
            out_features: Output dimension size  
            rank: Initial rank for low-rank decomposition (default from config)
            alpha: Scaling factor for adapter output (default from config)
            device: Device to place tensors on (default: current device)
            dtype: Data type for parameters (default: float32)
            
        Raises:
            ValueError: If rank is larger than min(in_features, out_features)
        """
        super().__init__()
        
        # Use defaults from global config if not provided
        self.rank = rank or CONFIG.apt.initial_rank
        self.alpha = alpha or CONFIG.apt.alpha_scaling
        
        # Validate rank constraint
        max_possible_rank = min(in_features, out_features)
        if self.rank > max_possible_rank:
            raise ValueError(
                f"Rank {self.rank} cannot exceed min(in_features, out_features)={max_possible_rank}"
            )
            
        self.in_features = in_features
        self.out_features = out_features
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32
        
        # Initialize low-rank matrices following LoRA scheme
        # A: Gaussian initialization, B: zero initialization
        self.A = nn.Parameter(torch.randn((self.rank, in_features), device=self.device, dtype=self.dtype) / self.rank)
        self.B = nn.Parameter(torch.zeros((out_features, self.rank), device=self.device, dtype=self.dtype))
        
        # Binary pruning masks (initially all ones - full connectivity)
        self.M_in = nn.Parameter(torch.ones(in_features, device=self.device, dtype=torch.bool), requires_grad=False)
        self.M_out = nn.Parameter(torch.ones(out_features, device=self.device, dtype=torch.bool), requires_grad=False)
        
        # Register buffer to track current effective rank
        self.register_buffer('current_rank', torch.tensor(self.rank, dtype=torch.int32))
        
        # Ensure proper scaling
        self.scaling = self.alpha / self.rank
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the APT adapter.
        
        Implements: output = base_weight @ x + (α/r) * (M_out ⊙ (B @ A)) @ (M_in ⊙ x)
        
        Args:
            x: Input tensor of shape (*, in_features)
            
        Returns:
            Output tensor of shape (*, out_features) with adapter applied
            
        Note:
            The base weight matrix W is assumed to be handled by the parent layer.
            This adapter only computes the residual update term.
        """
        # Apply input mask
        x_masked = x * self.M_in
        
        # Compute low-rank transformation: B @ A @ x_masked
        # Using efficient matmul chain
        result = x_masked @ self.A.T  # (*, rank)
        result = result @ self.B.T   # (*, out_features)
        
        # Scale the result
        result = result * self.scaling
        
        # Apply output mask
        result = result * self.M_out
        
        return result
    
    def increase_rank(self, delta: int) -> None:
        """
        Dynamically increase the adapter rank by delta while maintaining stable output.
        
        Following paper's initialization strategy:
        - New rows in A are initialized with random Gaussian noise
        - New columns in B are initialized with zeros
        - This ensures the layer output remains unchanged before/after expansion
        
        Args:
            delta: Number of ranks to add (must be positive)
            
        Raises:
            ValueError: If delta is not positive
        """
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")
            
        old_rank = self.rank
        new_rank = old_rank + delta
        
        # Store old parameters
        old_A = self.A.data
        old_B = self.B.data
        
        # Create new larger matrices with proper initialization
        new_A = torch.empty((new_rank, self.in_features), device=self.device, dtype=self.dtype)
        new_B = torch.empty((self.out_features, new_rank), device=self.device, dtype=self.dtype)
        
        # Copy existing values to top-left portion
        new_A[:old_rank, :] = old_A
        new_B[:, :old_rank] = old_B
        
        # Initialize new portions following LoRA scheme
        # New rows in A: Gaussian noise scaled by 1/rank
        new_A[old_rank:, :] = torch.randn((delta, self.in_features), device=self.device, dtype=self.dtype) / new_rank
        
        # New columns in B: zeros
        new_B[:, old_rank:] = 0.0
        
        # Replace parameters
        self.A = nn.Parameter(new_A)
        self.B = nn.Parameter(new_B)
        
        # Update metadata
        self.rank = new_rank
        self.current_rank.fill_(new_rank)
        self.scaling = self.alpha / self.rank
        
    def apply_masks(self, 
                   M_in: Optional[torch.BoolTensor] = None, 
                   M_out: Optional[torch.BoolTensor] = None,
                   gradual: bool = False,
                   decay_step: float = None) -> None:
        """
        Apply new pruning masks to control input/output dimensions.
        
        Args:
            M_in: New input mask of shape (in_features,)
            M_out: New output mask of shape (out_features,)
            gradual: Whether to decay masks gradually instead of abrupt change
            decay_step: Step size for gradual decay (default from config)
            
        Note:
            If gradual=True, masks are decayed by decay_step per call rather than set directly.
            This improves training stability as mentioned in paper.
        """
        if decay_step is None:
            decay_step = CONFIG.pruning.mask_decay_step
            
        if M_in is not None:
            if M_in.shape != (self.in_features,):
                raise ValueError(f"M_in shape {M_in.shape} mismatch with in_features {self.in_features}")
                
            if gradual:
                # Gradual decay: decrease mask values slowly
                current_float = self.M_in.float()
                target_float = M_in.float()
                update = (target_float - current_float) * decay_step
                updated = current_float + update
                # Clip to [0,1] and convert back to bool
                self.M_in.data = (updated >= 0.5).to(torch.bool)
            else:
                self.M_in.data = M_in.to(torch.bool)
                
        if M_out is not None:
            if M_out.shape != (self.out_features,):
                raise ValueError(f"M_out shape {M_out.shape} mismatch with out_features {self.out_features}")
                
            if gradual:
                current_float = self.M_out.float()
                target_float = M_out.float()
                update = (target_float - current_float) * decay_step
                updated = current_float + update
                self.M_out.data = (updated >= 0.5).to(torch.bool)
            else:
                self.M_out.data = M_out.to(torch.bool)
    
    def get_params_count(self) -> int:
        """
        Calculate the total number of trainable parameters in this adapter.
        
        Returns:
            Total count of trainable parameters (A + B matrices only)
            
        Note:
            Masks are not counted as they are boolean and not trained with gradients.
            Only A and B matrices contribute to training cost.
        """
        return self.A.numel() + self.B.numel()
    
    def get_masked_params_ratio(self) -> Tuple[float, float]:
        """
        Get the ratio of masked (pruned) parameters in input and output dimensions.
        
        Returns:
            Tuple of (input_masked_ratio, output_masked_ratio)
        """
        in_ratio = 1.0 - (self.M_in.sum().item() / self.in_features)
        out_ratio = 1.0 - (self.M_out.sum().item() / self.out_features)
        return in_ratio, out_ratio
    
    def get_config(self) -> AdapterConfig:
        """
        Return the current configuration of this adapter.
        
        Returns:
            AdapterConfig object containing all configuration parameters
        """
        return AdapterConfig(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            alpha=self.alpha,
            device=self.device,
            dtype=self.dtype
        )
    
    def extra_repr(self) -> str:
        """Provide additional string representation for debugging."""
        in_masked, out_masked = self.get_masked_params_ratio()
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}, '
                f'input_pruned={in_masked:.3f}, output_pruned={out_masked:.3f}')
