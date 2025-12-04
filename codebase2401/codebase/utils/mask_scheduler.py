## utils/mask_scheduler.py
import torch
from typing import Optional
from config import CONFIG

def cubic_schedule(
    current_step: int,
    total_steps: int,
    initial_sparsity: float = 0.0,
    final_sparsity: Optional[float] = None
) -> float:
    """
    Compute the target sparsity level using a cubic scheduling function.
    
    Implements the paper's formulation: s(t) = s_target * (t / T)^3
    This gradually increases sparsity over training, delaying aggressive pruning
    until later stages for better stability and performance.
    
    Args:
        current_step: Current global training step (0-indexed)
        total_steps: Total number of training steps in the schedule
        initial_sparsity: Starting sparsity ratio (default: 0.0)
        final_sparsity: Target sparsity ratio at end of training.
                       If None, uses value from config (apt.target_sparsity)
    
    Returns:
        Sparsity ratio between 0.0 and 1.0 representing the proportion of 
        parameters to be pruned at this step.
        
    Raises:
        ValueError: If inputs are invalid (negative steps, out-of-range sparsity)
        
    Example:
        >>> cubic_schedule(50, 100, final_sparsity=0.6)
        0.075  # Only 7.5% sparse at midpoint
    """
    # Validate inputs
    if current_step < 0:
        raise ValueError(f"current_step must be non-negative, got {current_step}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if current_step > total_steps:
        raise ValueError(f"current_step ({current_step}) cannot exceed total_steps ({total_steps})")
    if not (0.0 <= initial_sparsity <= 1.0):
        raise ValueError(f"initial_sparsity must be in [0,1], got {initial_sparsity}")
    
    # Use config value if final_sparsity not provided
    if final_sparsity is None:
        final_sparsity = CONFIG.apt.target_sparsity
    
    if not (0.0 <= final_sparsity <= 1.0):
        raise ValueError(f"final_sparsity must be in [0,1], got {final_sparsity}")
    
    # Compute progress ratio
    progress_ratio = current_step / total_steps
    
    # Apply cubic interpolation
    sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (progress_ratio ** 3)
    
    # Clamp to valid range
    sparsity = max(0.0, min(1.0, sparsity))
    
    return sparsity


def decay_mask_values(
    mask: torch.Tensor,
    decay_step: Optional[float] = None,
    min_value: float = 0.0
) -> torch.Tensor:
    """
    Gradually decay mask values to improve training stability during pruning.
    
    Instead of abruptly setting mask values to 0 when pruning blocks, this function
    decreases them by a small fixed amount per update. This prevents sudden changes
    in gradient flow and model output, aligning with the paper's approach:
    
    "gradually decrease the pruning masks of pruned blocks by α instead of instantly 
     setting them from ones to zeros."
    
    Args:
        mask: Input mask tensor (can be bool or float). Will be converted to float
              for computation and returned as float.
        decay_step: Amount to subtract from each element. If None, uses value from
                   config (pruning.mask_decay_step).
        min_value: Minimum allowed value after decay (typically 0.0).
    
    Returns:
        New mask tensor with values decayed by decay_step and clamped to min_value.
        The returned tensor is always floating-point type.
        
    Note:
        This function does not modify the input mask in-place. For efficiency in
        high-frequency calls, consider in-place operations externally.
        
    Example:
        >>> mask = torch.ones(10)
        >>> new_mask = decay_mask_values(mask, decay_step=0.01)
        >>> (new_mask == 0.99).all()
        True
    """
    # Use default decay step from config if not provided
    if decay_step is None:
        decay_step = CONFIG.pruning.mask_decay_step
    
    # Validate inputs
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"mask must be a torch.Tensor, got {type(mask)}")
    if decay_step < 0:
        raise ValueError(f"decay_step must be non-negative, got {decay_step}")
    if min_value < 0:
        raise ValueError(f"min_value must be >= 0, got {min_value}")
    if min_value > 1:
        raise ValueError(f"min_value must be <= 1, got {min_value}")
    
    # Ensure mask is float for arithmetic operations
    mask_float = mask.float()
    
    # Apply decay and clamp
    updated_mask = torch.clamp(mask_float - decay_step, min=min_value)
    
    return updated_mask


def linear_weight_schedule(
    current_step: int,
    total_steps: int,
    start_weight: float = 0.0,
    end_weight: float = 1.0
) -> float:
    """
    Compute a linearly interpolated weight for combining multiple loss terms.
    
    Used primarily for knowledge distillation where the coefficient λ_t linearly
    increases from 0 to 1 over training:
    
    L_total = (1 - λ_t)L_SFT + λ_t L_KD
    
    This allows early focus on supervised fine-tuning and gradual transition to
    knowledge distillation as the teacher stabilizes.
    
    Args:
        current_step: Current global training step (0-indexed)
        total_steps: Total number of training steps in the schedule
        start_weight: Initial weight value at step 0
        end_weight: Final weight value at last step
    
    Returns:
        Interpolated weight value between start_weight and end_weight.
        
    Raises:
        ValueError: If inputs are invalid (negative steps, etc.)
        
    Example:
        >>> linear_weight_schedule(50, 100, start_weight=0.0, end_weight=1.0)
        0.5  # Halfway through schedule
    """
    # Validate inputs
    if current_step < 0:
        raise ValueError(f"current_step must be non-negative, got {current_step}")
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if current_step > total_steps:
        raise ValueError(f"current_step ({current_step}) cannot exceed total_steps ({total_steps})")
    
    # Compute progress ratio
    progress_ratio = current_step / total_steps
    
    # Linear interpolation
    weight = start_weight + (end_weight - start_weight) * progress_ratio
    
    return weight


__all__ = [
    'cubic_schedule',
    'decay_mask_values',
    'linear_weight_schedule'
]
