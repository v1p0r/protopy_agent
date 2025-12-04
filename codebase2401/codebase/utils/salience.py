## utils/salience.py
import torch
import torch.nn as nn
from typing import Callable, Dict, Any, Optional, List
from dataclasses import dataclass
import numpy as np
from config import CONFIG

# Global storage for activations and gradients
# Key: module name, Value: tensor
_cached_activations: Dict[str, torch.Tensor] = {}
_cached_gradients: Dict[str, torch.Tensor] = {}

@dataclass
class SalienceConfig:
    """Configuration specific to salience computation."""
    use_kurtosis: bool = True
    compression_dims: tuple = (0, 1)  # dimensions to sum over (batch, seq)
    kurtosis_dim: int = -1           # dimension along which to compute kurtosis (features)
    epsilon: float = 1e-8            # small value for numerical stability

def compute_kurtosis(tensor: torch.Tensor, dim: int = -1, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute excess kurtosis of a tensor along specified dimension.
    
    Kurtosis measures the "tailedness" of the distribution. High-kurtosis units are more likely
    to contain outlier-sensitive information that should be preserved during pruning.
    
    Formula: kurtosis = μ₄/σ⁴ - 3
    
    Args:
        tensor: Input tensor of any shape
        dim: Dimension along which to compute kurtosis (default: last dimension)
        epsilon: Small value added to denominator for numerical stability
        
    Returns:
        Kurtosis tensor with `dim` reduced, excess kurtosis (subtracted 3)
        
    Note:
        This function preserves gradient flow for backpropagation if needed.
    """
    # Move dim to last for easier indexing
    if dim != -1:
        tensor = tensor.transpose(dim, -1)
    
    # Compute mean
    mean = tensor.mean(dim=-1, keepdim=True)
    
    # Compute deviations
    diffs = tensor - mean
    
    # Compute variance and standard deviation
    var = torch.mean(diffs ** 2, dim=-1, keepdim=True)
    std = torch.sqrt(var + epsilon)
    
    # Compute standardized fourth moment
    # Shape: [..., 1] after reduction
    z_scores = diffs / std
    fourth_moment = torch.mean(z_scores ** 4, dim=-1)
    
    # Excess kurtosis (subtract 3 so normal distribution has kurtosis 0)
    kurtosis = fourth_moment - 3.0
    
    return kurtosis

def compute_salience(H: torch.Tensor, 
                    grad_H: torch.Tensor,
                    use_kurtosis: bool = None,
                    compression_dims: tuple = None,
                    kurtosis_dim: int = None) -> torch.Tensor:
    """
    Compute outlier-aware salience score for parameter blocks.
    
    Implements the paper's formulation:
    S_block = (Σ|H|) * (Σ|∇H L|) * κ(H)
    
    Where:
    - Σ|H|: compressed activation magnitude
    - Σ|∇H L|: compressed gradient magnitude  
    - κ(H): excess kurtosis (optional)
    
    Args:
        H: Activations tensor of shape (*, D)
        grad_H: Gradients w.r.t. activations, same shape as H
        use_kurtosis: Whether to include kurtosis in scoring (default: from config)
        compression_dims: Dimensions to sum over for compression (default: (0,1))
        kurtosis_dim: Dimension along which to compute kurtosis (default: -1)
        
    Returns:
        Salience scores per feature/dimension/neuron of shape (D,)
        
    Raises:
        ValueError: If H and grad_H shapes don't match
        RuntimeError: If NaN values detected in computation
    """
    if H.shape != grad_H.shape:
        raise ValueError(f"Activation and gradient shapes mismatch: {H.shape} vs {grad_H.shape}")
    
    # Use defaults from config if not provided
    if use_kurtosis is None:
        use_kurtosis = CONFIG.pruning.use_kurtosis
    if compression_dims is None:
        compression_dims = CONFIG.pruning.salience.get('compression_dims', (0, 1))
    if kurtosis_dim is None:
        kurtosis_dim = CONFIG.pruning.salience.get('kurtosis_dim', -1)
    
    # Compress tensors by summing absolute values across specified dimensions
    # This reduces memory overhead while preserving relative importance
    act_norm = torch.sum(torch.abs(H), dim=compression_dims)  # Shape: (D,)
    grad_norm = torch.sum(torch.abs(grad_H), dim=compression_dims)  # Shape: (D,)
    
    # Initialize base salience score
    salience = act_norm * grad_norm
    
    # Optionally include kurtosis to preserve outlier-sensitive parameters
    if use_kurtosis:
        kurtosis_score = compute_kurtosis(H, dim=kurtosis_dim)
        
        # Average kurtosis across non-feature dimensions if necessary
        if kurtosis_score.dim() > 1:
            # Find dimensions to average over (all except feature dim)
            avg_dims = tuple(i for i in range(kurtosis_score.dim()) if i != kurtosis_dim)
            kurtosis_avg = kurtosis_score.mean(dim=avg_dims)
        else:
            kurtosis_avg = kurtosis_score
            
        # Ensure kurtosis_avg has same length as salience
        if kurtosis_avg.numel() != salience.numel():
            # Reshape or broadcast as needed
            kurtosis_avg = kurtosis_avg.view(-1)
            
        # Multiply kurtosis factor (clamp to avoid extreme values)
        kurtosis_factor = torch.clamp(kurtosis_avg + 3.0, min=0.1)  # Add 3 back, clamp minimum
        salience = salience * kurtosis_factor
    
    # Final validation
    if torch.isnan(salience).any():
        raise RuntimeError("NaN values detected in salience computation")
    if torch.isinf(salience).any():
        salience = torch.clamp(salience, max=torch.finfo(salience.dtype).max)
    
    return salience

def register_activation_hook(module: nn.Module, name: str):
    """
    Register forward hook to capture activations of a module.
    
    Args:
        module: PyTorch module to hook
        name: Unique identifier for this module (e.g., 'encoder.layer.0.attention.query')
    """
    def hook_fn(_, input_tensor, output_tensor):
        # Store the output (activation) in global cache
        # Detach from graph but preserve requires_grad for backward hooks
        if isinstance(output_tensor, torch.Tensor):
            _cached_activations[name] = output_tensor.detach()
        elif isinstance(output_tensor, (list, tuple)):
            # Handle multiple outputs - take first one (common case)
            _cached_activations[name] = output_tensor[0].detach()
    
    handle = module.register_forward_hook(hook_fn)
    return handle

def register_gradient_hook(module: nn.Module, name: str):
    """
    Register backward hook to capture gradients of a module's output.
    
    Args:
        module: PyTorch module to hook  
        name: Unique identifier for this module
    """
    def hook_fn(grad_output):
        # Store the gradient in global cache
        if isinstance(grad_output, torch.Tensor):
            _cached_gradients[name] = grad_output.detach()
        elif isinstance(grad_output, (list, tuple)):
            # Handle multiple gradients - take first one
            _cached_gradients[name] = grad_output[0].detach()
    
    # Register hook on module's output
    # We need to get the output tensor first
    if hasattr(module, '_forward_hooks'):
        # Wait until forward pass completes
        def register_backward_hook(_, __, output):
            if isinstance(output, torch.Tensor):
                output.register_hook(hook_fn)
            elif isinstance(output, (list, tuple)):
                output[0].register_hook(hook_fn)
                
        module.register_forward_hook(register_backward_hook)
    
    return None  # Cannot return handle easily here

def register_gradient_hooks(model: nn.Module, callback: Callable[[nn.Module, str], bool]):
    """
    Register gradient and activation hooks on selected modules of a model.
    
    Args:
        model: The PyTorch model to instrument
        callback: Function that takes (module, name) and returns True if hooks should be registered
        
    Returns:
        List of hook handles for later removal
    """
    handles = []
    
    def add_hooks(module: nn.Module, prefix: str = ''):
        # Get module name
        name = prefix if prefix else module.__class__.__name__
        
        # Check if this module should be hooked
        if callback(module, name):
            # Register both activation and gradient hooks
            act_handle = register_activation_hook(module, name)
            if act_handle is not None:
                handles.append(act_handle)
                
            # Gradient hook registration handled differently
            # We'll use the forward hook to attach backward hook
            def register_bw_hook(_, __, output):
                if isinstance(output, torch.Tensor):
                    if output.requires_grad:
                        grad_handle = output.register_hook(
                            lambda grad: _cached_gradients.update({name: grad.detach()})
                        )
                        handles.append(grad_handle)
                elif isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor) and out.requires_grad:
                            grad_handle = out.register_hook(
                                lambda grad: _cached_gradients.update({name: grad.detach()})
                            )
                            handles.append(grad_handle)
                            
            fw_handle = module.register_forward_hook(register_bw_hook)
            handles.append(fw_handle)
        
        # Recursively add hooks to children
        for child_name, child_module in module.named_children():
            child_prefix = f"{name}.{child_name}" if name else child_name
            add_hooks(child_module, child_prefix)
    
    add_hooks(model)
    return handles

def get_cached_activations() -> Dict[str, torch.Tensor]:
    """
    Retrieve all cached activations.
    
    Returns:
        Dictionary mapping module names to activation tensors
    """
    return _cached_activations.copy()

def get_cached_gradients() -> Dict[str, torch.Tensor]:
    """
    Retrieve all cached gradients.
    
    Returns:
        Dictionary mapping module names to gradient tensors
    """
    return _cached_gradients.copy()

def clear_cache():
    """
    Clear all cached activations and gradients to free memory.
    
    Should be called after each salience computation step.
    """
    _cached_activations.clear()
    _cached_gradients.clear()

def reset_cache():
    """
    Reset the global caches to empty dictionaries.
    
    Useful for cleanup between training phases.
    """
    global _cached_activations, _cached_gradients
    _cached_activations = {}
    _cached_gradients = {}

# Predefined callbacks for common use cases
def create_apt_target_callback(target_layers: List[str] = None) -> Callable[[nn.Module, str], bool]:
    """
    Create a callback function that identifies APT-targeted layers.
    
    Args:
        target_layers: List of layer types to target (default from config)
        
    Returns:
        Callable that returns True if module should be hooked
    """
    if target_layers is None:
        target_layers = CONFIG.pruning.apply_to
    
    def callback(module: nn.Module, name: str) -> bool:
        # Check if this module type is in target list
        if any(tgt in name.lower() for tgt in target_layers):
            return True
        return False
        
    return callback

__all__ = [
    'compute_kurtosis',
    'compute_salience',
    'register_activation_hook',
    'register_gradient_hook',
    'register_gradient_hooks',
    'get_cached_activations',
    'get_cached_gradients',
    'clear_cache',
    'reset_cache',
    'create_apt_target_callback'
]
