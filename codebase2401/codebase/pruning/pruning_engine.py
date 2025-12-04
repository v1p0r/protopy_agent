## pruning/pruning_engine.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from config import CONFIG
from utils.salience import compute_salience, get_cached_activations, get_cached_gradients, clear_cache
from utils.mask_scheduler import cubic_schedule
from model.apt_adapter import APTAdapter

logger = logging.getLogger(__name__)

@dataclass
class BlockInfo:
    """Information about a prunable block in the model."""
    name: str
    layer_ref: APTAdapter
    block_type: str  # "attention_head", "ffn_neuron", "hidden_dimension"
    param_count: int
    dim_slice: Tuple[int, int]  # (start_idx, end_idx) in the dimension
    salience_score: Optional[float] = None
    salience_density: Optional[float] = None

class PruningEngine:
    """
    Pruning engine for Adaptive Pruning and Tuning (APT) framework.
    
    This module implements structured pruning of Transformer components including:
    - Attention heads
    - FFN neurons
    - Hidden dimensions
    
    It uses outlier-aware salience scoring to identify less important blocks and
    gradually prunes them during training according to a cubic schedule.
    
    Key features:
    - Computes salience scores using activation-gradient product and kurtosis
    - Uses binary search to satisfy sparsity constraints efficiently
    - Gradually decays masks for training stability
    - Dynamically updates block information as model changes
    
    Based on paper's formulation in sec:apt-prune and alg:epa.
    """
    
    def __init__(self, model: nn.Module, total_steps: int):
        """
        Initialize the pruning engine with a model and training configuration.
        
        Args:
            model: The PyTorch model containing APT adapters
            total_steps: Total number of training steps for scheduling
            
        Raises:
            ValueError: If model has no APT adapters or invalid configuration
        """
        self.model = model
        self.total_steps = total_steps
        
        # Configuration from global config
        self.target_sparsity = CONFIG.apt.target_sparsity
        self.sparsity_schedule = CONFIG.apt.sparsity_schedule
        self.update_frequency = CONFIG.apt.update_frequency
        self.mask_decay_step = CONFIG.pruning.mask_decay_step
        self.use_kurtosis = CONFIG.pruning.use_kurtosis
        self.apply_to_layers = CONFIG.pruning.apply_to
        
        # Internal state
        self.blocks: List[BlockInfo] = []
        self.layer_to_blocks: Dict[str, List[BlockInfo]] = {}
        self.current_sparsity = 0.0
        self.hook_handles = []
        
        # Validate configuration
        if not self.apply_to_layers:
            raise ValueError("No layers specified for pruning in config.pruning.apply_to")
            
        if self.target_sparsity < 0 or self.target_sparsity >= 1.0:
            raise ValueError(f"target_sparsity must be in [0,1), got {self.target_sparsity}")
            
        if self.mask_decay_step <= 0:
            raise ValueError(f"mask_decay_step must be positive, got {self.mask_decay_step}")
        
        # Scan model and register hooks
        self._scan_model()
        self._register_hooks()
        
        logger.info(f"PruningEngine initialized with {len(self.blocks)} prunable blocks")
    
    def _scan_model(self) -> None:
        """
        Scan the model to identify all APT adapters and create block registry.
        
        Populates self.blocks with BlockInfo objects for all prunable units.
        Also builds layer-to-blocks mapping for efficient updates.
        """
        self.blocks.clear()
        self.layer_to_blocks.clear()
        
        def find_apt_adapters(module: nn.Module, prefix: str = ''):
            # Check if this module is an APT adapter
            if isinstance(module, APTAdapter):
                layer_name = prefix
                
                # Determine which types of blocks to extract based on layer type
                block_types = []
                if any(tgt in layer_name.lower() for tgt in ['query', 'value']):
                    block_types.append('attention_head')
                if any(tgt in layer_name.lower() for tgt in ['ffn', 'intermediate']):
                    block_types.append('ffn_neuron')
                
                # Always consider hidden dimension for all APT-adapted layers
                block_types.append('hidden_dimension')
                
                # Create blocks for each type
                for block_type in block_types:
                    self._create_blocks_for_layer(module, layer_name, block_type)
                    
                # Register this layer
                if layer_name not in self.layer_to_blocks:
                    self.layer_to_blocks[layer_name] = []
                    
            # Recursively scan children
            for child_name, child_module in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                find_apt_adapters(child_module, child_prefix)
        
        find_apt_adapters(self.model)
        
        # Sort blocks by name for deterministic ordering
        self.blocks.sort(key=lambda x: x.name)
        
        # Group blocks by layer for efficient access
        for block in self.blocks:
            if block.layer_ref not in self.layer_to_blocks:
                self.layer_to_blocks[block.layer_ref] = []
            self.layer_to_blocks[block.layer_ref].append(block)
    
    def _create_blocks_for_layer(self, adapter: APTAdapter, layer_name: str, block_type: str) -> None:
        """
        Create block entries for a specific layer and block type.
        
        Args:
            adapter: Reference to the APT adapter
            layer_name: Name of the layer
            block_type: Type of block to create ("attention_head", "ffn_neuron", "hidden_dimension")
        """
        if block_type == "attention_head":
            # For attention layers, assume heads are evenly distributed
            head_dim = adapter.out_features // 12  # Assume 12 heads for RoBERTa/T5
            num_heads = adapter.out_features // head_dim
            
            for i in range(num_heads):
                start_idx = i * head_dim
                end_idx = (i + 1) * head_dim
                
                # Estimate parameter count: includes both B and A matrices for this head
                # B matrix: head_dim rows, A matrix: head_dim columns in input
                param_count_b = head_dim * adapter.rank
                param_count_a = adapter.rank * min(head_dim, adapter.in_features)  # Handle case where head_dim > in_features
                total_params = param_count_b + param_count_a
                
                block_name = f"{layer_name}.head_{i}"
                block = BlockInfo(
                    name=block_name,
                    layer_ref=adapter,
                    block_type=block_type,
                    param_count=total_params,
                    dim_slice=(start_idx, end_idx)
                )
                self.blocks.append(block)
                
        elif block_type == "ffn_neuron":
            # For FFN layers, treat intermediate dimension as neurons
            # Assume intermediate size is out_features (for down_proj) or in_features (for up_proj)
            # Use smaller dimension to avoid overcounting
            neuron_size = min(adapter.in_features, adapter.out_features)
            
            for i in range(neuron_size):
                # Each neuron affects one dimension in projection
                param_count = adapter.rank * 2  # One parameter in A, one in B per rank
                block_name = f"{layer_name}.neuron_{i}"
                block = BlockInfo(
                    name=block_name,
                    layer_ref=adapter,
                    block_type=block_type,
                    param_count=param_count,
                    dim_slice=(i, i + 1)
                )
                self.blocks.append(block)
                
        elif block_type == "hidden_dimension":
            # Hidden dimension affects both input and output dimensions
            # Create blocks for both dimensions
            for dim_type, dim_size in [("in", adapter.in_features), ("out", adapter.out_features)]:
                for i in range(dim_size):
                    # Parameter count depends on whether this is input or output dimension
                    if dim_type == "in":
                        # Input dimension affects A matrix
                        param_count = adapter.rank
                    else:
                        # Output dimension affects B matrix  
                        param_count = adapter.rank
                        
                    block_name = f"{layer_name}.hidden_dim_{dim_type}_{i}"
                    block = BlockInfo(
                        name=block_name,
                        layer_ref=adapter,
                        block_type=block_type,
                        param_count=param_count,
                        dim_slice=(i, i + 1)
                    )
                    self.blocks.append(block)
    
    def _register_hooks(self) -> None:
        """
        Register forward and backward hooks to capture activations and gradients.
        
        Uses the utility functions from utils/salience.py to register hooks on
        APT adapter layers that are targeted for pruning.
        """
        # Clear any existing handles
        self._remove_hooks()
        
        # Create callback to identify target layers
        def hook_callback(module: nn.Module, name: str) -> bool:
            if isinstance(module, APTAdapter):
                return any(tgt in name.lower() for tgt in self.apply_to_layers)
            return False
        
        # Register hooks using utility function
        self.hook_handles = []
        try:
            self.hook_handles = register_gradient_hooks(self.model, hook_callback)
            logger.debug(f"Registered {len(self.hook_handles)} hooks for salience computation")
        except Exception as e:
            logger.error(f"Failed to register hooks: {e}")
            raise
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks to prevent memory leaks."""
        for handle in self.hook_handles:
            try:
                handle.remove()
            except (AttributeError, KeyError):
                pass  # Handle might already be removed
        self.hook_handles.clear()
    
    def compute_salience(self) -> Dict[str, float]:
        """
        Compute outlier-aware salience scores for all prunable blocks.
        
        Implements the paper's formulation:
        S_block = (Σ|H|) * (Σ|∇H L|) * κ(H)
        
        Where κ(H) is excess kurtosis for outlier sensitivity.
        
        Returns:
            Dictionary mapping block names to their salience scores
            
        Raises:
            RuntimeError: If activations or gradients are not available
        """
        # Get cached activations and gradients
        activations = get_cached_activations()
        gradients = get_cached_gradients()
        
        if not activations:
            raise RuntimeError("No activations captured. Ensure forward pass completed.")
        if not gradients:
            raise RuntimeError("No gradients captured. Ensure backward pass completed.")
        
        # Clear cache after use
        clear_cache()
        
        # Compute salience for each block
        salience_dict = {}
        
        for block in self.blocks:
            layer_name = self._get_layer_name(block.layer_ref)
            
            if layer_name not in activations or layer_name not in gradients:
                logger.warning(f"Missing activation/gradient for {layer_name}, skipping block {block.name}")
                salience_dict[block.name] = 0.0
                continue
                
            H = activations[layer_name]
            grad_H = gradients[layer_name]
            
            # Extract relevant slice based on block type and dim_slice
            if block.block_type == "attention_head" or block.block_type == "hidden_dimension":
                # For output dimension blocks
                if len(H.shape) >= 2:
                    # Handle different tensor shapes
                    if H.shape[-1] == block.layer_ref.out_features:
                        # Slice along last dimension (output)
                        start, end = block.dim_slice
                        H_slice = H[..., start:end]
                        grad_H_slice = grad_H[..., start:end]
                    else:
                        # Fallback: use full tensor
                        H_slice = H
                        grad_H_slice = grad_H
                else:
                    H_slice = H
                    grad_H_slice = grad_H
            else:
                # For input dimension blocks (FFN neurons)
                if len(H.shape) >= 2:
                    if H.shape[-1] == block.layer_ref.in_features:
                        start, end = block.dim_slice
                        H_slice = H[..., start:end]
                        grad_H_slice = grad_H[..., start:end]
                    else:
                        H_slice = H
                        grad_H_slice = grad_H
                else:
                    H_slice = H
                    grad_H_slice = grad_H
            
            try:
                # Compute salience using utility function
                salience_score = compute_salience(
                    H_slice, 
                    grad_H_slice, 
                    use_kurtosis=self.use_kurtosis
                )
                
                # Reduce to scalar if needed
                if salience_score.numel() > 1:
                    salience_value = salience_score.mean().item()
                else:
                    salience_value = salience_score.item()
                    
            except Exception as e:
                logger.warning(f"Failed to compute salience for {block.name}: {e}")
                salience_value = 0.0
            
            salience_dict[block.name] = salience_value
            block.salience_score = salience_value
        
        return salience_dict
    
    def _get_layer_name(self, adapter: APTAdapter) -> str:
        """
        Get the name of the layer corresponding to an APT adapter.
        
        This is a simplified implementation - in practice, you might need
        a more sophisticated way to map adapters to layer names.
        
        Args:
            adapter: APT adapter instance
            
        Returns:
            Layer name as string
        """
        # In a real implementation, you would have a proper mapping
        # For now, return a placeholder
        for name, module in self.model.named_modules():
            if module is adapter:
                return name
        return "unknown_layer"
    
    def calculate_total_params(self) -> int:
        """
        Calculate the total number of parameters in the current model configuration.
        
        Includes both frozen backbone parameters and trainable APT adapter parameters.
        
        Returns:
            Total parameter count
        """
        total_params = 0
        
        # Count parameters in APT adapters
        for block in self.blocks:
            # Only count active blocks (those not fully pruned)
            if self._is_block_active(block):
                total_params += block.param_count
        
        # Add parameters from non-APT parts of the model
        # This is an approximation - in practice, you'd traverse the full model
        for name, param in self.model.named_parameters():
            if not any(f"adapter.{k}" in name for k in ["A", "B", "M_in", "M_out"]):
                total_params += param.numel()
                
        return total_params
    
    def _is_block_active(self, block: BlockInfo) -> bool:
        """
        Check if a block is still active (not fully pruned).
        
        Args:
            block: BlockInfo object
            
        Returns:
            True if block is still contributing to computation
        """
        mask_in = getattr(block.layer_ref, 'M_in', None)
        mask_out = getattr(block.layer_ref, 'M_out', None)
        
        if block.block_type in ["attention_head", "hidden_dimension"]:
            if mask_out is not None and len(mask_out) > block.dim_slice[0]:
                start, end = block.dim_slice
                return mask_out[start:end].any().item()
        else:
            if mask_in is not None and len(mask_in) > block.dim_slice[0]:
                start, end = block.dim_slice
                return mask_in[start:end].any().item()
                
        return True
    
    def select_blocks_to_prune(self, target_sparsity: float) -> List[str]:
        """
        Select blocks to prune using binary search over salience density.
        
        Implements the latency-saliency knapsack problem solution described in the paper.
        Finds the maximum number of highest-salience-density blocks that can be retained
        while satisfying the parameter budget constraint.
        
        Args:
            target_sparsity: Target sparsity level (0.0 to 1.0)
            
        Returns:
            List of block names to keep (others will be pruned)
        """
        if not self.blocks:
            return []
            
        # Calculate current total parameters and target
        current_total = self.calculate_total_params()
        target_params = int(current_total * (1.0 - target_sparsity))
        
        # Compute salience density for each block
        for block in self.blocks:
            if block.salience_score is None:
                block.salience_score = 0.0
                
            if block.param_count > 0:
                block.salience_density = block.salience_score / block.param_count
            else:
                block.salience_density = 0.0
        
        # Sort blocks by salience density (descending)
        sorted_blocks = sorted(self.blocks, key=lambda x: x.salience_density, reverse=True)
        
        # Binary search for maximum number of blocks to retain
        left, right = 0, len(sorted_blocks)
        best_retain_count = 0
        
        while left <= right:
            mid = (left + right) // 2
            retained_blocks = sorted_blocks[:mid]
            
            # Calculate parameter count if these blocks are retained
            retained_params = sum(block.param_count for block in retained_blocks)
            
            if retained_params <= target_params:
                best_retain_count = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # Return names of blocks to keep
        keep_blocks = sorted_blocks[:best_retain_count]
        keep_names = [block.name for block in keep_blocks]
        
        # Update current sparsity
        if current_total > 0:
            self.current_sparsity = 1.0 - (sum(b.param_count for b in keep_blocks) / current_total)
        
        logger.debug(f"Selected {best_retain_count}/{len(self.blocks)} blocks to keep "
                    f"(target sparsity: {target_sparsity:.3f}, actual: {self.current_sparsity:.3f})")
        
        return keep_names
    
    def update_masks(self, blocks_to_keep: List[str], gradual: bool = True) -> None:
        """
        Update pruning masks based on selected blocks.
        
        Applies gradual mask decay for training stability as described in the paper:
        "gradually decrease the pruning masks of pruned blocks by α instead of instantly setting them from ones to zeros."
        
        Args:
            blocks_to_keep: List of block names to retain
            gradual: Whether to apply gradual decay or abrupt pruning
        """
        # Group blocks by layer for batch updates
        blocks_by_layer = {}
        for block in self.blocks:
            layer_id = id(block.layer_ref)
            if layer_id not in blocks_by_layer:
                blocks_by_layer[layer_id] = []
            blocks_by_layer[layer_id].append(block)
        
        # Update masks for each layer
        for layer_id, layer_blocks in blocks_by_layer.items():
            # Find the adapter for this layer group
            adapter = None
            for block in layer_blocks:
                adapter = block.layer_ref
                break
                
            if adapter is None:
                continue
                
            # Prepare new masks
            new_M_in = adapter.M_in.data.float().clone()
            new_M_out = adapter.M_out.data.float().clone()
            
            # Update masks based on blocks to keep
            for block in layer_blocks:
                should_keep = block.name in blocks_to_keep
                
                if block.block_type in ["attention_head", "hidden_dimension"]:
                    # Output dimension mask
                    start, end = block.dim_slice
                    if end <= len(new_M_out):
                        if should_keep:
                            new_M_out[start:end] = 1.0
                        else:
                            if gradual:
                                # Gradual decay
                                new_M_out[start:end] = torch.clamp(
                                    new_M_out[start:end] - self.mask_decay_step, 
                                    min=0.0
                                )
                            else:
                                new_M_out[start:end] = 0.0
                else:
                    # Input dimension mask (FFN neurons)
                    start, end = block.dim_slice
                    if end <= len(new_M_in):
                        if should_keep:
                            new_M_in[start:end] = 1.0
                        else:
                            if gradual:
                                # Gradual decay  
                                new_M_in[start:end] = torch.clamp(
                                    new_M_in[start:end] - self.mask_decay_step, 
                                    min=0.0
                                )
                            else:
                                new_M_in[start:end] = 0.0
            
            # Apply updated masks (convert back to bool)
            adapter.M_in.data = (new_M_in >= 0.5).to(torch.bool)
            adapter.M_out.data = (new_M_out >= 0.5).to(torch.bool)
    
    def adaptive_update(self, global_step: int) -> Dict[str, Any]:
        """
        Perform adaptive pruning update at specified training step.
        
        This is the main interface called by the trainer. It:
        1. Computes salience scores
        2. Determines target sparsity via scheduling
        3. Selects blocks to prune
        4. Updates masks gradually
        
        Args:
            global_step: Current global training step
            
        Returns:
            Dictionary with pruning statistics
        """
        # Skip if not time for update
        if global_step % self.update_frequency != 0:
            return {
                "step": global_step,
                "update_performed": False,
                "current_sparsity": self.current_sparsity,
                "target_sparsity": self.target_sparsity
            }
        
        try:
            # Compute current target sparsity using schedule
            if self.sparsity_schedule == "cubic":
                target_sparsity = cubic_schedule(
                    current_step=global_step,
                    total_steps=self.total_steps,
                    final_sparsity=self.target_sparsity
                )
            else:
                # Default to linear if unknown schedule
                progress = global_step / max(self.total_steps, 1)
                target_sparsity = self.target_sparsity * progress
                
            # Compute salience scores
            salience_scores = self.compute_salience()
            
            # Select blocks to keep
            blocks_to_keep = self.select_blocks_to_prune(target_sparsity)
            
            # Update masks
            self.update_masks(blocks_to_keep, gradual=True)
            
            # Collect statistics
            stats = {
                "step": global_step,
                "update_performed": True,
                "current_sparsity": self.current_sparsity,
                "target_sparsity": target_sparsity,
                "blocks_considered": len(self.blocks),
                "blocks_kept": len(blocks_to_keep),
                "sparsity_error": abs(self.current_sparsity - target_sparsity),
                "mean_salience": float(torch.tensor(list(salience_scores.values())).mean()),
                "std_salience": float(torch.tensor(list(salience_scores.values())).std())
            }
            
            logger.info(f"Pruning update at step {global_step}: "
                       f"sparsity={self.current_sparsity:.3f} (target={target_sparsity:.3f}), "
                       f"kept {len(blocks_to_keep)}/{len(self.blocks)} blocks")
            
            return stats
            
        except Exception as e:
            logger.error(f"Pruning update failed at step {global_step}: {e}")
            return {
                "step": global_step,
                "update_performed": False,
                "error": str(e),
                "current_sparsity": self.current_sparsity
            }
    
    def get_current_sparsity(self) -> float:
        """
        Get the current sparsity level of the model.
        
        Returns:
            Current sparsity ratio (0.0 to 1.0)
        """
        return self.current_sparsity
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current pruning state.
        
        Returns:
            Dictionary with comprehensive pruning information
        """
        # Count active vs total parameters in adapters
        total_adapter_params = 0
        active_adapter_params = 0
        
        for block in self.blocks:
            total_adapter_params += block.param_count
            if self._is_block_active(block):
                active_adapter_params += block.param_count
        
        return {
            "total_blocks": len(self.blocks),
            "active_blocks": sum(1 for b in self.blocks if self._is_block_active(b)),
            "total_adapter_params": total_adapter_params,
            "active_adapter_params": active_adapter_params,
            "adapter_sparsity": 1.0 - (active_adapter_params / max(total_adapter_params, 1)),
            "current_sparsity": self.current_sparsity,
            "target_sparsity": self.target_sparsity,
            "update_frequency": self.update_frequency
        }
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self._remove_hooks()

# Import at bottom to avoid circular imports
from utils.salience import register_gradient_hooks

__all__ = ['PruningEngine']
