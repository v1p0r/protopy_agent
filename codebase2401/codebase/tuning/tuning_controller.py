## tuning/tuning_controller.py
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
import math
from config import CONFIG
from model.apt_adapter import APTAdapter

logger = logging.getLogger(__name__)

@dataclass
class AdapterInfo:
    """Information about an APT adapter for tuning decisions."""
    name: str
    adapter_ref: APTAdapter
    salience_score: float = 0.0
    param_count: int = 0
    current_rank: int = 8

class TuningController:
    """
    Adaptive Tuning Controller for the APT framework.
    
    This module dynamically increases the rank of APT adapters in selected layers based on their 
    importance during training. It implements the paper's adaptive tuning strategy (𝒜_T) that:
    - Computes per-adapter importance via parameter-level salience
    - Selects top-half most important adapters
    - Increases their ranks while respecting total tuning budget
    
    Key features:
    - Uses weight-gradient product for salience scoring
    - Supports exponential moving average (EMA) for stable importance tracking
    - Enforces maximum tuning parameter ratio constraint
    - Implements stable parameter initialization when increasing rank
    - Integrates with optimizer state management
    
    Based on paper formulation in sec:apt-tune and alg:epa.
    """
    
    def __init__(self, model: nn.Module, total_steps: int):
        """
        Initialize the tuning controller with a model and training configuration.
        
        Args:
            model: The PyTorch model containing APT adapters
            total_steps: Total number of training steps for scheduling
            
        Raises:
            ValueError: If no APT adapters found or invalid configuration
        """
        self.model = model
        self.total_steps = total_steps
        
        # Configuration from global config
        self.initial_rank = CONFIG.apt.initial_rank
        self.alpha_scaling = CONFIG.apt.alpha_scaling
        self.update_frequency = CONFIG.apt.update_frequency
        self.max_tuning_params_ratio = CONFIG.apt.max_tuning_params_ratio
        self.ema_beta = 0.9  # For smoothing salience scores over time (not specified in paper)
        
        # Internal state
        self.adapters: List[AdapterInfo] = []
        self.running_salience: Dict[str, float] = {}
        self.current_total_tuning_params = 0
        self.step_counter = 0
        
        # Validate configuration
        if self.initial_rank <= 0:
            raise ValueError(f"initial_rank must be positive, got {self.initial_rank}")
        if not (0.0 < self.max_tuning_params_ratio <= 1.0):
            raise ValueError(f"max_tuning_params_ratio must be in (0,1], got {self.max_tuning_params_ratio}")
        if self.update_frequency <= 0:
            raise ValueError(f"update_frequency must be positive, got {self.update_frequency}")
            
        # Scan model to discover APT adapters
        self._scan_model()
        
        # Estimate base model size for budget calculation
        self.base_model_size = self._estimate_base_model_size()
        
        logger.info(f"TuningController initialized with {len(self.adapters)} APT adapters. "
                   f"Base model estimated at {self.base_model_size:,} parameters.")
    
    def _scan_model(self) -> None:
        """
        Scan the model to identify all APT adapters and create adapter registry.
        
        Populates self.adapters with AdapterInfo objects for all tunable units.
        Also initializes running salience estimates.
        """
        self.adapters.clear()
        self.running_salience.clear()
        
        def find_apt_adapters(module: nn.Module, prefix: str = ''):
            # Check if this module is an APT adapter
            if isinstance(module, APTAdapter):
                layer_name = prefix
                
                # Create adapter info
                param_count = module.get_params_count()
                adapter_info = AdapterInfo(
                    name=layer_name,
                    adapter_ref=module,
                    param_count=param_count,
                    current_rank=module.rank
                )
                self.adapters.append(adapter_info)
                
                # Initialize running salience
                self.running_salience[layer_name] = 0.0
                
            # Recursively scan children
            for child_name, child_module in module.named_children():
                child_prefix = f"{prefix}.{child_name}" if prefix else child_name
                find_apt_adapters(child_module, child_prefix)
        
        find_apt_adapters(self.model)
        
        # Sort adapters by name for deterministic ordering
        self.adapters.sort(key=lambda x: x.name)
        
        # Validate we found some adapters
        if len(self.adapters) == 0:
            raise ValueError("No APT adapters found in model. Ensure model has been properly wrapped.")
    
    def _estimate_base_model_size(self) -> int:
        """
        Estimate the total number of parameters in the base model (excluding APT adapters).
        
        This is used to compute the absolute limit on tuning parameters via max_tuning_params_ratio.
        
        Returns:
            Estimated total parameter count of the base model
        """
        total_params = 0
        adapter_params = 0
        
        for name, param in self.model.named_parameters():
            if any(k in name for k in ["adapter.A", "adapter.B", "adapter.M_in", "adapter.M_out"]):
                adapter_params += param.numel()
            else:
                total_params += param.numel()
                
        logger.debug(f"Base model parameters: {total_params:,}, APT adapter parameters: {adapter_params:,}")
        return total_params
    
    def compute_adapter_importance(self) -> Dict[str, float]:
        """
        Compute importance scores for all APT adapters using parameter-level salience.
        
        Implements the paper's formulation:
        S_adapter = Σ ||W_A * ∇W_A L|| + ||W_B * ∇W_B L||
        
        Where gradients are available since these are tuning parameters.
        
        Returns:
            Dictionary mapping adapter names to their importance scores
            
        Raises:
            RuntimeError: If gradients are not available (backward pass not completed)
        """
        importance_scores = {}
        
        for adapter_info in self.adapters:
            adapter = adapter_info.adapter_ref
            name = adapter_info.name
            
            # Check if gradients are available
            if adapter.A.grad is None or adapter.B.grad is None:
                logger.warning(f"Gradients not available for {name}. Skipping in this update.")
                importance_scores[name] = 0.0
                continue
                
            try:
                # Compute salience for W_A: ||W_A ⊙ ∇W_A||
                salience_A = torch.sum(torch.abs(adapter.A.data * adapter.A.grad)).item()
                
                # Compute salience for W_B: ||W_B ⊙ ∇W_B||
                salience_B = torch.sum(torch.abs(adapter.B.data * adapter.B.grad)).item()
                
                # Aggregate adapter-level importance
                total_salience = salience_A + salience_B
                
                # Apply EMA for stability (similar to AdaLoRA)
                if name in self.running_salience:
                    self.running_salience[name] = (
                        self.ema_beta * self.running_salience[name] + 
                        (1.0 - self.ema_beta) * total_salience
                    )
                else:
                    self.running_salience[name] = total_salience
                    
                importance_scores[name] = self.running_salience[name]
                adapter_info.salience_score = self.running_salience[name]
                
            except Exception as e:
                logger.warning(f"Failed to compute importance for {name}: {e}")
                importance_scores[name] = 0.0
        
        # Update current total tuning parameter count
        self.current_total_tuning_params = sum(ai.param_count for ai in self.adapters)
        
        return importance_scores
    
    def _get_available_budget(self) -> int:
        """
        Calculate the number of additional tuning parameters allowed under the budget constraint.
        
        Budget is defined as: max_tuning_params = base_model_size * max_tuning_params_ratio
        
        Returns:
            Number of additional parameters that can be allocated
        """
        max_allowed_params = int(self.base_model_size * self.max_tuning_params_ratio)
        available_budget = max_allowed_params - self.current_total_tuning_params
        
        # Can't go negative
        return max(0, available_budget)
    
    def grow_ranks(self, importance_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Increase ranks of top-salient adapters according to paper's adaptive tuning strategy.
        
        Implements:
        1. Sort adapters by importance descending
        2. Select top-half adapters
        3. Distribute available budget uniformly among them
        4. Expand adapter ranks with stable initialization
        
        Args:
            importance_scores: Dictionary of adapter importance scores from compute_adapter_importance()
            
        Returns:
            Dictionary with statistics about the rank growth operation
        """
        if not importance_scores:
            logger.warning("No importance scores provided. Skipping rank growth.")
            return {
                "rank_growth_performed": False,
                "adapters_modified": 0,
                "total_new_params": 0,
                "budget_remaining": self._get_available_budget()
            }
        
        # Sort adapters by importance
        sorted_adapters = sorted(
            self.adapters, 
            key=lambda x: importance_scores.get(x.name, 0.0), 
            reverse=True
        )
        
        # Select top-half adapters
        num_to_grow = max(1, len(sorted_adapters) // 2)  # At least one
        top_adapters = sorted_adapters[:num_to_grow]
        
        # Calculate available budget
        available_budget = self._get_available_budget()
        
        if available_budget <= 0:
            logger.info("Tuning parameter budget exhausted. No rank growth performed.")
            return {
                "rank_growth_performed": False,
                "adapters_modified": 0,
                "total_new_params": 0,
                "budget_remaining": 0,
                "reason": "budget_exhausted"
            }
        
        # Distribute budget uniformly among top adapters
        # Each adapter will get approximately the same number of new parameters
        avg_params_per_adapter = available_budget // num_to_grow
        
        total_new_params = 0
        modified_adapters = 0
        
        for adapter_info in top_adapters:
            adapter = adapter_info.adapter_ref
            name = adapter_info.name
            
            # Calculate how many ranks we can add
            # Adding delta_r ranks adds delta_r * (in_features + out_features) parameters
            params_per_rank = adapter.in_features + adapter.out_features
            if params_per_rank == 0:
                continue
                
            max_delta_r = avg_params_per_adapter // params_per_rank
            actual_delta_r = min(max_delta_r, 64)  # Limit step size to avoid huge jumps
            
            if actual_delta_r > 0:
                try:
                    # Store old parameter count
                    old_param_count = adapter.get_params_count()
                    
                    # Increase rank with stable initialization
                    adapter.increase_rank(actual_delta_r)
                    
                    # Update adapter info
                    adapter_info.current_rank = adapter.rank
                    adapter_info.param_count = adapter.get_params_count()
                    
                    # Update total count
                    total_new_params += (adapter_info.param_count - old_param_count)
                    modified_adapters += 1
                    
                    logger.debug(f"Increased rank of {name} by {actual_delta_r} (new rank={adapter.rank})")
                    
                except Exception as e:
                    logger.error(f"Failed to increase rank for {name}: {e}")
        
        # Update current total
        self.current_total_tuning_params += total_new_params
        
        stats = {
            "rank_growth_performed": True,
            "adapters_modified": modified_adapters,
            "total_new_params": total_new_params,
            "budget_remaining": self._get_available_budget(),
            "avg_params_added_per_adapter": total_new_params / max(modified_adapters, 1),
            "top_adapters": [ai.name for ai in top_adapters],
            "min_importance": min(importance_scores.get(ai.name, 0.0) for ai in top_adapters),
            "max_importance": max(importance_scores.get(ai.name, 0.0) for ai in top_adapters)
        }
        
        logger.info(f"Rank growth completed: added {total_new_params:,} parameters across {modified_adapters} adapters")
        
        return stats
    
    def adaptive_update(self, global_step: int) -> Dict[str, Any]:
        """
        Perform adaptive tuning update at specified training step.
        
        This is the main interface called by the trainer. It:
        1. Checks if it's time for an update
        2. Computes adapter importance scores
        3. Grows ranks of top-salient adapters within budget
        4. Returns statistics
        
        Args:
            global_step: Current global training step
            
        Returns:
            Dictionary with tuning statistics
        """
        # Skip if not time for update
        if global_step % self.update_frequency != 0:
            return {
                "step": global_step,
                "update_performed": False,
                "current_tuning_params": self.current_total_tuning_params,
                "budget_remaining": self._get_available_budget()
            }
        
        try:
            # Compute adapter importance
            importance_scores = self.compute_adapter_importance()
            
            # Grow ranks
            growth_stats = self.grow_ranks(importance_scores)
            
            # Combine stats
            stats = {
                "step": global_step,
                "update_performed": True,
                "current_tuning_params": self.current_total_tuning_params,
                "total_base_params": self.base_model_size,
                "tuning_ratio": self.current_total_tuning_params / max(self.base_model_size, 1),
                "budget_remaining": self._get_available_budget(),
                "adapters_count": len(self.adapters),
                **growth_stats
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Tuning update failed at step {global_step}: {e}")
            return {
                "step": global_step,
                "update_performed": False,
                "error": str(e),
                "current_tuning_params": self.current_total_tuning_params
            }
    
    def get_current_tuning_ratio(self) -> float:
        """
        Get the current ratio of tuning parameters to base model parameters.
        
        Returns:
            Ratio of trainable tuning parameters to total base model parameters
        """
        return self.current_total_tuning_params / max(self.base_model_size, 1)
    
    def get_tuning_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the current tuning state.
        
        Returns:
            Dictionary with comprehensive tuning information
        """
        # Get current importance scores
        importance_scores = {}
        for adapter_info in self.adapters:
            importance_scores[adapter_info.name] = adapter_info.salience_score
        
        # Find min/max/mean ranks
        ranks = [ai.current_rank for ai in self.adapters]
        
        return {
            "total_adapters": len(self.adapters),
            "active_adapters": len([ai for ai in self.adapters if ai.current_rank > 0]),
            "current_tuning_params": self.current_total_tuning_params,
            "base_model_size": self.base_model_size,
            "tuning_ratio": self.get_current_tuning_ratio(),
            "max_allowed_ratio": self.max_tuning_params_ratio,
            "budget_remaining": self._get_available_budget(),
            "min_rank": min(ranks) if ranks else 0,
            "max_rank": max(ranks) if ranks else 0,
            "mean_rank": sum(ranks) / len(ranks) if ranks else 0,
            "std_rank": math.sqrt(sum((r - sum(ranks)/len(ranks))**2 for r in ranks) / len(ranks)) if len(ranks) > 1 else 0,
            "top_adapters_by_importance": sorted(
                [(ai.name, ai.salience_score) for ai in self.adapters],
                key=lambda x: x[1], reverse=True
            )[:5],
            "bottom_adapters_by_importance": sorted(
                [(ai.name, ai.salience_score) for ai in self.adapters],
                key=lambda x: x[1]
            )[:5]
        }

__all__ = ['TuningController']
