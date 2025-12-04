## model/base_model.py
"""
Base model implementation for APT framework.
Handles loading pretrained models (RoBERTa, T5, LLaMA) and integrating APT adapters
into specified layers according to the configuration in config.yaml.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass

# Import from local modules
from model.apt_adapter import APTAdapter
from config.hparams import (
    MODEL_CONFIG, TRAINING_CONFIG, PRUNING_CONFIG,
    ADAPTER_CONFIG, MODEL_ARCHITECTURES
)


@dataclass
class ModelLayerInfo:
    """Information about a model layer for adapter integration."""
    layer_type: str  # 'mha', 'ffn', 'cross_attn'
    component: str   # 'q_proj', 'v_proj', etc.
    layer_idx: int
    submodule_path: str


class BaseModel(nn.Module):
    """
    Base model class that loads pretrained transformer models and integrates APT adapters.
    
    This class handles:
    - Loading pretrained models from Hugging Face
    - Freezing original model parameters
    - Injecting APT adapters into specified layers
    - Providing interfaces for pruning and tuning operations
    
    Based on Section 4.1 of the paper and config.yaml specifications.
    """
    
    def __init__(self, model_name: str, device: Optional[torch.device] = None):
        """
        Initialize the base model with APT adapter integration.
        
        Args:
            model_name: Name of the pretrained model to load (e.g., 'roberta-base')
            device: Device to place the model on (default: cuda if available)
            
        Raises:
            ValueError: If model_name is not supported
            ImportError: If required model architecture cannot be loaded
        """
        super().__init__()
        
        # Validate model name
        if model_name not in MODEL_ARCHITECTURES:
            raise ValueError(f"Unsupported model architecture: {model_name}. "
                           f"Supported architectures: {MODEL_ARCHITECTURES}")
        
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine model type based on name
        if "roberta" in model_name.lower():
            self.model_type = "encoder_only"
        elif "t5" in model_name.lower():
            self.model_type = "encoder_decoder"
        elif "llama" in model_name.lower():
            self.model_type = "decoder_only"
        else:
            raise ValueError(f"Cannot determine model type from name: {model_name}")
        
        # Load pretrained model
        try:
            self.config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, config=self.config)
        except Exception as e:
            raise ImportError(f"Failed to load model {model_name}: {str(e)}")
        
        # Freeze all original parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Store adapter configurations
        self.apply_to_mha = ADAPTER_CONFIG['apply_to_mha']
        self.apply_to_ffn = ADAPTER_CONFIG['apply_to_ffn']
        self.mha_components = ADAPTER_CONFIG['mha_components']
        self.ffn_components = ADAPTER_CONFIG['ffn_components']
        
        # Initialize tracking variables
        self.adapters: Dict[str, APTAdapter] = {}
        self.layer_info: List[ModelLayerInfo] = []
        self.adapter_params_count = 0
        
        # Apply APT adapters to appropriate layers
        self._integrate_apt_adapters()
        
        # Move model to device
        self.to(self.device)
        
        # Verify adapter integration
        if len(self.adapters) == 0:
            print(f"Warning: No APT adapters were integrated into {model_name}. "
                  "Check configuration settings.")
    
    def _integrate_apt_adapters(self):
        """Integrate APT adapters into the model according to configuration."""
        if self.model_type == "encoder_only":
            self._process_encoder_layers(self.model.encoder.layer, "encoder")
        elif self.model_type == "encoder_decoder":
            # Process encoder layers
            if hasattr(self.model.encoder, 'layer'):
                self._process_encoder_layers(self.model.encoder.layer, "encoder")
            # Process decoder layers
            if hasattr(self.model.decoder, 'layers'):
                self._process_decoder_layers(self.model.decoder.layers, "decoder")
        elif self.model_type == "decoder_only":
            if hasattr(self.model, 'layers'):  # LLaMA-style
                self._process_decoder_layers(self.model.layers, "decoder")
            elif hasattr(self.model, 'layer'):  # Alternative naming
                self._process_decoder_layers(self.model.layer, "decoder")
    
    def _process_encoder_layers(self, encoder_layers: nn.ModuleList, prefix: str):
        """
        Process encoder layers to integrate APT adapters.
        
        Args:
            encoder_layers: ModuleList containing encoder layers
            prefix: Prefix for naming (e.g., "encoder")
        """
        for layer_idx, layer in enumerate(encoder_layers):
            # Process MHA components
            if self.apply_to_mha and hasattr(layer, 'attention'):
                attention = layer.attention
                
                # Handle different MHA component naming conventions
                mha_submodules = {
                    'q_proj': None,
                    'v_proj': None
                }
                
                # Try common attribute names for query projection
                for attr_name in ['self.query', 'query', 'q_proj']:
                    if hasattr(attention, attr_name.replace('.', '_')):
                        mha_submodules['q_proj'] = getattr(attention, attr_name.replace('.', '_'))
                        break
                
                # Try common attribute names for value projection  
                for attr_name in ['self.value', 'value', 'v_proj']:
                    if hasattr(attention, attr_name.replace('.', '_')):
                        mha_submodules['v_proj'] = getattr(attention, attr_name.replace('.', '_'))
                        break
                
                # Integrate adapters for MHA components
                for component, submodule in mha_submodules.items():
                    if component in self.mha_components and submodule is not None:
                        self._add_apt_adapter(
                            submodule=submodule,
                            layer_type='mha',
                            component=component,
                            layer_idx=layer_idx,
                            parent_module=attention,
                            submodule_name=component.split('.')[-1]
                        )
            
            # Process FFN components
            if self.apply_to_ffn and hasattr(layer, 'intermediate') and hasattr(layer, 'output'):
                intermediate = layer.intermediate
                output = layer.output
                
                # Handle different FFN component naming conventions
                ffn_submodules = {
                    'intermediate_dense': getattr(intermediate, 'dense', None),
                    'output_dense': getattr(output, 'dense', None)
                }
                
                # Integrate adapters for FFN components
                for component, submodule in ffn_submodules.items():
                    if component in self.ffn_components and submodule is not None:
                        parent_module = intermediate if 'intermediate' in component else output
                        self._add_apt_adapter(
                            submodule=submodule,
                            layer_type='ffn',
                            component=component,
                            layer_idx=layer_idx,
                            parent_module=parent_module,
                            submodule_name=component.split('_')[0]  # 'intermediate' or 'output'
                        )
    
    def _process_decoder_layers(self, decoder_layers: nn.ModuleList, prefix: str):
        """
        Process decoder layers to integrate APT adapters.
        
        Args:
            decoder_layers: ModuleList containing decoder layers
            prefix: Prefix for naming (e.g., "decoder")
        """
        for layer_idx, layer in enumerate(decoder_layers):
            # Process self-attention components
            if self.apply_to_mha:
                # Handle different self-attention naming conventions
                self_attention = None
                for attr_name in ['self_attn', 'self_attention', 'attention']:
                    if hasattr(layer, attr_name):
                        self_attention = getattr(layer, attr_name)
                        break
                
                if self_attention is not None:
                    # Handle different MHA component naming conventions
                    mha_submodules = {
                        'q_proj': None,
                        'v_proj': None
                    }
                    
                    # Query projection
                    for attr_name in ['q_proj', 'query']:
                        if hasattr(self_attention, attr_name):
                            mha_submodules['q_proj'] = getattr(self_attention, attr_name)
                            break
                    
                    # Value projection  
                    for attr_name in ['v_proj', 'value']:
                        if hasattr(self_attention, attr_name):
                            mha_submodules['v_proj'] = getattr(self_attention, attr_name)
                            break
                    
                    # Integrate adapters for MHA components
                    for component, submodule in mha_submodules.items():
                        if component in self.mha_components and submodule is not None:
                            self._add_apt_adapter(
                                submodule=submodule,
                                layer_type='mha',
                                component=component,
                                layer_idx=layer_idx,
                                parent_module=self_attention,
                                submodule_name=component
                            )
            
            # Process FFN components
            if self.apply_to_ffn:
                # Handle different FFN naming conventions
                ffn_modules = None
                for attr_name in ['mlp', 'ffn', 'feed_forward']:
                    if hasattr(layer, attr_name):
                        ffn_modules = getattr(layer, attr_name)
                        break
                
                if ffn_modules is not None:
                    # Handle gated FFNs (T5, LLaMA)
                    if hasattr(ffn_modules, 'gate_proj') and hasattr(ffn_modules, 'down_proj'):
                        ffn_submodules = {
                            'gate_proj': ffn_modules.gate_proj,
                            'up_proj': ffn_modules.up_proj,
                            'down_proj': ffn_modules.down_proj
                        }
                        
                        # For gated FFNs, we consider gate and up projections as intermediate,
                        # and down projection as output
                        for component, submodule in ffn_submodules.items():
                            if ('gate' in component or 'up' in component) and 'intermediate_dense' in self.ffn_components:
                                self._add_apt_adapter(
                                    submodule=submodule,
                                    layer_type='ffn',
                                    component=f"{component}_intermediate",
                                    layer_idx=layer_idx,
                                    parent_module=ffn_modules,
                                    submodule_name=component
                                )
                            elif 'down' in component and 'output_dense' in self.ffn_components:
                                self._add_apt_adapter(
                                    submodule=submodule,
                                    layer_type='ffn',
                                    component=f"{component}_output",
                                    layer_idx=layer_idx,
                                    parent_module=ffn_modules,
                                    submodule_name=component
                                )
                    
                    # Handle standard FFNs
                    elif hasattr(ffn_modules, 'intermediate') and hasattr(ffn_modules, 'output'):
                        intermediate = ffn_modules.intermediate
                        output = ffn_modules.output
                        
                        ffn_submodules = {
                            'intermediate_dense': getattr(intermediate, 'dense', None),
                            'output_dense': getattr(output, 'dense', None)
                        }
                        
                        for component, submodule in ffn_submodules.items():
                            if component in self.ffn_components and submodule is not None:
                                parent_module = intermediate if 'intermediate' in component else output
                                self._add_apt_adapter(
                                    submodule=submodule,
                                    layer_type='ffn',
                                    component=component,
                                    layer_idx=layer_idx,
                                    parent_module=parent_module,
                                    submodule_name=component.split('_')[0]
                                )
            
            # Process cross-attention for encoder-decoder models
            if self.model_type == "encoder_decoder" and self.apply_to_mha:
                cross_attention = None
                for attr_name in ['cross_attn', 'cross_attention', 'encoder_attn']:
                    if hasattr(layer, attr_name):
                        cross_attention = getattr(layer, attr_name)
                        break
                
                if cross_attention is not None:
                    # Handle different cross-attention component naming conventions
                    cross_mha_submodules = {
                        'q_proj': None,
                        'v_proj': None
                    }
                    
                    # Query projection
                    for attr_name in ['q_proj', 'query']:
                        if hasattr(cross_attention, attr_name):
                            cross_mha_submodules['q_proj'] = getattr(cross_attention, attr_name)
                            break
                    
                    # Value projection
                    for attr_name in ['v_proj', 'value']:
                        if hasattr(cross_attention, attr_name):
                            cross_mha_submodules['v_proj'] = getattr(cross_attention, attr_name)
                            break
                    
                    # Integrate adapters for cross-attention components
                    for component, submodule in cross_mha_submodules.items():
                        if component in self.mha_components and submodule is not None:
                            self._add_apt_adapter(
                                submodule=submodule,
                                layer_type='cross_attn',
                                component=component,
                                layer_idx=layer_idx,
                                parent_module=cross_attention,
                                submodule_name=component
                            )
    
    def _add_apt_adapter(self, submodule: nn.Module, layer_type: str, component: str,
                        layer_idx: int, parent_module: nn.Module, submodule_name: str):
        """
        Add an APT adapter to a submodule.
        
        Args:
            submodule: The module to wrap with adapter
            layer_type: Type of layer ('mha', 'ffn', 'cross_attn')
            component: Component name ('q_proj', 'v_proj', etc.)
            layer_idx: Index of the layer
            parent_module: Parent module containing the submodule
            submodule_name: Name of the submodule
        """
        # Get dimensions from submodule
        if hasattr(submodule, 'weight'):
            in_features = submodule.weight.size(1)  # Input dimension
            out_features = submodule.weight.size(0)  # Output dimension
        else:
            # Fallback for modules without weight (should not happen)
            raise ValueError(f"Submodule {submodule_name} has no weight parameter")
        
        # Create APT adapter
        adapter = APTAdapter(
            in_features=in_features,
            out_features=out_features,
            rank=TRAINING_CONFIG.initial_rank,
            scaling=TRAINING_CONFIG.scaling_factor
        )
        
        # Create unique name for this adapter
        adapter_name = f"{layer_type}_{component}_layer{layer_idx}"
        
        # Store adapter as module attribute
        setattr(self, f"adapter_{adapter_name}", adapter)
        
        # Store reference
        self.adapters[adapter_name] = adapter
        
        # Store layer information
        self.layer_info.append(ModelLayerInfo(
            layer_type=layer_type,
            component=component,
            layer_idx=layer_idx,
            submodule_path=f"{parent_module.__class__.__name__}.{submodule_name}"
        ))
        
        # Wrap forward pass to include adapter
        original_forward = submodule.forward
        
        def wrapped_forward(x):
            # Original transformation
            original_output = original_forward(x)
            # Adapter transformation
            adapter_output = adapter(x)
            # Combine with residual connection
            return original_output + adapter_output
        
        # Replace forward method
        submodule.forward = wrapped_forward
        
        # Update adapter parameters count
        self.adapter_params_count += sum(p.numel() for p in adapter.parameters())
    
    def forward(self, *args, **kwargs):
        """
        Forward pass through the base model.
        
        Args:
            *args: Positional arguments passed to model forward
            **kwargs: Keyword arguments passed to model forward
            
        Returns:
            Model output
        """
        return self.model(*args, **kwargs)
    
    def get_adapter_params(self) -> Dict[str, nn.Parameter]:
        """
        Get all adapter parameters for optimizer setup.
        
        Returns:
            Dictionary mapping parameter names to parameters
        """
        params = {}
        for name, adapter in self.adapters.items():
            for param_name, param in adapter.named_parameters():
                params[f"adapter_{name}.{param_name}"] = param
        return params
    
    def get_frozen_params(self) -> Dict[str, nn.Parameter]:
        """
        Get all frozen parameters (original model weights).
        
        Returns:
            Dictionary mapping parameter names to parameters
        """
        params = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                params[name] = param
        return params
    
    def merge_adapters(self):
        """
        Merge adapter weights into base model for inference efficiency.
        
        This method should be called after training is complete.
        """
        for adapter_name, adapter in self.adapters.items():
            # Get the corresponding submodule (this would require storing references)
            # In practice, we would need to store the original submodules
            pass  # Implementation depends on how we track original modules
    
    def reset_optimizer(self):
        """
        Reset optimizer state when adapter shapes change.
        
        This method should be called by the trainer when ranks are adjusted.
        """
        # This would typically be handled by the trainer
        # but we provide a hook for any internal state reset
        pass
    
    def get_sparsity_stats(self) -> Dict[str, Any]:
        """
        Get current sparsity statistics.
        
        Returns:
            Dictionary containing sparsity information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'adapter_parameters': self.adapter_params_count,
            'sparsity_ratio': frozen_params / total_params if total_params > 0 else 0.0,
            'num_adapters': len(self.adapters)
        }
    
    def get_layer_info(self) -> List[ModelLayerInfo]:
        """
        Get information about all layers with adapters.
        
        Returns:
            List of ModelLayerInfo objects
        """
        return self.layer_info.copy()
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def to(self, device: torch.device):
        """
        Move model to specified device.
        
        Args:
            device: Target device
        """
        super().to(device)
        self.model.to(device)
        return self
    
    def train(self, mode: bool = True):
        """
        Set the model to training mode.
        
        Args:
            mode: Whether to set to training mode (True) or evaluation mode (False)
        """
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set the model to evaluation mode."""
        return self.train(False)
