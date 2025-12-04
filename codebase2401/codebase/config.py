## config.py
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import yaml

@dataclass
class ModelConfig:
    """Configuration for the base language model."""
    name: str = "roberta-base"
    tokenizer: str = "auto"
    use_gradient_checkpointing: bool = False

@dataclass
class DataConfig:
    """Configuration for dataset loading and preprocessing."""
    task: str = "sst2"
    dataset_path: Optional[str] = None
    max_length: int = 512
    batch_size: int = 32
    eval_batch_size: int = 128

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters and optimization."""
    optimizer: str = "AdamW"
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    epochs: int = 15
    warmup_steps: int = 100
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"

@dataclass
class APTConfig:
    """Configuration specific to Adaptive Pruning and Tuning (APT)."""
    initial_rank: int = 8
    alpha_scaling: float = 2.0
    target_sparsity: float = 0.6
    sparsity_schedule: str = "cubic"
    update_frequency: int = 100
    max_tuning_params_ratio: float = 0.01

@dataclass
class PruningConfig:
    """Configuration for structured pruning behavior."""
    apply_to: list = field(default_factory=lambda: ["mha.query", "mha.value", "ffn"])
    structured_blocks: list = field(default_factory=lambda: ["attention_head", "ffn_neuron", "hidden_dimension"])
    salience_method: str = "outlier_aware"
    use_kurtosis: bool = True
    mask_decay_step: float = 0.01

@dataclass
class DistillationConfig:
    """Configuration for self-knowledge distillation."""
    enabled: bool = True
    teacher_momentum: float = 0.999
    lambda_start: float = 0.0
    lambda_end: float = 1.0
    lambda_schedule: str = "linear"
    layer_mapping: str = "closest_non_pruned"
    transformation_module: str = "lora"

@dataclass
class EvalConfig:
    """Configuration for evaluation metrics and tasks."""
    inference_batch_size: Dict[str, int] = field(default_factory=lambda: {
        "small": 128,
        "llama_7b": 32,
        "llama_13b": 4
    })
    lm_eval_tasks: list = field(default_factory=lambda: [
        "arc_challenge",
        "hellaswag",
        "mmlu",
        "truthfulqa_mc"
    ])
    lm_eval_batches: int = 32
    compute_speedup: bool = True
    compute_memory: bool = True

@dataclass
class OutputConfig:
    """Configuration for output and logging behavior."""
    output_dir: str = "outputs/apt"
    save_model: bool = True
    save_final_only: bool = True
    log_level: str = "info"
    report_to: str = "none"

@dataclass
class Config:
    """
    Central configuration class that holds all settings for the APT system.
    This is the single source of truth for all modules.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    apt: APTConfig = field(default_factory=APTConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from a YAML file and return a Config object.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Config: Fully populated configuration object
        
    Raises:
        FileNotFoundError: If config file does not exist
        yaml.YAMLError: If config file has invalid YAML syntax
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            yaml_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")
    
    # Convert nested dictionaries to dataclass instances
    config = Config(
        model=ModelConfig(**yaml_config.get('model', {})),
        data=DataConfig(**yaml_config.get('data', {})),
        training=TrainingConfig(**yaml_config.get('training', {})),
        apt=APTConfig(**yaml_config.get('apt', {})),
        pruning=PruningConfig(**yaml_config.get('pruning', {})),
        distillation=DistillationConfig(**yaml_config.get('distillation', {})),
        eval=EvalConfig(**yaml_config.get('eval', {})),
        output=OutputConfig(**yaml_config.get('output', {}))
    )
    
    return config

# Global configuration instance
CONFIG: Config = load_config()
