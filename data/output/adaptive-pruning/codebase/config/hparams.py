## config/hparams.py
"""
Global hyperparameters configuration for APT experiments.
This file centralizes all hyperparameters as specified in the paper's Table 6
and the provided config.yaml, ensuring consistency across all modules.
"""

from typing import Dict, Any, Optional
import yaml
import os
from dataclasses import dataclass

# Load configuration from YAML file
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file {CONFIG_PATH} not found. Please ensure it exists.")

with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    learning_rate: float
    batch_size: int
    epochs: int
    distill_epochs: Optional[int]
    target_sparsity: float
    initial_rank: int
    scaling_factor: float
    mask_decay_rate: float
    momentum_salience: float
    warmup_steps: Optional[int]
    optimizer: str
    weight_decay: float
    max_grad_norm: float

@dataclass
class ModelConfig:
    """Model architecture and adapter configuration."""
    architectures: list
    apply_to_mha: bool
    apply_to_ffn: bool
    mha_components: list
    ffn_components: list

@dataclass
class DataConfig:
    """Data loading and processing configuration."""
    datasets: list
    task_type: dict
    num_labels: dict
    max_length: dict

@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    inference_batch_size: dict
    metrics: dict
    eval_tasks: dict
    normalize_metrics: bool
    baseline_ft: bool
    baseline_lora: bool

@dataclass
class PruningConfig:
    """Pruning configuration."""
    scheduler: str
    salience_type: str
    use_kurtosis: bool
    block_types: list
    binary_search: bool
    update_frequency: int
    min_sparsity: float
    target_sparsity: float
    gradual_mask_update: bool

@dataclass
class TuningConfig:
    """Adaptive tuning configuration."""
    adaptive_rank_update: bool
    rank_increase_strategy: str
    max_tuning_params_ratio: float
    init_std: float

@dataclass
class DistillationConfig:
    """Self-distillation configuration."""
    use_self_distillation: bool
    share_frozen_params: bool
    teacher_layer_mapping: str
    num_teacher_layers: int
    distill_loss_weight_start: float
    distill_loss_weight_end: float
    distill_layers_sampled: int
    tr_layer_init: str

@dataclass
class HardwareConfig:
    """Hardware configuration."""
    gpu_type: str
    num_gpus: int
    distributed_training: bool
    precision: str

# Extract configurations from YAML
TRAINING_CONFIG = TrainingConfig(**CONFIG['training'])
MODEL_CONFIG = ModelConfig(**CONFIG['model'])
DATA_CONFIG = DataConfig(**CONFIG['data'])
EVALUATION_CONFIG = EvaluationConfig(**CONFIG['evaluation'])
PRUNING_CONFIG = PruningConfig(**CONFIG['pruning'])
TUNING_CONFIG = TuningConfig(**CONFIG['tuning'])
DISTILLATION_CONFIG = DistillationConfig(**CONFIG['distillation'])
HARDWARE_CONFIG = HardwareConfig(**CONFIG['hardware'])

# Legacy dictionaries for backward compatibility with existing code
GLUE_SMALL_HPARAMS: Dict[str, Any] = {
    'learning_rate': TRAINING_CONFIG.learning_rate,
    'batch_size': TRAINING_CONFIG.batch_size,
    'epochs': TRAINING_CONFIG.epochs,
    'distill_epochs': TRAINING_CONFIG.distill_epochs
}

GLUE_BIG_HPARAMS: Dict[str, Any] = {
    'learning_rate': TRAINING_CONFIG.learning_rate,
    'batch_size': TRAINING_CONFIG.batch_size,
    'epochs': TRAINING_CONFIG.epochs,
    'distill_epochs': TRAINING_CONFIG.distill_epochs
}

SQUAD_HPARAMS: Dict[str, Any] = {
    'learning_rate': TRAINING_CONFIG.learning_rate,
    'batch_size': TRAINING_CONFIG.batch_size,
    'epochs': TRAINING_CONFIG.epochs,
    'distill_epochs': TRAINING_CONFIG.distill_epochs
}

CNN_DM_HPARAMS: Dict[str, Any] = {
    'learning_rate': TRAINING_CONFIG.learning_rate,
    'batch_size': TRAINING_CONFIG.batch_size,
    'epochs': TRAINING_CONFIG.epochs,
    'distill_epochs': TRAINING_CONFIG.distill_epochs
}

ALPACA_HPARAMS: Dict[str, Any] = {
    'learning_rate': TRAINING_CONFIG.learning_rate,
    'batch_size': TRAINING_CONFIG.batch_size,
    'epochs': TRAINING_CONFIG.epochs,
    'distill_epochs': TRAINING_CONFIG.distill_epochs
}

# APT-specific parameters
APT_PARAMS: Dict[str, Any] = {
    'target_sparsity': TRAINING_CONFIG.target_sparsity,
    'initial_rank': TRAINING_CONFIG.initial_rank,
    'scaling_factor': TRAINING_CONFIG.scaling_factor,
    'mask_decay_rate': TRAINING_CONFIG.mask_decay_rate,
    'momentum_salience': TRAINING_CONFIG.momentum_salience,
    'optimizer': TRAINING_CONFIG.optimizer,
    'weight_decay': TRAINING_CONFIG.weight_decay,
    'max_grad_norm': TRAINING_CONFIG.max_grad_norm
}

# Model-specific configurations
MODEL_ARCHITECTURES: list = MODEL_CONFIG.architectures
ADAPTER_CONFIG: Dict[str, Any] = {
    'apply_to_mha': MODEL_CONFIG.apply_to_mha,
    'apply_to_ffn': MODEL_CONFIG.apply_to_ffn,
    'mha_components': MODEL_CONFIG.mha_components,
    'ffn_components': MODEL_CONFIG.ffn_components
}

# Data configurations
DATASETS: list = DATA_CONFIG.datasets
TASK_TYPE: dict = DATA_CONFIG.task_type
NUM_LABELS: dict = DATA_CONFIG.num_labels
MAX_LENGTH: dict = DATA_CONFIG.max_length

# Evaluation configurations
INFERENCE_BATCH_SIZE: dict = EVALUATION_CONFIG.inference_batch_size
METRICS: dict = EVALUATION_CONFIG.metrics
EVAL_TASKS: dict = EVALUATION_CONFIG.eval_tasks
NORMALIZE_METRICS: bool = EVALUATION_CONFIG.normalize_metrics
BASELINE_FT: bool = EVALUATION_CONFIG.baseline_ft
BASELINE_LORA: bool = EVALUATION_CONFIG.baseline_lora

# Pruning configurations
PRUNING_SCHEDULER: str = PRUNING_CONFIG.scheduler
SALIENCE_TYPE: str = PRUNING_CONFIG.salience_type
USE_KURTOSIS: bool = PRUNING_CONFIG.use_kurtosis
BLOCK_TYPES: list = PRUNING_CONFIG.block_types
BINARY_SEARCH: bool = PRUNING_CONFIG.binary_search
UPDATE_FREQUENCY: int = PRUNING_CONFIG.update_frequency
MIN_SPARSITY: float = PRUNING_CONFIG.min_sparsity
TARGET_SPARSITY: float = PRUNING_CONFIG.target_sparsity
GRADUAL_MASK_UPDATE: bool = PRUNING_CONFIG.gradual_mask_update

# Tuning configurations
ADAPTIVE_RANK_UPDATE: bool = TUNING_CONFIG.adaptive_rank_update
RANK_INCREASE_STRATEGY: str = TUNING_CONFIG.rank_increase_strategy
MAX_TUNING_PARAMS_RATIO: float = TUNING_CONFIG.max_tuning_params_ratio
INIT_STD: float = TUNING_CONFIG.init_std

# Distillation configurations
USE_SELF_DISTILLATION: bool = DISTILLATION_CONFIG.use_self_distillation
SHARE_FROZEN_PARAMS: bool = DISTILLATION_CONFIG.share_frozen_params
TEACHER_LAYER_MAPPING: str = DISTILLATION_CONFIG.teacher_layer_mapping
NUM_TEACHER_LAYERS: int = DISTILLATION_CONFIG.num_teacher_layers
DISTILL_LOSS_WEIGHT_START: float = DISTILLATION_CONFIG.distill_loss_weight_start
DISTILL_LOSS_WEIGHT_END: float = DISTILLATION_CONFIG.distill_loss_weight_end
DISTILL_LAYERS_SAMPLED: int = DISTILLATION_CONFIG.distill_layers_sampled
TR_LAYER_INIT: str = DISTILLATION_CONFIG.tr_layer_init

# Hardware configurations
GPU_TYPE: str = HARDWARE_CONFIG.gpu_type
NUM_GPUS: int = HARDWARE_CONFIG.num_gpus
DISTRIBUTED_TRAINING: bool = HARDWARE_CONFIG.distributed_training
PRECISION: str = HARDWARE_CONFIG.precision

# Default values for any missing configurations
DEFAULT_TARGET_SPARSITY: float = 0.6
DEFAULT_INITIAL_RANK: int = 8
DEFAULT_SCALING_FACTOR: float = 2.0
DEFAULT_MASK_DECAY_RATE: float = 0.01
DEFAULT_MOMENTUM_SALIENCE: float = 0.9
DEFAULT_OPTIMIZER: str = "Adam"
DEFAULT_WEIGHT_DECAY: float = 0.0
DEFAULT_MAX_GRAD_NORM: float = 1.0
