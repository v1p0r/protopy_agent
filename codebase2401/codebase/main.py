## main.py
import argparse
import logging
import os
import sys
from typing import Dict, Any, Optional

# Set up logging before any other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_args():
    """Setup command line arguments with defaults from config.yaml."""
    parser = argparse.ArgumentParser(description="Adaptive Pruning and Tuning (APT) Framework")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default=None,
                       help="Pretrained model name (e.g., roberta-base, llama-7b)")
    
    # Data configuration
    parser.add_argument("--task", type=str, default=None,
                       help="Task name (sst2, mnli, squad_v2, cnn_dm, alpaca)")
    parser.add_argument("--max_length", type=int, default=None,
                       help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                       help="Evaluation batch size")
    
    # APT configuration
    parser.add_argument("--initial_rank", type=int, default=None,
                       help="Initial rank for APT adapters")
    parser.add_argument("--alpha_scaling", type=float, default=None,
                       help="Scaling factor for adapter output")
    parser.add_argument("--target_sparsity", type=float, default=None,
                       help="Target sparsity ratio (0.0 to 1.0)")
    parser.add_argument("--update_frequency", type=int, default=None,
                       help="Frequency of adaptive updates in steps")
    parser.add_argument("--max_tuning_params_ratio", type=float, default=None,
                       help="Maximum ratio of tuning parameters to base model")
    
    # Pruning configuration
    parser.add_argument("--use_kurtosis", action="store_true",
                       help="Use kurtosis in salience scoring")
    parser.add_argument("--mask_decay_step", type=float, default=None,
                       help="Step size for gradual mask decay")
    
    # Distillation configuration
    parser.add_argument("--no_distill", action="store_true",
                       help="Disable self-distillation")
    parser.add_argument("--teacher_momentum", type=float, default=None,
                       help="EMA momentum for teacher parameters")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=None,
                       help="Weight decay for optimizer")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=None,
                       help="Number of warmup steps")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for saving results")
    parser.add_argument("--save_model", action="store_true",
                       help="Save trained model")
    parser.add_argument("--log_level", type=str, default="info",
                       choices=["debug", "info", "warning", "error"],
                       help="Logging level")
    
    return parser.parse_args()

def override_config_with_args(config: 'Config', args: argparse.Namespace) -> 'Config':
    """
    Override configuration values with command line arguments.
    
    Args:
        config: Base configuration object
        args: Parsed command line arguments
        
    Returns:
        Updated configuration object
    """
    # Model config
    if args.model_name is not None:
        config.model.name = args.model_name
    
    # Data config
    if args.task is not None:
        config.data.task = args.task
    if args.max_length is not None:
        config.data.max_length = args.max_length
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.eval_batch_size is not None:
        config.data.eval_batch_size = args.eval_batch_size
    
    # APT config
    if args.initial_rank is not None:
        config.apt.initial_rank = args.initial_rank
    if args.alpha_scaling is not None:
        config.apt.alpha_scaling = args.alpha_scaling
    if args.target_sparsity is not None:
        config.apt.target_sparsity = args.target_sparsity
    if args.update_frequency is not None:
        config.apt.update_frequency = args.update_frequency
    if args.max_tuning_params_ratio is not None:
        config.apt.max_tuning_params_ratio = args.max_tuning_params_ratio
    
    # Pruning config
    if args.use_kurtosis:
        config.pruning.use_kurtosis = True
    if args.mask_decay_step is not None:
        config.pruning.mask_decay_step = args.mask_decay_step
    
    # Distillation config
    if args.no_distill:
        config.distillation.enabled = False
    if args.teacher_momentum is not None:
        config.distillation.teacher_momentum = args.teacher_momentum
    
    # Training config
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
    
    # Output config
    if args.output_dir is not None:
        config.output.output_dir = args.output_dir
    if args.save_model:
        config.output.save_model = True
    if args.log_level:
        config.output.log_level = args.log_level.upper()
    
    return config

def validate_config(config: 'Config') -> None:
    """
    Validate configuration settings for consistency and correctness.
    
    Args:
        config: Configuration object to validate
        
    Raises:
        ValueError: If any configuration value is invalid
    """
    # Validate model
    if not config.model.name:
        raise ValueError("model.name must be specified")
    
    # Validate task
    valid_tasks = ["sst2", "mnli", "squad_v2", "cnn_dm", "alpaca"]
    if config.data.task not in valid_tasks:
        raise ValueError(f"task must be one of {valid_tasks}, got {config.data.task}")
    
    # Validate dimensions
    if config.data.max_length <= 0:
        raise ValueError(f"max_length must be positive, got {config.data.max_length}")
    if config.data.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {config.data.batch_size}")
    if config.data.eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {config.data.eval_batch_size}")
    
    # Validate APT parameters
    if config.apt.initial_rank <= 0:
        raise ValueError(f"initial_rank must be positive, got {config.apt.initial_rank}")
    if not (0.0 <= config.apt.alpha_scaling <= 10.0):
        raise ValueError(f"alpha_scaling should be reasonable, got {config.apt.alpha_scaling}")
    if not (0.0 <= config.apt.target_sparsity < 1.0):
        raise ValueError(f"target_sparsity must be in [0,1), got {config.apt.target_sparsity}")
    if config.apt.update_frequency <= 0:
        raise ValueError(f"update_frequency must be positive, got {config.apt.update_frequency}")
    if not (0.0 < config.apt.max_tuning_params_ratio <= 1.0):
        raise ValueError(f"max_tuning_params_ratio must be in (0,1], got {config.apt.max_tuning_params_ratio}")
    
    # Validate pruning
    if config.pruning.mask_decay_step <= 0:
        raise ValueError(f"mask_decay_step must be positive, got {config.pruning.mask_decay_step}")
    
    # Validate distillation
    if config.distillation.enabled:
        if not (0.0 <= config.distillation.teacher_momentum < 1.0):
            raise ValueError(f"teacher_momentum must be in [0,1), got {config.distillation.teacher_momentum}")
        if not (0.0 <= config.distillation.lambda_start <= 1.0):
            raise ValueError(f"lambda_start must be in [0,1], got {config.distillation.lambda_start}")
        if not (0.0 <= config.distillation.lambda_end <= 1.0):
            raise ValueError(f"lambda_end must be in [0,1], got {config.distillation.lambda_end}")
    
    # Validate training
    if config.training.learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {config.training.learning_rate}")
    if config.training.weight_decay < 0:
        raise ValueError(f"weight_decay cannot be negative, got {config.training.weight_decay}")
    if config.training.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {config.training.epochs}")
    if config.training.warmup_steps < 0:
        raise ValueError(f"warmup_steps cannot be negative, got {config.training.warmup_steps}")

def main():
    """Main execution function for the APT framework."""
    try:
        # Parse command line arguments
        args = setup_args()
        
        # Load configuration
        from config import CONFIG as config
        
        # Override with command line arguments
        config = override_config_with_args(config, args)
        
        # Validate configuration
        validate_config(config)
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, config.output.log_level.upper()))
        
        logger.info(f"Starting APT framework with configuration:")
        logger.info(f"  Model: {config.model.name}")
        logger.info(f"  Task: {config.data.task}")
        logger.info(f"  Target sparsity: {config.apt.target_sparsity:.3f}")
        logger.info(f"  Initial rank: {config.apt.initial_rank}")
        logger.info(f"  Distillation: {'enabled' if config.distillation.enabled else 'disabled'}")
        
        # Import modules after configuration is set
        from transformers import AutoTokenizer
        from data.data_loader import DatasetLoader
        from train.trainer import APTrainer
        from eval.evaluator import Evaluator
        
        # Load tokenizer
        logger.info(f"Loading tokenizer for {config.model.name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.tokenizer if config.model.tokenizer != "auto" 
            else config.model.name
        )
        
        # For decoder-only models like LLaMA, set padding side to left
        if "llama" in config.model.name.lower():
            tokenizer.padding_side = "left"
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token for LLaMA model")
        
        # Create dataset loader
        logger.info("Initializing dataset loader...")
        dataset_loader = DatasetLoader(
            tokenizer=tokenizer,
            max_length=config.data.max_length,
            batch_size=config.data.batch_size,
            eval_batch_size=config.data.eval_batch_size
        )
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = APTrainer()
        trainer.setup()
        
        # Train model
        logger.info("Starting training...")
        training_summary = trainer.train()
        
        # Evaluate model
        if hasattr(trainer, 'eval_dataloader') and trainer.eval_dataloader is not None:
            logger.info("Running evaluation...")
            evaluator = Evaluator(
                model=trainer.model,
                tokenizer=tokenizer,
                task=config.data.task
            )
            eval_metrics = evaluator.evaluate(trainer.eval_dataloader)
            
            # Log evaluation results
            logger.info("Evaluation Results:")
            for key, value in eval_metrics.items():
                if value is not None and not isinstance(value, dict):
                    logger.info(f"  {key}: {value:.4f}")
            
            # Include evaluation metrics in summary
            training_summary.update({f"final_{k}": v for k, v in eval_metrics.items() if v is not None})
        
        # Save final results
        import json
        results_path = os.path.join(config.output.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        logger.info(f"Training completed. Results saved to {results_path}")
        logger.info(f"Final performance: {training_summary.get('final_accuracy', training_summary.get('final_f1', 'N/A'))}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
