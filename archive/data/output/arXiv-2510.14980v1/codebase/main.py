#!/usr/bin/env python3
"""
Central CLI orchestrator for the BesiegeField system.
Implements all experimental workflows as specified in the paper:
generate, evaluate, train, finetune, benchmark.
All configuration is loaded from config.yaml.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Import all required modules from the design
from utils.config import Config
from utils.logger import Logger
from dataset.loader import DatasetLoader
from env.block_registry import BlockRegistry
from utils.validator import JSONValidator
from env.besiegefield import BesiegeFieldSimulator
from env.simulation import SimulationEngine
from utils.parallel_sim import ParallelSimulator
from reward.calculator import RewardCalculator
from agent.single_agent import SingleAgent
from agent.meta_designer import MetaDesigner
from agent.designer import Designer
from agent.querier import ActiveEnvQuerier
from agent.refiner import Refiner
from agent.inspector_refiner import InspectorRefiner
from agent.hierarchical_design import HierarchicalDesign
from agent.iterative_editing import IterativeEditing
from rl.trainer import RLTrainer
from eval.metrics import EvaluationMetrics
from eval.visualizer import Visualizer
from representation.construction_tree import ConstructionTree


def setup_logging(config: Config) -> None:
    """
    Initialize global logging with configuration from config.yaml.
    Ensures all modules use the same logger instance.
    """
    log_file = config.get("logging.log_file")
    log_level = config.get("logging.level")
    
    # Configure root logger to handle all logs
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='[%(asctime)sZ] [%(levelname)s] [%(name)s]: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S.%f',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set up a dedicated logger for main
    logger = Logger(__name__, log_file, log_level)
    logger.info(f"Logging initialized with level {log_level} and file {log_file}")


def load_configuration() -> Config:
    """
    Load and validate configuration from config.yaml.
    Exit with error if critical keys are missing or invalid.
    
    Returns:
        Config: Validated configuration object
    """
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        config = Config(config_path)
    except Exception as e:
        print(f"ERROR: Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    # Validate critical keys
    required_keys = [
        "training.cold_start.learning_rate",
        "training.rl_finetune.learning_rate",
        "training.rl_finetune.steps",
        "training.hardware.gpus",
        "model.base_model_name",
        "model.max_input_length",
        "model.max_output_length",
        "simulation.duration_seconds",
        "simulation.state_log_interval",
        "simulation.catapult_height_threshold",
        "agent.search_rounds",
        "agent.candidates_per_round",
        "agent.max_retries_per_node",
        "dataset.cold_start_dataset_path",
        "dataset.test_prompts_path",
        "dataset.num_test_prompts",
        "output.results_dir",
        "output.visualizations_dir",
        "output.model_checkpoints_dir",
        "logging.log_file",
        "logging.level"
    ]
    
    missing_keys = []
    for key in required_keys:
        try:
            config.get(key)
        except KeyError:
            missing_keys.append(key)
    
    if missing_keys:
        print(f"ERROR: Missing required configuration keys: {missing_keys}")
        sys.exit(1)
    
    # Validate numerical values
    try:
        gpus = config.get_int("training.hardware.gpus")
        if gpus <= 0:
            print("ERROR: training.hardware.gpus must be positive")
            sys.exit(1)
        
        steps = config.get_int("training.rl_finetune.steps")
        if steps <= 0:
            print("ERROR: training.rl_finetune.steps must be positive")
            sys.exit(1)
            
        duration = config.get_float("simulation.duration_seconds")
        if duration <= 0:
            print("ERROR: simulation.duration_seconds must be positive")
            sys.exit(1)
            
        height_threshold = config.get_float("simulation.catapult_height_threshold")
        if height_threshold < 0:
            print("ERROR: simulation.catapult_height_threshold must be non-negative")
            sys.exit(1)
            
        num_test_prompts = config.get_int("dataset.num_test_prompts")
        if num_test_prompts <= 0:
            print("ERROR: dataset.num_test_prompts must be positive")
            sys.exit(1)
            
    except ValueError as e:
        print(f"ERROR: Invalid configuration value: {str(e)}")
        sys.exit(1)
    
    # Validate file paths
    cold_start_path = config.get("dataset.cold_start_dataset_path")
    test_prompts_path = config.get("dataset.test_prompts_path")
    
    if not os.path.exists(cold_start_path):
        print(f"ERROR: Cold-start dataset file not found: {cold_start_path}")
        sys.exit(1)
        
    if not os.path.exists(test_prompts_path):
        print(f"ERROR: Test prompts file not found: {test_prompts_path}")
        sys.exit(1)
    
    # Validate output directories
    for dir_path in [
        config.get("output.results_dir"),
        config.get("output.visualizations_dir"),
        config.get("output.model_checkpoints_dir")
    ]:
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                print(f"ERROR: Cannot create output directory {dir_path}: {str(e)}")
                sys.exit(1)
    
    return config


def initialize_components(config: Config) -> Dict[str, Any]:
    """
    Initialize all core components needed for the system.
    Returns a dictionary of initialized objects for use by CLI commands.
    """
    logger = Logger(__name__)
    
    # Initialize shared components
    block_registry = BlockRegistry()
    json_validator = JSONValidator(block_registry._valid_block_names)
    
    # Initialize simulation components
    simulator_config = {
        "duration_seconds": config.get_float("simulation.duration_seconds"),
        "state_log_interval": config.get_float("simulation.state_log_interval"),
        "gravity": config.get_float("simulation.gravity"),
        "collision_threshold": config.get_float("simulation.collision_threshold"),
        "catapult_height_threshold": config.get_float("simulation.catapult_height_threshold")
    }
    
    simulator = BesiegeFieldSimulator(
        block_list=list(block_registry._valid_block_names),
        physics_config=simulator_config
    )
    
    parallel_simulator = ParallelSimulator(
        num_workers=config.get_int("training.hardware.parallel_sim_workers")
    )
    
    # Initialize reward calculators
    car_reward_calculator = RewardCalculator(
        task="car",
        catapult_height_threshold=config.get_float("simulation.catapult_height_threshold")
    )
    
    catapult_reward_calculator = RewardCalculator(
        task="catapult",
        catapult_height_threshold=config.get_float("simulation.catapult_height_threshold")
    )
    
    # Initialize simulation engine
    simulation_engine = SimulationEngine(
        task="car",  # Will be overridden per task
        simulator=simulator,
        reward_calculator=car_reward_calculator,
        config=config
    )
    
    # Initialize evaluation metrics
    evaluation_metrics = EvaluationMetrics(config=config)
    
    # Initialize visualizer
    visualizer = Visualizer(config=config)
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(
        cold_start_dataset_path=config.get("dataset.cold_start_dataset_path"),
        test_prompts_path=config.get("dataset.test_prompts_path"),
        config=config
    )
    
    # Initialize agents with model name from config
    model_name = config.get("model.base_model_name")
    
    # SingleAgent for baseline generation
    single_agent = SingleAgent(
        llm_model=model_name,
        prompt_template=None  # Use default template
    )
    
    # MetaDesigner and Designer for hierarchical workflow
    meta_designer = MetaDesigner(
        llm_model=model_name
    )
    
    designer = Designer(
        llm_model=model_name
    )
    
    # ActiveEnvQuerier for feedback
    querier = ActiveEnvQuerier(
        simulator=simulator
    )
    
    # Refiner and InspectorRefiner for iterative editing
    refiner = Refiner(
        llm_model=model_name
    )
    
    inspector_refiner = InspectorRefiner(
        llm_model=model_name
    )
    
    # HierarchicalDesign workflow
    hierarchical_design = HierarchicalDesign(
        meta_designer=meta_designer,
        designer=designer,
        parallel_simulator=parallel_simulator,
        config=config
    )
    
    # IterativeEditing workflow
    iterative_editing = IterativeEditing(
        designer=designer,
        inspector_refiner=inspector_refiner,
        querier=querier,
        refiner=refiner,
        parallel_sim=parallel_simulator,
        reward_calc=car_reward_calculator,  # Will be overridden per task
        task="car",  # Will be overridden per task
        config=config
    )
    
    # RLTrainer for fine-tuning
    rl_trainer = RLTrainer(config=config)
    
    # Return all initialized components
    components = {
        "config": config,
        "block_registry": block_registry,
        "json_validator": json_validator,
        "simulator": simulator,
        "parallel_simulator": parallel_simulator,
        "car_reward_calculator": car_reward_calculator,
        "catapult_reward_calculator": catapult_reward_calculator,
        "simulation_engine": simulation_engine,
        "evaluation_metrics": evaluation_metrics,
        "visualizer": visualizer,
        "dataset_loader": dataset_loader,
        "single_agent": single_agent,
        "meta_designer": meta_designer,
        "designer": designer,
        "querier": querier,
        "refiner": refiner,
        "inspector_refiner": inspector_refiner,
        "hierarchical_design": hierarchical_design,
        "iterative_editing": iterative_editing,
        "rl_trainer": rl_trainer
    }
    
    logger.info("All components initialized successfully")
    return components


def generate_command(components: Dict[str, Any], task: str) -> None:
    """
    Execute the 'generate' command: Generate one machine design from a task prompt.
    Uses SingleAgent to generate a single design, simulates it, and renders it.
    
    Args:
        components: Dictionary of initialized components
        task: Natural language task description
    """
    logger = Logger(__name__)
    logger.info(f"Executing generate command with task: '{task}'")
    
    # Get components
    single_agent = components["single_agent"]
    simulation_engine = components["simulation_engine"]
    visualizer = components["visualizer"]
    evaluation_metrics = components["evaluation_metrics"]
    config = components["config"]
    
    # Generate machine design
    logger.info("Generating machine design with SingleAgent...")
    try:
        # SingleAgent.generate returns (full_response, tree, success)
        # We need to implement this method to return ConstructionTree
        full_response, tree, success = single_agent.generate(task)
        
        if not success:
            logger.error("SingleAgent failed to generate a valid machine design")
            return
            
        if tree is None:
            logger.error("Generated tree is None")
            return
            
    except Exception as e:
        logger.error(f"Failed to generate machine design: {str(e)}")
        return
    
    # Validate the tree
    is_valid, error_msg = tree.validate()
    if not is_valid:
        logger.error(f"Generated machine is invalid: {error_msg}")
        return
    
    # Simulate the machine
    logger.info("Simulating machine design...")
    try:
        # Determine task type from prompt
        task_type = "catapult" if any(keyword in task.lower() for keyword in ["throw", "boulder", "launch", "catapult"]) else "car"
        
        # Update simulation engine for correct task
        if task_type == "car":
            simulation_engine.reward_calculator = components["car_reward_calculator"]
        else:
            simulation_engine.reward_calculator = components["catapult_reward_calculator"]
        
        simulation_engine.task = task_type
        
        # Simulate and get reward
        reward, is_valid = simulation_engine.simulate(tree)
        
        if reward is None:
            logger.error("Simulation failed to return a reward")
            return
            
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        return
    
    # Render visualization
    logger.info("Rendering machine visualization...")
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            config.get("output.visualizations_dir"),
            f"generate_{task_type}_{timestamp}.png"
        )
        
        visualizer.render_machine(
            tree=tree,
            output_path=output_path,
            task=task_type,
            prompt_id=0,
            machine_id=0
        )
        
        logger.info(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to render visualization: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATE COMMAND RESULTS")
    print("="*60)
    print(f"Task: {task}")
    print(f"File Valid: {is_valid}")
    print(f"Machine Valid: {is_valid}")
    print(f"Simulation Score: {reward:.3f}")
    print(f"Visualization: {output_path}")
    print("="*60)


def evaluate_command(components: Dict[str, Any], agent_type: str, task: str) -> None:
    """
    Execute the 'evaluate' command: Evaluate an agent type on 100 test prompts.
    Uses the specified agent workflow (single, iterative, hierarchical) to generate
    one machine per prompt, then computes evaluation metrics.
    
    Args:
        components: Dictionary of initialized components
        agent_type: Type of agent to evaluate ("single", "iterative", "hierarchical")
        task: Task type ("car" or "catapult")
    """
    logger = Logger(__name__)
    logger.info(f"Executing evaluate command with agent={agent_type}, task={task}")
    
    # Validate inputs
    if agent_type not in ["single", "iterative", "hierarchical"]:
        logger.error(f"Invalid agent_type: {agent_type}. Must be 'single', 'iterative', or 'hierarchical'")
        return
        
    if task not in ["car", "catapult"]:
        logger.error(f"Invalid task: {task}. Must be 'car' or 'catapult'")
        return
    
    # Get components
    dataset_loader = components["dataset_loader"]
    evaluation_metrics = components["evaluation_metrics"]
    visualizer = components["visualizer"]
    config = components["config"]
    
    # Load test prompts
    logger.info("Loading test prompts...")
    _, _, test_prompts = dataset_loader.load_train_val_test()
    
    if len(test_prompts) != config.get_int("dataset.num_test_prompts"):
        logger.error(f"Expected {config.get_int('dataset.num_test_prompts')} test prompts, got {len(test_prompts)}")
        return
    
    logger.info(f"Loaded {len(test_prompts)} test prompts for evaluation")
    
    # Initialize agent based on type
    if agent_type == "single":
        agent = components["single_agent"]
    elif agent_type == "iterative":
        # Create a new iterative editing instance for this task
        iterative_editing = IterativeEditing(
            designer=components["designer"],
            inspector_refiner=components["inspector_refiner"],
            querier=components["querier"],
            refiner=components["refiner"],
            parallel_sim=components["parallel_simulator"],
            reward_calc=components["car_reward_calculator"] if task == "car" else components["catapult_reward_calculator"],
            task=task,
            config=config
        )
        agent = iterative_editing
    elif agent_type == "hierarchical":
        # Create a new hierarchical design instance for this task
        hierarchical_design = HierarchicalDesign(
            meta_designer=components["meta_designer"],
            designer=components["designer"],
            parallel_simulator=components["parallel_simulator"],
            config=config
        )
        agent = hierarchical_design
    else:
        logger.error(f"Unknown agent type: {agent_type}")
        return
    
    # Initialize simulation engine for this task
    simulation_engine = components["simulation_engine"]
    if task == "car":
        simulation_engine.reward_calculator = components["car_reward_calculator"]
    else:
        simulation_engine.reward_calculator = components["catapult_reward_calculator"]
    simulation_engine.task = task
    
    # Generate machines for all prompts
    logger.info(f"Generating machines for {len(test_prompts)} prompts using {agent_type} agent...")
    generated_machines = []
    
    for prompt_idx, prompt in enumerate(test_prompts):
        if prompt_idx % 10 == 0:
            logger.info(f"Processing prompt {prompt_idx + 1}/{len(test_prompts)}")
        
        try:
            if agent_type == "single":
                # SingleAgent generates one machine
                full_response, tree, success = agent.generate(prompt)
                if not success or tree is None:
                    logger.warning(f"SingleAgent failed for prompt {prompt_idx}: {prompt[:50]}...")
                    continue
                generated_machines.append(tree)
                
            elif agent_type == "iterative":
                # IterativeEditing generates one machine
                tree = agent.run()
                if tree is None:
                    logger.warning(f"IterativeEditing failed for prompt {prompt_idx}: {prompt[:50]}...")
                    continue
                generated_machines.append(tree)
                
            elif agent_type == "hierarchical":
                # HierarchicalDesign generates one machine
                tree = agent.run(prompt)
                if tree is None:
                    logger.warning(f"HierarchicalDesign failed for prompt {prompt_idx}: {prompt[:50]}...")
                    continue
                generated_machines.append(tree)
                
        except Exception as e:
            logger.error(f"Agent failed for prompt {prompt_idx}: {str(e)}")
            continue
    
    logger.info(f"Successfully generated {len(generated_machines)} machines out of {len(test_prompts)} prompts")
    
    # Compute evaluation metrics
    logger.info("Computing evaluation metrics...")
    metrics = evaluation_metrics.compute_all(
        designs=generated_machines,
        tasks=[task] * len(generated_machines)
    )
    
    # Save results
    results_dir = config.get("output.results_dir")
    results_file = os.path.join(results_dir, f"eval_{agent_type}_{task}.json")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "agent_type": agent_type,
            "task": task,
            "metrics": metrics,
            "num_prompts": len(test_prompts),
            "num_generated": len(generated_machines)
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print(f"EVALUATE COMMAND RESULTS ({agent_type} agent, {task} task)")
    print("="*60)
    print(f"File Validity Rate: {metrics['file_validity_rate']:.4f}")
    print(f"Spatial Validity Rate: {metrics['spatial_validity_rate']:.4f}")
    print(f"Machine Validity Rate: {metrics['machine_validity_rate']:.4f}")
    print(f"Mean Simulation Score: {metrics['mean_simulation_score']:.4f}")
    print(f"Max Simulation Score: {metrics['max_simulation_score']:.4f}")
    print(f"Pass@1: {metrics['pass_at_k']['k=1']:.4f}")
    print(f"Pass@8: {metrics['pass_at_k']['k=8']:.4f}")
    print(f"Pass@64: {metrics['pass_at_k']['k=64']:.4f}")
    print(f"Results saved to: {results_file}")
    print("="*60)


def train_command(components: Dict[str, Any]) -> None:
    """
    Execute the 'train' command: Cold-start fine-tuning using QOFT.
    Uses the curated cold-start dataset to align LLM reasoning with expert CoT.
    
    Args:
        components: Dictionary of initialized components
    """
    logger = Logger(__name__)
    logger.info("Executing train command (cold-start fine-tuning)")
    
    # Get components
    dataset_loader = components["dataset_loader"]
    rl_trainer = components["rl_trainer"]
    
    # Load cold-start dataset
    logger.info("Loading cold-start dataset...")
    train_dataset, _, _ = dataset_loader.load_train_val_test()
    
    if len(train_dataset) != 9984:
        logger.error(f"Expected 9984 cold-start samples, got {len(train_dataset)}")
        return
    
    logger.info(f"Loaded {len(train_dataset)} cold-start samples for training")
    
    # Perform cold-start fine-tuning
    logger.info("Starting cold-start fine-tuning...")
    try:
        rl_trainer.cold_start_finetune(dataset=train_dataset)
        logger.info("Cold-start fine-tuning completed successfully")
    except Exception as e:
        logger.error(f"Cold-start fine-tuning failed: {str(e)}")
        return


def finetune_command(components: Dict[str, Any]) -> None:
    """
    Execute the 'finetune' command: RL fine-tuning using GRPO with LoRA.
    Uses the cold-started model as initialization and applies RL with verifiable rewards.
    
    Args:
        components: Dictionary of initialized components
    """
    logger = Logger(__name__)
    logger.info("Executing finetune command (RL fine-tuning)")
    
    # Get components
    dataset_loader = components["dataset_loader"]
    rl_trainer = components["rl_trainer"]
    
    # Load cold-start dataset (used for RL training prompts)
    logger.info("Loading cold-start dataset for RL fine-tuning...")
    train_dataset, _, _ = dataset_loader.load_train_val_test()
    
    if len(train_dataset) != 9984:
        logger.error(f"Expected 9984 cold-start samples, got {len(train_dataset)}")
        return
    
    logger.info(f"Loaded {len(train_dataset)} cold-start samples for RL fine-tuning")
    
    # Perform RL fine-tuning
    logger.info("Starting RL fine-tuning...")
    try:
        rl_trainer.rl_finetune(dataset=train_dataset)
        logger.info("RL fine-tuning completed successfully")
    except Exception as e:
        logger.error(f"RL fine-tuning failed: {str(e)}")
        return


def benchmark_command(components: Dict[str, Any], task: str) -> None:
    """
    Execute the 'benchmark' command: Run all agent types and RL models on 100 test prompts.
    Compares performance across single-agent, iterative, hierarchical, and RL-finetuned models.
    
    Args:
        components: Dictionary of initialized components
        task: Task type ("car" or "catapult")
    """
    logger = Logger(__name__)
    logger.info(f"Executing benchmark command for task: {task}")
    
    # Get components
    dataset_loader = components["dataset_loader"]
    evaluation_metrics = components["evaluation_metrics"]
    rl_trainer = components["rl_trainer"]
    config = components["config"]
    
    # Load test prompts
    logger.info("Loading test prompts...")
    _, _, test_prompts = dataset_loader.load_train_val_test()
    
    if len(test_prompts) != config.get_int("dataset.num_test_prompts"):
        logger.error(f"Expected {config.get_int('dataset.num_test_prompts')} test prompts, got {len(test_prompts)}")
        return
    
    logger.info(f"Loaded {len(test_prompts)} test prompts for benchmarking")
    
    # Initialize simulation engine for this task
    simulation_engine = components["simulation_engine"]
    if task == "car":
        simulation_engine.reward_calculator = components["car_reward_calculator"]
    else:
        simulation_engine.reward_calculator = components["catapult_reward_calculator"]
    simulation_engine.task = task
    
    # Initialize agents
    single_agent = components["single_agent"]
    
    iterative_editing = IterativeEditing(
        designer=components["designer"],
        inspector_refiner=components["inspector_refiner"],
        querier=components["querier"],
        refiner=components["refiner"],
        parallel_sim=components["parallel_simulator"],
        reward_calc=components["car_reward_calculator"] if task == "car" else components["catapult_reward_calculator"],
        task=task,
        config=config
    )
    
    hierarchical_design = HierarchicalDesign(
        meta_designer=components["meta_designer"],
        designer=components["designer"],
        parallel_simulator=components["parallel_simulator"],
        config=config
    )
    
    # Store agent configurations
    agent_configs = [
        {
            "name": "single",
            "agent": single_agent,
            "type": "single"
        },
        {
            "name": "iterative",
            "agent": iterative_editing,
            "type": "iterative"
        },
        {
            "name": "hierarchical",
            "agent": hierarchical_design,
            "type": "hierarchical"
        }
    ]
    
    # Run all agent types
    logger.info("Running agent evaluations...")
    agent_results = {}
    
    for agent_config in agent_configs:
        agent_name = agent_config["name"]
        agent = agent_config["agent"]
        agent_type = agent_config["type"]
        
        logger.info(f"Running {agent_name} agent...")
        generated_machines = []
        
        for prompt_idx, prompt in enumerate(test_prompts):
            if prompt_idx % 10 == 0:
                logger.info(f"Processing prompt {prompt_idx + 1}/{len(test_prompts)} for {agent_name}")
            
            try:
                if agent_type == "single":
                    full_response, tree, success = agent.generate(prompt)
                    if not success or tree is None:
                        logger.warning(f"{agent_name} failed for prompt {prompt_idx}")
                        continue
                    generated_machines.append(tree)
                    
                elif agent_type == "iterative":
                    tree = agent.run()
                    if tree is None:
                        logger.warning(f"{agent_name} failed for prompt {prompt_idx}")
                        continue
                    generated_machines.append(tree)
                    
                elif agent_type == "hierarchical":
                    tree = agent.run(prompt)
                    if tree is None:
                        logger.warning(f"{agent_name} failed for prompt {prompt_idx}")
                        continue
                    generated_machines.append(tree)
                    
            except Exception as e:
                logger.error(f"{agent_name} failed for prompt {prompt_idx}: {str(e)}")
                continue
        
        # Compute metrics for this agent
        metrics = evaluation_metrics.compute_all(
            designs=generated_machines,
            tasks=[task] * len(generated_machines)
        )
        
        agent_results[agent_name] = {
            "metrics": metrics,
            "num_generated": len(generated_machines),
            "num_prompts": len(test_prompts)
        }
        
        logger.info(f"{agent_name} completed: {len(generated_machines)} machines generated")
    
    # Run RL evaluation (Pass@k)
    logger.info("Running RL evaluation (Pass@k)...")
    try:
        # Use the RL trainer to evaluate Pass@64
        pass_k_results = rl_trainer.evaluate_pass_k(prompts=test_prompts, k=64)
        agent_results["rl_finetuned"] = {
            "metrics": {
                "file_validity_rate": pass_k_results["validity_rate"],
                "spatial_validity_rate": pass_k_results["validity_rate"],
                "machine_validity_rate": pass_k_results["validity_rate"],
                "mean_simulation_score": pass_k_results["mean_score"],
                "max_simulation_score": pass_k_results["max_score"],
                "pass_at_k": {
                    "k=1": pass_k_results["pass@1"],
                    "k=8": pass_k_results["pass@k"],
                    "k=64": pass_k_results["pass@k"]
                }
            },
            "num_generated": pass_k_results["total_rollouts"],
            "num_prompts": len(test_prompts)
        }
    except Exception as e:
        logger.error(f"RL evaluation failed: {str(e)}")
        agent_results["rl_finetuned"] = {
            "metrics": {
                "file_validity_rate": 0.0,
                "spatial_validity_rate": 0.0,
                "machine_validity_rate": 0.0,
                "mean_simulation_score": 0.0,
                "max_simulation_score": 0.0,
                "pass_at_k": {
                    "k=1": 0.0,
                    "k=8": 0.0,
                    "k=64": 0.0
                }
            },
            "num_generated": 0,
            "num_prompts": len(test_prompts)
        }
    
    # Compile results into CSV format
    results_dir = config.get("output.results_dir")
    csv_file = os.path.join(results_dir, f"benchmark_{task}.csv")
    
    # Write CSV header
    header = "Agent,File Validity Rate,Spatial Validity Rate,Machine Validity Rate,Mean Simulation Score,Max Simulation Score,Pass@1,Pass@8,Pass@64,Number Generated\n"
    
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write(header)
        
        for agent_name, result in agent_results.items():
            metrics = result["metrics"]
            pass_at_k = metrics["pass_at_k"]
            
            line = (
                f"{agent_name},"
                f"{metrics['file_validity_rate']:.4f},"
                f"{metrics['spatial_validity_rate']:.4f},"
                f"{metrics['machine_validity_rate']:.4f},"
                f"{metrics['mean_simulation_score']:.4f},"
                f"{metrics['max_simulation_score']:.4f},"
                f"{pass_at_k['k=1']:.4f},"
                f"{pass_at_k['k=8']:.4f},"
                f"{pass_at_k['k=64']:.4f},"
                f"{result['num_generated']}\n"
            )
            
            f.write(line)
    
    logger.info(f"Benchmark results saved to: {csv_file}")
    
    # Print summary table
    print("\n" + "="*100)
    print(f"BENCHMARK RESULTS FOR {task.upper()} TASK")
    print("="*100)
    print(f"{'Agent':<15} {'File Valid':<12} {'Machine Valid':<15} {'Mean Score':<12} {'Max Score':<12} {'Pass@64':<12} {'Generated':<10}")
    print("-"*100)
    
    for agent_name, result in agent_results.items():
        metrics = result["metrics"]
        pass_at_k = metrics["pass_at_k"]
        
        print(f"{agent_name:<15} {metrics['file_validity_rate']:<12.4f} {metrics['machine_validity_rate']:<15.4f} "
              f"{metrics['mean_simulation_score']:<12.4f} {metrics['max_simulation_score']:<12.4f} "
              f"{pass_at_k['k=64']:<12.4f} {result['num_generated']:<10}")
    
    print("="*100)
    print(f"Results saved to: {csv_file}")
    print("="*100)


def main():
    """
    Main entry point for the BesiegeField system.
    Parses CLI arguments and routes to appropriate command handlers.
    """
    # Setup logging
    config = load_configuration()
    setup_logging(config)
    
    # Initialize components
    components = initialize_components(config)
    
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="BesiegeField: Agentic Design of Compositional Machines"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # generate command
    generate_parser = subparsers.add_parser("generate", help="Generate a single machine design from a task prompt")
    generate_parser.add_argument("--task", type=str, required=True, help="Natural language task description")
    
    # evaluate command
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate an agent type on test prompts")
    evaluate_parser.add_argument("--agent", type=str, required=True, choices=["single", "iterative", "hierarchical"], 
                                 help="Agent type to evaluate")
    evaluate_parser.add_argument("--task", type=str, required=True, choices=["car", "catapult"], 
                                 help="Task type")
    
    # train command
    train_parser = subparsers.add_parser("train", help="Perform cold-start fine-tuning using QOFT")
    
    # finetune command
    finetune_parser = subparsers.add_parser("finetune", help="Perform RL fine-tuning using GRPO")
    
    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run full benchmark across all agents and RL models")
    benchmark_parser.add_argument("--task", type=str, required=True, choices=["car", "catapult"], 
                                  help="Task type to benchmark")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Route to command handler
    if args.command == "generate":
        generate_command(components, args.task)
    elif args.command == "evaluate":
        evaluate_command(components, args.agent, args.task)
    elif args.command == "train":
        train_command(components)
    elif args.command == "finetune":
        finetune_command(components)
    elif args.command == "benchmark":
        benchmark_command(components, args.task)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
