# eval/metrics.py
import json
from typing import List, Dict, Any, Tuple, Optional
from utils.config import Config
from utils.logger import Logger
from utils.parallel_sim import ParallelSimulator
from representation.construction_tree import ConstructionTree
from env.simulation import SimulationEngine
from reward.calculator import RewardCalculator
from env.besiegefield import BesiegeFieldSimulator
from env.block_registry import BlockRegistry


class EvaluationMetrics:
    """
    Computes all evaluation metrics for compositional machine design as specified in the paper.
    Evaluates generated machine designs across file validity, spatial validity, machine validity,
    simulation scores, and Pass@k performance.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the EvaluationMetrics with configuration and required components.
        
        Args:
            config (Config, optional): Configuration loader. If None, loads from "config.yaml"
        """
        self.config = config or Config("config.yaml")
        self.logger = Logger(__name__)
        
        # Load configuration values
        self.duration_seconds = self.config.get("simulation.duration_seconds", 5.0)
        self.state_log_interval = self.config.get("simulation.state_log_interval", 0.2)
        self.collision_threshold = self.config.get("simulation.collision_threshold", 0.01)
        self.catapult_height_threshold = self.config.get("simulation.catapult_height_threshold", 3.0)
        self.parallel_sim_workers = self.config.get("training.hardware.parallel_sim_workers", 8)
        
        # Validate configuration
        if not isinstance(self.duration_seconds, (int, float)) or self.duration_seconds <= 0:
            raise ValueError("simulation.duration_seconds must be a positive number")
        if not isinstance(self.state_log_interval, (int, float)) or self.state_log_interval <= 0:
            raise ValueError("simulation.state_log_interval must be a positive number")
        if not isinstance(self.collision_threshold, (int, float)) or self.collision_threshold <= 0:
            raise ValueError("simulation.collision_threshold must be a positive number")
        if not isinstance(self.catapult_height_threshold, (int, float)) or self.catapult_height_threshold < 0:
            raise ValueError("simulation.catapult_height_threshold must be a non-negative number")
        if not isinstance(self.parallel_sim_workers, int) or self.parallel_sim_workers <= 0:
            raise ValueError("training.hardware.parallel_sim_workers must be a positive integer")
        
        # Initialize components
        self.block_registry = BlockRegistry()
        self.parallel_simulator = ParallelSimulator(num_workers=self.parallel_sim_workers)
        
        # Initialize reward calculator (task will be set per design)
        self.reward_calculator = RewardCalculator(task="car", catapult_height_threshold=self.catapult_height_threshold)
        
        # Define Pass@k values from config (used for evaluation)
        self.pass_k_values = [1, 8, 64]  # Always compute for k=1, 8, 64 as per paper
        
        self.logger.info(f"EvaluationMetrics initialized with config: {self.config.get_all()}")

    def compute_all(self, designs: List[ConstructionTree], tasks: List[str]) -> Dict[str, Any]:
        """
        Compute all evaluation metrics for a batch of generated machine designs.
        
        Args:
            designs (List[ConstructionTree]): List of generated machine designs (must be valid ConstructionTree objects)
            tasks (List[str]): List of task types corresponding to each design ("car" or "catapult")
            
        Returns:
            Dict[str, Any]: Dictionary containing all evaluation metrics:
            {
                "file_validity_rate": float,           # Proportion of file-valid designs
                "spatial_validity_rate": float,        # Proportion of spatially valid designs
                "machine_validity_rate": float,        # Proportion of machine-valid designs
                "mean_simulation_score": float,        # Mean reward of machine-valid designs
                "max_simulation_score": float,         # Maximum reward of machine-valid designs
                "pass_at_k": {                         # Pass@k scores for k=1,8,64
                    "k=1": float,
                    "k=8": float,
                    "k=64": float
                }
            }
            
        Raises:
            ValueError: If designs and tasks lists have different lengths
        """
        # Validate inputs
        if len(designs) != len(tasks):
            raise ValueError(f"designs and tasks must have the same length: got {len(designs)} designs and {len(tasks)} tasks")
            
        if len(designs) == 0:
            self.logger.warning("No designs provided for evaluation")
            return {
                "file_validity_rate": 0.0,
                "spatial_validity_rate": 0.0,
                "machine_validity_rate": 0.0,
                "mean_simulation_score": 0.0,
                "max_simulation_score": 0.0,
                "pass_at_k": {"k=1": 0.0, "k=8": 0.0, "k=64": 0.0}
            }
            
        # Initialize counters and lists
        file_valid_count = 0
        spatial_valid_count = 0
        machine_valid_count = 0
        valid_rewards = []  # Rewards for machine-valid designs
        
        # Track simulation failures
        simulation_failures = 0
        
        # Initialize simulator for spatial validation
        temp_simulator = BesiegeFieldSimulator(
            block_list=list(self.block_registry._valid_block_names),
            physics_config={
                "duration_seconds": self.duration_seconds,
                "state_log_interval": self.state_log_interval,
                "gravity": 9.81,
                "collision_threshold": self.collision_threshold,
                "catapult_height_threshold": self.catapult_height_threshold
            }
        )
        
        # Track all simulation results for Pass@k
        all_simulation_results = []  # List of (reward, task) tuples for all designs
        
        # Process each design
        for i, (design, task) in enumerate(zip(designs, tasks)):
            # Ensure task is valid
            if task not in ["car", "catapult"]:
                self.logger.warning(f"Invalid task type '{task}' for design {i}, skipping")
                continue
                
            # 1. File Validity Check
            try:
                is_file_valid, error_msg = design.validate()
            except Exception as e:
                is_file_valid = False
                error_msg = f"Validation exception: {str(e)}"
                
            if not is_file_valid:
                # Skip further checks for file-invalid designs
                continue
                
            file_valid_count += 1
            
            # 2. Spatial Validity Check
            try:
                # Build machine in simulator
                build_success = temp_simulator.build_from_tree(design)
                if not build_success:
                    is_spatial_valid = False
                else:
                    # Check for self-collision
                    has_collision = not temp_simulator.check_self_collision()
                    is_spatial_valid = not has_collision
            except Exception as e:
                is_spatial_valid = False
                self.logger.debug(f"Spatial validation failed for design {i}: {str(e)}")
                
            if is_spatial_valid:
                spatial_valid_count += 1
                
            # 3. Machine Validity Check (intersection of file and spatial validity)
            if is_file_valid and is_spatial_valid:
                machine_valid_count += 1
                
                # 4. Simulation and Reward Calculation
                try:
                    # Initialize simulation engine for this task
                    simulation_engine = SimulationEngine(
                        task=task,
                        simulator=temp_simulator,
                        reward_calculator=self.reward_calculator,
                        config=self.config
                    )
                    
                    # Simulate and get reward
                    reward, r_valid = simulation_engine.simulate(design)
                    
                    # Note: r_valid should be True since we already checked file and spatial validity
                    # But we still use the computed reward
                    if reward is not None and r_valid:
                        valid_rewards.append(reward)
                        all_simulation_results.append((reward, task))
                    else:
                        # Even if simulation fails, we don't count it as valid
                        pass
                        
                except Exception as e:
                    simulation_failures += 1
                    self.logger.debug(f"Simulation failed for machine-valid design {i}: {str(e)}")
                    # Do not count this design in valid_rewards
                    continue
                    
        # Compute basic validity rates
        total_designs = len(designs)
        file_validity_rate = file_valid_count / total_designs if total_designs > 0 else 0.0
        spatial_validity_rate = spatial_valid_count / total_designs if total_designs > 0 else 0.0
        machine_validity_rate = machine_valid_count / total_designs if total_designs > 0 else 0.0
        
        # Compute mean and max simulation scores
        mean_simulation_score = sum(valid_rewards) / len(valid_rewards) if valid_rewards else 0.0
        max_simulation_score = max(valid_rewards) if valid_rewards else 0.0
        
        # Compute Pass@k scores (k=1, 8, 64)
        pass_at_k_scores = {}
        
        # Group designs by prompt for Pass@k calculation
        # We assume that designs are grouped by prompt: for each prompt, there are exactly k rollouts
        # The total number of designs must be divisible by k
        # We'll compute for each k in pass_k_values
        
        for k in self.pass_k_values:
            # Check if we have enough designs for grouping
            if len(all_simulation_results) % k != 0:
                self.logger.warning(f"Number of designs ({len(all_simulation_results)}) not divisible by k={k}. "
                                  f"Pass@{k} may be inaccurate. Proceeding anyway.")
                
            # Group into chunks of size k
            num_groups = len(all_simulation_results) // k
            best_scores_per_prompt = []
            
            for group_idx in range(num_groups):
                start_idx = group_idx * k
                end_idx = start_idx + k
                group_rewards = [result[0] for result in all_simulation_results[start_idx:end_idx]]
                
                # Take the maximum reward in this group
                if group_rewards:
                    best_score = max(group_rewards)
                    best_scores_per_prompt.append(best_score)
                else:
                    best_scores_per_prompt.append(0.0)
                    
            # Compute mean of best scores across all prompts
            if best_scores_per_prompt:
                pass_at_k_scores[f"k={k}"] = sum(best_scores_per_prompt) / len(best_scores_per_prompt)
            else:
                pass_at_k_scores[f"k={k}"] = 0.0
                
        # Log summary
        self.logger.info(f"Evaluation completed: "
                        f"Total designs={total_designs}, "
                        f"File valid={file_valid_count}, "
                        f"Spatial valid={spatial_valid_count}, "
                        f"Machine valid={machine_valid_count}, "
                        f"Valid rewards={len(valid_rewards)}, "
                        f"Sim failures={simulation_failures}")
                        
        # Return all metrics
        return {
            "file_validity_rate": file_validity_rate,
            "spatial_validity_rate": spatial_validity_rate,
            "machine_validity_rate": machine_validity_rate,
            "mean_simulation_score": mean_simulation_score,
            "max_simulation_score": max_simulation_score,
            "pass_at_k": pass_at_k_scores
        }

