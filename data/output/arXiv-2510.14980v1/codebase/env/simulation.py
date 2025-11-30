# env/simulation.py
from typing import List, Dict, Any, Tuple, Optional
from utils.config import Config
from env.besiegefield import BesiegeFieldSimulator
from representation.construction_tree import ConstructionTree
from reward.calculator import RewardCalculator
import logging


class SimulationEngine:
    """
    Orchestrates the simulation of a machine design and computes the final reward.
    Wraps BesiegeFieldSimulator and delegates reward calculation to RewardCalculator.
    Implements the exact reward logic and validity constraints specified in the paper.
    """

    def __init__(self, task: str, simulator: BesiegeFieldSimulator, reward_calculator: RewardCalculator, config: Config):
        """
        Initialize the SimulationEngine.
        
        Args:
            task (str): Task type, either "car" or "catapult"
            simulator (BesiegeFieldSimulator): Instance of the physics simulation engine
            reward_calculator (RewardCalculator): Instance for computing task-specific rewards
            config (Config): Configuration loader from config.yaml
        """
        self.task = task.lower()
        if self.task not in ["car", "catapult"]:
            raise ValueError(f"Invalid task: {task}. Must be 'car' or 'catapult'.")
        
        self.simulator = simulator
        self.reward_calculator = reward_calculator
        self.config = config
        
        # Load configuration values
        self.duration_seconds = self.config.get("simulation.duration_seconds")
        self.state_log_interval = self.config.get("simulation.state_log_interval")
        self.catapult_height_threshold = self.config.get("simulation.catapult_height_threshold")
        
        # Validate configuration
        if not isinstance(self.duration_seconds, (int, float)) or self.duration_seconds <= 0:
            raise ValueError("simulation.duration_seconds must be a positive number")
        if not isinstance(self.state_log_interval, (int, float)) or self.state_log_interval <= 0:
            raise ValueError("simulation.state_log_interval must be a positive number")
        if not isinstance(self.catapult_height_threshold, (int, float)) or self.catapult_height_threshold < 0:
            raise ValueError("simulation.catapult_height_threshold must be a non-negative number")
        
        # Calculate expected number of timesteps
        self.expected_timesteps = int(self.duration_seconds / self.state_log_interval)
        
        # Set up logger
        self.logger = logging.getLogger(__name__)

    def simulate(self, machine: ConstructionTree) -> Tuple[float, bool]:
        """
        Simulate a machine design and return the reward and validity status.
        
        Steps:
        1. Build the machine in the simulator
        2. Check for self-collision before simulation
        3. Run physics simulation
        4. Extract state log
        5. Compute R_valid and R_task using RewardCalculator
        
        Returns:
            Tuple[float, bool]: (reward_value, is_valid)
            - reward_value: scalar reward (R_task) if valid, 0.0 if invalid
            - is_valid: boolean indicating if machine satisfies all validity constraints
        """
        # Step 1: Build machine in simulator
        build_success = self.simulator.build_from_tree(machine)
        if not build_success:
            self.logger.error(f"Failed to build machine: build_from_tree returned False")
            return 0.0, False
        
        # Step 2: Check for self-collision before simulation
        has_self_collision = not self.simulator.check_self_collision()
        if has_self_collision:
            self.logger.debug(f"Machine has self-collision before simulation")
            return 0.0, False
        
        # Step 3: Run physics simulation
        try:
            self.simulator.simulate()
        except Exception as e:
            self.logger.error(f"Simulation failed: {str(e)}")
            return 0.0, False
        
        # Step 4: Extract state log
        state_log = self.simulator.get_state_log()
        
        # Validate state log length
        if len(state_log) != self.expected_timesteps:
            self.logger.warning(f"Expected {self.expected_timesteps} timesteps, got {len(state_log)}")
            # Still proceed with available data, but log warning
        
        # Step 5: Compute reward and validity using RewardCalculator
        try:
            reward_value, is_valid = self.reward_calculator.compute(state_log, self.task)
            return reward_value, is_valid
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {str(e)}")
            return 0.0, False
