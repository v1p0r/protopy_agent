# reward/calculator.py
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class RewardCalculator:
    """
    Computes the reward R = R_valid * R_task for a simulated machine.
    Task-specific logic for car and catapult as defined in the paper.
    Does not depend on LLMs, agents, or file parsing — only simulation state.
    Uses configuration values passed as parameters to avoid direct config.yaml dependency.
    """

    task: str
    catapult_height_threshold: float = 3.0

    def __post_init__(self):
        """
        Validate task type on initialization.
        """
        if self.task not in ["car", "catapult"]:
            raise ValueError(f"Invalid task: {self.task}. Must be 'car' or 'catapult'.")

    def compute(self, state_log: List[Dict[str, Any]]) -> Tuple[float, bool]:
        """
        Compute the reward R = R_valid * R_task from simulation state log.
        
        Args:
            state_log (List[Dict[str, Any]]): List of timestep snapshots from simulation.
                Each dict contains:
                - "block_id": int
                - "type": str (e.g., "Starting Block", "Boulder")
                - "position": [x, y, z] (float)
                - "orientation": [qx, qy, qz, qw] (float)
                - "velocity": [vx, vy, vz] (float)
                - "angular_velocity": [wx, wy, wz] (float)
                - "integrity": float (1.0 = intact, <=0.1 = broken)
                
        Returns:
            Tuple[float, bool]: (reward_value, is_valid)
            - reward_value: scalar R_task if valid, 0.0 if invalid
            - is_valid: boolean indicating if machine satisfies all validity constraints
            
        Validity Conditions (R_valid):
        1. Machine must be intact throughout simulation (all blocks integrity >= 0.1)
        2. For catapult: boulder must reach > catapult_height_threshold height
        3. File validity and no self-collision are assumed (handled by ConstructionTree and BesiegeFieldSimulator)
        
        Task-Specific R_task:
        - Car: max(0, final_x - initial_x) of root block (forward displacement only)
        - Catapult: max_height * max_distance of boulder (if height > threshold)
        """
        # Edge case: no simulation ran (empty log)
        if not state_log:
            return 0.0, False

        # Check integrity: any block broke during simulation?
        for timestep in state_log:
            if timestep.get("integrity", 1.0) < 0.1:
                return 0.0, False

        # R_valid is True at this point (intact throughout)
        r_valid = True

        if self.task == "car":
            # Find all timesteps for the root block (type="Starting Block")
            root_blocks = [timestep for timestep in state_log if timestep.get("type") == "Starting Block"]
            
            # If no root block found, invalid (should not happen in valid machine)
            if not root_blocks:
                return 0.0, False
            
            # Get initial position (first timestep) and final position (last timestep)
            initial_position = root_blocks[0].get("position", [0.0, 0.0, 0.0])
            final_position = root_blocks[-1].get("position", [0.0, 0.0, 0.0])
            
            # Extract x-coordinate (designated forward direction)
            initial_x = initial_position[0]
            final_x = final_position[0]
            
            # Compute forward displacement (non-negative: backward motion gives 0 reward)
            r_task = max(0.0, final_x - initial_x)

        elif self.task == "catapult":
            # Find all timesteps for the boulder (type="Boulder")
            boulders = [timestep for timestep in state_log if timestep.get("type") == "Boulder"]
            
            # If no boulder found, invalid (should not happen in valid catapult design)
            if not boulders:
                return 0.0, False
            
            # Extract all positions
            boulder_positions = [timestep.get("position", [0.0, 0.0, 0.0]) for timestep in boulders]
            
            # Compute maximum height (y-axis) and maximum distance (x-axis)
            max_height = max(pos[1] for pos in boulder_positions)
            max_distance = max(pos[0] for pos in boulder_positions)
            
            # Apply height threshold for validity
            if max_height <= self.catapult_height_threshold:
                return 0.0, False
            
            # Compute R_task as product of max height and max distance
            r_task = max_height * max_distance

        else:
            # This should be unreachable due to __post_init__ validation
            raise ValueError(f"Unknown task: {self.task}")

        return r_task, r_valid
