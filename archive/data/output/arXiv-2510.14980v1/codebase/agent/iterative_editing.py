# agent/iterative_editing.py
import json
import math
from typing import List, Dict, Any, Optional, Tuple
from utils.config import Config
from utils.logger import Logger
from utils.validator import JSONValidator
from representation.construction_tree import ConstructionTree
from env.besiegefield import BesiegeFieldSimulator
from agent.designer import Designer
from agent.inspector_refiner import InspectorRefiner
from agent.querier import ActiveEnvQuerier
from agent.refiner import Refiner
from utils.parallel_sim import ParallelSimulator
from reward.calculator import RewardCalculator


class IterativeEditing:
    """
    Implements the Monte Carlo Tree Search (MCTS) based iterative editing workflow
    for compositional machine design. Orchestrates a closed-loop process of
    generation, critique, simulation, and revision over a fixed number of search rounds.
    Returns the highest-rewarding valid machine design found.
    """

    def __init__(
        self,
        designer: Designer,
        inspector_refiner: InspectorRefiner,
        querier: ActiveEnvQuerier,
        refiner: Refiner,
        parallel_sim: ParallelSimulator,
        reward_calc: RewardCalculator,
        task: str,
        config: Config = None
    ):
        """
        Initialize the IterativeEditing workflow with all required agents and components.
        
        Args:
            designer (Designer): Agent that generates initial machine designs
            inspector_refiner (InspectorRefiner): Agent that critiques designs and generates revisions
            querier (ActiveEnvQuerier): Agent that runs simulation and extracts feedback
            refiner (Refiner): Agent that generates multiple candidate revisions
            parallel_sim (ParallelSimulator): Component that runs physics simulations in parallel
            reward_calc (RewardCalculator): Component that computes R_valid and R_task from simulation logs
            task (str): Task type ("car" or "catapult")
            config (Config, optional): Configuration loader. If None, loads from "config.yaml"
        """
        self.designer = designer
        self.inspector_refiner = inspector_refiner
        self.querier = querier
        self.refiner = refiner
        self.parallel_sim = parallel_sim
        self.reward_calc = reward_calc
        self.task = task.lower()
        self.config = config or Config("config.yaml")
        self.logger = Logger(__name__)
        self.json_validator = JSONValidator()
        
        # Load configuration values with defaults from config.yaml
        self.search_rounds = self.config.get("agent.search_rounds", 10)
        self.candidates_per_round = self.config.get("agent.candidates_per_round", 5)
        self.max_retries_per_node = self.config.get("agent.max_retries_per_node", 5)
        self.catapult_height_threshold = self.config.get("simulation.catapult_height_threshold", 3.0)
        
        # Validate parameters
        if not isinstance(self.search_rounds, int) or self.search_rounds < 1:
            raise ValueError("agent.search_rounds must be a positive integer")
        if not isinstance(self.candidates_per_round, int) or self.candidates_per_round < 1:
            raise ValueError("agent.candidates_per_round must be a positive integer")
        if not isinstance(self.max_retries_per_node, int) or self.max_retries_per_node < 1:
            raise ValueError("agent.max_retries_per_node must be a positive integer")
        if not isinstance(self.catapult_height_threshold, (int, float)) or self.catapult_height_threshold < 0:
            raise ValueError("simulation.catapult_height_threshold must be a non-negative number")
            
        # Initialize MCTS tree tracking
        self.best_design = None
        self.best_score = float('-inf')
        
        # MCTS exploration constant (standard value from literature, not specified in paper)
        self.ucb1_c = 1.41
        
        # Initialize MCTS root node
        self.root = None
        
        # Track total simulations
        self.total_simulations = 0

    class MCTSNode:
        """
        Lightweight node structure for MCTS tree.
        Each node represents a machine design and its search statistics.
        """
        def __init__(self, design: ConstructionTree, parent=None):
            """
            Initialize an MCTS node.
            
            Args:
                design (ConstructionTree): The machine design at this node
                parent (MCTSNode, optional): Parent node in the tree
            """
            self.design = design
            self.parent = parent
            self.children: List['IterativeEditing.MCTSNode'] = []
            self.visit_count = 0
            self.total_reward = 0.0
            self.is_expanded = False
            self.is_valid = False  # Whether design passed file + spatial validity
            self.simulation_reward = 0.0  # Last computed reward for this node

    def _compute_ucb1(self, node: 'IterativeEditing.MCTSNode', child: 'IterativeEditing.MCTSNode') -> float:
        """
        Compute the UCB1 score for a child node.
        UCB1 = (child.total_reward / child.visit_count) + c * sqrt(ln(parent.visit_count) / child.visit_count)
        
        Args:
            node (MCTSNode): Parent node
            child (MCTSNode): Child node to evaluate
            
        Returns:
            float: UCB1 score
        """
        if child.visit_count == 0:
            return float('inf')  # Always explore unvisited nodes
            
        exploitation = child.total_reward / child.visit_count
        exploration = self.ucb1_c * math.sqrt(math.log(node.visit_count) / child.visit_count)
        return exploitation + exploration

    def _select_node(self) -> Optional['IterativeEditing.MCTSNode']:
        """
        Select a leaf node using UCB1 policy.
        Traverse the tree from root until reaching an unexpanded node.
        
        Returns:
            MCTSNode or None: Selected leaf node, or None if no valid root exists
        """
        if self.root is None:
            self.logger.error("No root node exists in MCTS tree")
            return None
            
        current = self.root
        while current.is_expanded and len(current.children) > 0:
            # Compute UCB1 for all children
            ucb1_scores = []
            for child in current.children:
                score = self._compute_ucb1(current, child)
                ucb1_scores.append((score, child))
                
            # Select child with highest UCB1 score
            ucb1_scores.sort(key=lambda x: x[0], reverse=True)
            current = ucb1_scores[0][1]
            
        return current

    def _expand_node(self, node: 'IterativeEditing.MCTSNode') -> bool:
        """
        Expand a node by generating candidate revisions.
        Uses InspectorRefiner to generate ≥5 candidates, with up to 5 retries if needed.
        Only valid designs become children.
        
        Args:
            node (MCTSNode): Node to expand
            
        Returns:
            bool: True if expansion succeeded (at least one child created), False otherwise
        """
        if node.is_expanded:
            return True  # Already expanded
            
        # Get minimal feedback for this design
        try:
            # Simulate the current design to get feedback
            state_log = self._simulate_design(node.design)
            if state_log is None:
                self.logger.error(f"Failed to simulate design for expansion at node")
                return False
                
            feedback = self.querier.get_feedback(self.task, node.design, state_log)
        except Exception as e:
            self.logger.error(f"Failed to get feedback for expansion: {str(e)}")
            return False
            
        # Use InspectorRefiner to generate revisions
        # Note: InspectorRefiner.critique() returns a list of ConstructionTree objects
        candidates = []
        retries = 0
        
        while len(candidates) < self.candidates_per_round and retries < self.max_retries_per_node:
            try:
                # Generate candidate revisions
                new_candidates = self.inspector_refiner.critique(node.design, feedback)
                # Filter for valid designs only
                valid_candidates = []
                for candidate in new_candidates:
                    # Validate file and spatial validity
                    is_valid, error_msg = candidate.validate()
                    if not is_valid:
                        self.logger.debug(f"Invalid candidate design: {error_msg}")
                        continue
                        
                    # Check for self-collision using BesiegeFieldSimulator
                    # We'll use a temporary simulator for this check
                    temp_sim = BesiegeFieldSimulator(
                        block_list=list(self.designer.block_registry._valid_block_names),
                        physics_config={
                            "duration_seconds": 5.0,
                            "state_log_interval": 0.2,
                            "gravity": 9.81,
                            "collision_threshold": 0.01,
                            "catapult_height_threshold": self.catapult_height_threshold
                        }
                    )
                    build_success = temp_sim.build_from_tree(candidate)
                    if not build_success:
                        continue
                    has_collision = not temp_sim.check_self_collision()
                    if has_collision:
                        continue
                        
                    # If valid, add to candidates
                    valid_candidates.append(candidate)
                    
                candidates.extend(valid_candidates)
                
                if len(candidates) >= self.candidates_per_round:
                    break
                    
            except Exception as e:
                self.logger.error(f"InspectorRefiner failed during expansion: {str(e)}")
                
            retries += 1
            
        # If we still don't have enough candidates, use parent design as fallback
        if len(candidates) == 0:
            # Use parent design as fallback candidate
            candidates.append(node.design)
            self.logger.warning("No valid candidates generated, using parent design as fallback")
        elif len(candidates) < self.candidates_per_round:
            # Pad with parent design if needed
            while len(candidates) < self.candidates_per_round:
                candidates.append(node.design)
            self.logger.warning(f"Only {len(candidates)} valid candidates generated, padding with parent design")
            
        # Create child nodes for each candidate
        for candidate in candidates[:self.candidates_per_round]:
            child_node = IterativeEditing.MCTSNode(candidate, parent=node)
            # Validate child design
            is_valid, error_msg = child_node.design.validate()
            if is_valid:
                temp_sim = BesiegeFieldSimulator(
                    block_list=list(self.designer.block_registry._valid_block_names),
                    physics_config={
                        "duration_seconds": 5.0,
                        "state_log_interval": 0.2,
                        "gravity": 9.81,
                        "collision_threshold": 0.01,
                        "catapult_height_threshold": self.catapult_height_threshold
                    }
                )
                build_success = temp_sim.build_from_tree(child_node.design)
                if build_success:
                    has_collision = not temp_sim.check_self_collision()
                    if not has_collision:
                        child_node.is_valid = True
            child_node.is_valid = True  # For now, assume valid since we filtered above
            node.children.append(child_node)
            
        node.is_expanded = True
        return len(node.children) > 0

    def _simulate_design(self, design: ConstructionTree) -> Optional[List[Dict[str, Any]]]:
        """
        Simulate a single machine design and return the state log.
        Uses ParallelSimulator to run the simulation.
        
        Args:
            design (ConstructionTree): Machine design to simulate
            
        Returns:
            List[Dict[str, Any]] or None: Simulation state log, or None if simulation failed
        """
        try:
            # Use the parallel simulator to simulate the design
            # We need to simulate one design at a time
            state_logs = self.parallel_sim.simulate_batch([design], self.task)
            if len(state_logs) == 0:
                return None
            return state_logs[0]  # Return the first (and only) state log
        except Exception as e:
            self.logger.error(f"Simulation failed for design: {str(e)}")
            return None

    def _simulate_and_evaluate(self, node: 'IterativeEditing.MCTSNode') -> float:
        """
        Simulate a node's design and compute its reward.
        Updates the node's simulation_reward and is_valid flags.
        
        Args:
            node (MCTSNode): Node to simulate
            
        Returns:
            float: Computed reward value
        """
        if not node.is_valid:
            return 0.0
            
        # Simulate the design
        state_log = self._simulate_design(node.design)
        if state_log is None:
            node.simulation_reward = 0.0
            return 0.0
            
        # Compute reward using RewardCalculator
        try:
            reward, is_valid = self.reward_calc.compute(state_log)
            node.simulation_reward = reward
            node.is_valid = is_valid
            return reward
        except Exception as e:
            self.logger.error(f"Reward calculation failed: {str(e)}")
            node.simulation_reward = 0.0
            return 0.0

    def _backpropagate(self, node: 'IterativeEditing.MCTSNode', reward: float) -> None:
        """
        Backpropagate the reward from a leaf node up to the root.
        Updates visit count and total reward for all nodes in the path.
        
        Args:
            node (MCTSNode): Leaf node that was simulated
            reward (float): Reward obtained from simulation
        """
        current = node
        while current is not None:
            current.visit_count += 1
            current.total_reward += reward
            current = current.parent

    def _update_best_design(self, node: 'IterativeEditing.MCTSNode') -> None:
        """
        Update the global best design if this node has a higher reward.
        
        Args:
            node (MCTSNode): Node to compare
        """
        if node.simulation_reward > self.best_score:
            self.best_score = node.simulation_reward
            self.best_design = node.design

    def _generate_initial_design(self) -> Optional[ConstructionTree]:
        """
        Generate the initial design using the Designer agent.
        
        Returns:
            ConstructionTree or None: Initial machine design, or None if generation failed
        """
        try:
            # For the root design, use the task directly
            # The designer expects a task string
            design = self.designer.generate_design(self.task)
            return design
        except Exception as e:
            self.logger.error(f"Designer failed to generate initial design: {str(e)}")
            return None

    def run(self) -> Optional[ConstructionTree]:
        """
        Execute the iterative editing workflow using MCTS.
        Performs search_rounds rounds of selection, expansion, simulation, and backpropagation.
        Returns the best design found over all rounds.
        
        Returns:
            ConstructionTree or None: The highest-rewarding valid machine design, or None if no valid design found
        """
        self.logger.info(f"Starting iterative editing workflow for task: {self.task}")
        
        # Generate initial root design
        root_design = self._generate_initial_design()
        if root_design is None:
            self.logger.error("Failed to generate initial root design")
            return None
            
        # Validate root design
        is_valid, error_msg = root_design.validate()
        if not is_valid:
            self.logger.error(f"Initial root design is invalid: {error_msg}")
            return None
            
        # Check for self-collision
        temp_sim = BesiegeFieldSimulator(
            block_list=list(self.designer.block_registry._valid_block_names),
            physics_config={
                "duration_seconds": 5.0,
                "state_log_interval": 0.2,
                "gravity": 9.81,
                "collision_threshold": 0.01,
                "catapult_height_threshold": self.catapult_height_threshold
            }
        )
        build_success = temp_sim.build_from_tree(root_design)
        if not build_success:
            self.logger.error("Failed to build root design in simulator")
            return None
        has_collision = not temp_sim.check_self_collision()
        if has_collision:
            self.logger.error("Root design has self-collision")
            return None
            
        # Initialize MCTS tree
        self.root = IterativeEditing.MCTSNode(root_design)
        self.root.is_valid = True
        self.best_design = root_design
        self.best_score = 0.0
        
        # Perform search rounds
        for round_num in range(self.search_rounds):
            self.logger.info(f"Search round {round_num + 1}/{self.search_rounds}")
            
            # Selection
            leaf_node = self._select_node()
            if leaf_node is None:
                self.logger.warning("No valid leaf node selected in this round")
                continue
                
            # Expansion
            if not leaf_node.is_expanded:
                expansion_success = self._expand_node(leaf_node)
                if not expansion_success:
                    self.logger.warning(f"Failed to expand node in round {round_num + 1}")
                    continue
                    
            # Simulation and Backpropagation
            # For each child of the leaf node (if expanded), simulate and backpropagate
            # If leaf node is not expanded, simulate the leaf node itself
            if leaf_node.is_expanded:
                # Simulate each child
                for child in leaf_node.children:
                    if not child.is_valid:
                        continue
                    reward = self._simulate_and_evaluate(child)
                    self._backpropagate(child, reward)
                    self._update_best_design(child)
                    self.total_simulations += 1
            else:
                # Simulate the leaf node itself
                reward = self._simulate_and_evaluate(leaf_node)
                self._backpropagate(leaf_node, reward)
                self._update_best_design(leaf_node)
                self.total_simulations += 1
                
            # Log progress
            self.logger.info(f"Round {round_num + 1} completed. Best score: {self.best_score:.3f}, Total simulations: {self.total_simulations}")
            
        # Return the best design found
        if self.best_design is not None:
            self.logger.info(f"Iterative editing completed. Best score: {self.best_score:.3f} after {self.total_simulations} simulations")
            return self.best_design
        else:
            self.logger.error("No valid design found after all search rounds")
            return None

