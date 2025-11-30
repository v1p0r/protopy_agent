# agent/hierarchical_design.py
import json
from typing import List, Dict, Any, Optional
from utils.config import Config
from utils.logger import Logger
from utils.parallel_sim import ParallelSimulator
from representation.construction_tree import ConstructionTree
from agent.meta_designer import MetaDesigner
from agent.designer import Designer
from reward.calculator import RewardCalculator
from env.block_registry import BlockRegistry


class HierarchicalDesign:
    """
    Orchestrates a hierarchical agentic workflow for machine design.
    Decomposes a task into 3-4 functional blocks, then autoregressively builds
    each block using 8 parallel designer agents per stage, filtering only valid designs.
    Returns the highest-performing machine after all stages.
    """

    def __init__(self, 
                 meta_designer: MetaDesigner, 
                 designer: Designer, 
                 parallel_simulator: ParallelSimulator,
                 config: Config = None):
        """
        Initialize the HierarchicalDesign workflow.
        
        Args:
            meta_designer (MetaDesigner): Agent that generates high-level functional blueprints
            designer (Designer): Agent that converts functional descriptions into detailed ConstructionTree
            parallel_simulator (ParallelSimulator): Component that validates machines in parallel
            config (Config, optional): Configuration loader. If None, loads from "config.yaml"
        """
        self.meta_designer = meta_designer
        self.designer = designer
        self.parallel_simulator = parallel_simulator
        self.config = config or Config("config.yaml")
        self.logger = Logger(__name__)
        self.block_registry = BlockRegistry()
        self.reward_calculator = RewardCalculator(task="car")  # Will be overridden per task
        
        # Load configuration values
        self.max_stages = self.config.get("agent.max_stages", 4)  # Paper: 3-4 stages
        self.builders_per_stage = self.config.get("training.hardware.parallel_sim_workers", 8)
        
        # Validate configuration
        if not isinstance(self.max_stages, int) or self.max_stages < 1:
            raise ValueError("agent.max_stages must be a positive integer")
        if not isinstance(self.builders_per_stage, int) or self.builders_per_stage < 1:
            raise ValueError("training.hardware.parallel_sim_workers must be a positive integer")
        
        # Define fallback functional block for padding
        self.fallback_block = "support_structure"

    def _create_root_tree(self) -> ConstructionTree:
        """
        Create the root construction tree with only the Starting Block.
        
        Returns:
            ConstructionTree: Valid tree with only the root block (id=0)
        """
        root_json = [
            {
                "type": "Starting Block",
                "id": 0,
                "parent": None,
                "face_id": None
            }
        ]
        tree = ConstructionTree(root_json)
        is_valid, error_msg = tree.validate()
        if not is_valid:
            raise RuntimeError(f"Failed to create valid root tree: {error_msg}")
        return tree

    def _parse_blueprint(self, blueprint: str) -> List[str]:
        """
        Parse the abstract blueprint string into a list of functional block names.
        Ensures 3-4 components, truncating or padding as needed.
        
        Args:
            blueprint (str): Comma-separated list of functional components
            
        Returns:
            List[str]: List of 3-4 functional block names
        """
        if not blueprint:
            return [self.fallback_block] * 3
            
        # Split by comma and clean whitespace
        components = [comp.strip() for comp in blueprint.split(',') if comp.strip()]
        
        # Truncate to max_stages (paper: 3-4)
        if len(components) > self.max_stages:
            components = components[:self.max_stages]
            self.logger.warning(f"Blueprint had {len(components)+1} components, truncated to {self.max_stages}")
            
        # Pad to minimum 3 if needed
        while len(components) < 3:
            components.append(self.fallback_block)
            self.logger.warning(f"Blueprint had {len(components)-1} components, padded to 3")
            
        return components

    def _generate_design_for_block(self, parent_tree: ConstructionTree, block_name: str) -> Optional[ConstructionTree]:
        """
        Generate a single design by extending the parent tree with one functional block.
        Uses the Designer agent with a prompt that references the current tree and target block.
        
        Args:
            parent_tree (ConstructionTree): The current partial machine design
            block_name (str): Name of the functional block to add (e.g., "lever arm")
            
        Returns:
            ConstructionTree or None: Generated machine, or None if invalid/unparseable
        """
        try:
            # Get current tree as JSON for prompt
            current_tree_json = json.dumps(parent_tree.to_json(), indent=2)
            
            # Build prompt: extend current machine with specified functional block
            prompt = f"""You are an expert mechanical designer in BesiegeField. You have built the following machine:

{current_tree_json}

Now, extend this machine by adding a {block_name}. The new block must be attached to an existing block in the machine. 
Use only the 27 allowed blocks from BesiegeField. Do not scale or rotate blocks post-attachment. 
Output ONLY a JSON list of block dictionaries in construction order, with all blocks including their "id", "parent", and "face_id". 
The first block must be the Starting Block (id=0). Do not include any explanations or markdown."""
            
            # Generate design
            new_tree = self.designer.generate_design(prompt)
            return new_tree
        except Exception as e:
            self.logger.error(f"Failed to generate design for block '{block_name}' with parent tree: {str(e)}")
            return None

    def _simulate_and_filter(self, candidate_trees: List[ConstructionTree], task: str) -> List[ConstructionTree]:
        """
        Simulate all candidate machines in parallel and filter for validity.
        Uses RewardCalculator to determine R_valid (file + spatial + integrity).
        
        Args:
            candidate_trees (List[ConstructionTree]): List of candidate machine designs
            task (str): Task type ("car" or "catapult") for reward calculation
            
        Returns:
            List[ConstructionTree]: List of valid machines (R_valid == True)
        """
        if not candidate_trees:
            return []
            
        # Update reward calculator for task
        self.reward_calculator = RewardCalculator(task=task, catapult_height_threshold=self.config.get("simulation.catapult_height_threshold", 3.0))
        
        # Simulate all candidates in parallel
        try:
            # Each simulation returns a dict with R_valid and R_task
            # We need to extract R_valid for filtering
            simulation_results = self.parallel_simulator.simulate_batch(candidate_trees, task)
            
            # Filter for validity (R_valid == True)
            valid_trees = []
            for i, result in enumerate(simulation_results):
                if result.get("R_valid", False):
                    valid_trees.append(candidate_trees[i])
                    
            return valid_trees
            
        except Exception as e:
            self.logger.error(f"Simulation batch failed: {str(e)}")
            return []

    def run(self, task: str) -> Optional[ConstructionTree]:
        """
        Execute the hierarchical design workflow.
        
        Args:
            task (str): Natural language task description (e.g., "Build a machine to throw a boulder as far as possible")
            
        Returns:
            ConstructionTree or None: The highest-performing valid machine, or None if no valid design found
        """
        self.logger.info(f"Starting hierarchical design for task: {task}")
        
        # Step 1: Generate high-level blueprint
        try:
            blueprint = self.meta_designer.generate_blueprint(task)
            self.logger.info(f"Generated blueprint: {blueprint}")
        except Exception as e:
            self.logger.error(f"MetaDesigner failed to generate blueprint: {str(e)}")
            return None
            
        # Step 2: Parse blueprint into functional blocks (3-4)
        functional_blocks = self._parse_blueprint(blueprint)
        self.logger.info(f"Parsed functional blocks: {functional_blocks}")
        
        # Step 3: Initialize with root block
        current_valid_trees = [self._create_root_tree()]
        self.logger.info(f"Stage 0 (root): 1 tree generated")
        
        # Step 4: Iterate over functional blocks
        for stage_idx, block_name in enumerate(functional_blocks):
            stage_num = stage_idx + 1
            self.logger.info(f"Stage {stage_num}: Adding '{block_name}' to {len(current_valid_trees)} valid designs")
            
            # Generate candidates: 8 builders per valid design
            candidate_trees = []
            for parent_tree in current_valid_trees:
                for _ in range(self.builders_per_stage):
                    new_tree = self._generate_design_for_block(parent_tree, block_name)
                    if new_tree is not None:
                        candidate_trees.append(new_tree)
                        
            self.logger.info(f"Stage {stage_num}: Generated {len(candidate_trees)} candidate machines")
            
            # Validate and filter candidates
            valid_trees = self._simulate_and_filter(candidate_trees, task)
            self.logger.info(f"Stage {stage_num}: {len(valid_trees)} valid machines after simulation")
            
            # Update current valid trees for next stage
            current_valid_trees = valid_trees
            
            # Early termination: if no valid designs, stop and return best from previous stage
            if not current_valid_trees:
                self.logger.warning(f"No valid designs after stage {stage_num}. Terminating early.")
                break
                
        # Step 5: Select best machine from final stage
        if not current_valid_trees:
            self.logger.error("No valid machines generated in any stage")
            return None
            
        # Compute scores for all valid final machines
        self.reward_calculator = RewardCalculator(task=task, catapult_height_threshold=self.config.get("simulation.catapult_height_threshold", 3.0))
        
        best_tree = None
        best_score = float('-inf')
        
        for tree in current_valid_trees:
            try:
                score, _ = self.reward_calculator.compute(tree.to_json())
                if score > best_score:
                    best_score = score
                    best_tree = tree
            except Exception as e:
                self.logger.error(f"Failed to compute score for tree: {str(e)}")
                continue
                
        if best_tree is None:
            self.logger.error("All valid machines failed score computation")
            return None
            
        self.logger.info(f"Hierarchical design completed. Best score: {best_score:.3f}")
        return best_tree
