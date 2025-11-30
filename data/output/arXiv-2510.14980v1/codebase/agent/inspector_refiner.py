# agent/inspector_refiner.py
import json
import re
from typing import List, Dict, Any, Optional
from utils.config import Config
from utils.logger import Logger
from representation.construction_tree import ConstructionTree
from agent.refiner import Refiner
from agent.querier import ActiveEnvQuerier


class InspectorRefiner:
    """
    Self-critic agent that performs abstract critique before simulation and
    invokes refinement after simulation in the iterative editing workflow.
    Combines high-level reasoning with physical feedback to generate revised designs.
    """

    def __init__(self, llm_model: str):
        """
        Initialize the InspectorRefiner with LLM model identifier.
        
        Args:
            llm_model (str): Identifier for the LLM to use (e.g., "qwen-2.5-14b-instruct")
        """
        self.llm_model = llm_model
        self.logger = Logger(__name__)
        self.config = Config("config.yaml")
        
        # Load configuration parameters
        self.temperature = self.config.get("agent.temperature", 0.7)
        self.top_p = self.config.get("agent.top_p", 0.95)
        self.candidates_per_round = self.config.get("agent.candidates_per_round", 5)
        self.max_input_length = self.config.get("model.max_input_length", 3440)
        self.max_output_length = self.config.get("model.max_output_length", 1168)
        self.catapult_height_threshold = self.config.get("simulation.catapult_height_threshold", 3.0)
        
        # Initialize Refiner as a sub-component
        self.refiner = Refiner(llm_model=llm_model)
        
        # Get block list from registry
        try:
            from env.block_registry import BlockRegistry
            self.block_registry = BlockRegistry()
            self.valid_block_types = set(self.block_registry._valid_block_names)
        except ImportError:
            self.block_registry = None
            self.valid_block_types = set()
        
        # Build prompt templates
        self.critique_prompt_template = self._build_critique_prompt_template()
        self.refinement_prompt_template = self._build_refinement_prompt_template()

    def _build_critique_prompt_template(self) -> str:
        """
        Build the prompt template for abstract critique (pre-simulation).
        Focuses on high-level design principles without physics simulation.
        
        Returns:
            str: Formatted prompt template with placeholders
        """
        # Format block list as comma-separated quoted strings
        block_list_str = ", ".join([f'"{block}"' for block in sorted(self.valid_block_types)])
        
        return f"""You are an expert mechanical engineer reviewing a machine design blueprint for BesiegeField. Your task is to perform an abstract critique of the design before simulation.

Available blocks (use only these 27): 
{block_list_str}

Construction Rules:
- Start with the "Starting Block" (ID=0).
- Each subsequent block must be attached to exactly one existing block (except Spring, which has two parents).
- Do NOT scale or rotate blocks after attachment.
- Do NOT use any block not in the list above.

Design:
{construction_tree_json}

Critique Instructions:
1. Analyze the design abstractly. Does it logically achieve the goal?
2. Identify missing components, structural weaknesses, or logical flaws.
3. Do NOT mention physics simulation results (none available yet).
4. Focus on design principles: symmetry, balance, force transmission, stability.
5. List 3–5 critical issues in bullet points. Be specific.

Output format (JSON list of strings):
["Issue 1", "Issue 2", "Issue 3"]

Do not include any other text, explanations, or markdown.
"""

    def _build_refinement_prompt_template(self) -> str:
        """
        Build the prompt template for refinement (post-simulation).
        Combines abstract critique with physical feedback to guide revision.
        
        Returns:
            str: Formatted prompt template with placeholders
        """
        # Format block list as comma-separated quoted strings
        block_list_str = ", ".join([f'"{block}"' for block in sorted(self.valid_block_types)])
        
        return f"""You are a mechanical design expert refining a machine based on critique and simulation feedback.

Available blocks (use only these 27): 
{block_list_str}

Construction Rules:
- Start with the "Starting Block" (ID=0).
- Each subsequent block must be attached to exactly one existing block (except Spring, which has two parents).
- Do NOT scale or rotate blocks after attachment.
- Do NOT use any block not in the list above.

Original Design:
{construction_tree_json}

Abstract Critique:
{critique_summary}

Simulation Feedback:
{feedback_summary}

Constraints:
- Output exactly {self.candidates_per_round} revised construction trees in JSON list format.
- Each revision must fix at least one issue from critique or feedback.
- Do not explain. Do not add comments.
- Return only a JSON list of lists.

Example output format:
[
  [
    {{"type": "Starting Block", "id": 0, "parent": null, "face_id": null}},
    {{"type": "Ballast", "id": 1, "parent": 0, "face_id": 0}}
  ],
  [
    {{"type": "Starting Block", "id": 0, "parent": null, "face_id": null}},
    {{"type": "Ballast", "id": 1, "parent": 0, "face_id": 4}}
  ]
]
"""

    def _parse_critique_json(self, raw_output: str) -> List[str]:
        """
        Parse LLM output for abstract critique into a list of strings.
        
        Args:
            raw_output (str): Raw text output from LLM
            
        Returns:
            List[str]: List of critique points, or empty list if malformed
        """
        if not raw_output:
            return []
            
        # Remove markdown code blocks
        cleaned = re.sub(r'^\s*```json\s*', '', raw_output, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*```.*?\s*', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.DOTALL)
        
        try:
            # Try to parse as JSON array
            parsed = json.loads(cleaned)
            if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
                return parsed
            else:
                # If not list of strings, try to extract strings
                if isinstance(parsed, list):
                    return [str(item) for item in parsed if isinstance(item, str)]
                else:
                    return []
        except json.JSONDecodeError:
            # Fallback: extract lines that look like bullet points
            lines = [line.strip().lstrip('-*•').strip() for line in cleaned.split('\n') if line.strip()]
            # Filter non-empty lines
            return [line for line in lines if line]
        
        return []

    def _format_feedback_summary(self, feedback: Dict[str, Any]) -> str:
        """
        Convert feedback dict into a natural language summary for the LLM.
        
        Args:
            feedback (Dict[str, Any]): Feedback from ActiveEnvQuerier
            
        Returns:
            str: Formatted feedback summary
        """
        if not feedback:
            return "No simulation feedback available."
            
        task = feedback.get("minimal", {}).get("task", "car")
        summary = []
        
        # Add minimal feedback
        minimal = feedback.get("minimal", {})
        
        if task == "catapult":
            boulder_height = minimal.get("boulder_max_height", 0.0)
            boulder_distance = minimal.get("boulder_max_distance", 0.0)
            
            if boulder_height < self.catapult_height_threshold:
                summary.append(f"Boulder maximum height: {boulder_height:.1f}m (below threshold of {self.catapult_height_threshold}m)")
            else:
                summary.append(f"Boulder maximum height: {boulder_height:.1f}m (above threshold)")
                
            summary.append(f"Boulder maximum distance: {boulder_distance:.1f}m")
            
        elif task == "car":
            max_distance = minimal.get("max_moving_distance", 0.0)
            max_speed = minimal.get("max_speed", 0.0)
            avg_speed = minimal.get("avg_speed_per_second", 0.0)
            
            summary.append(f"Machine maximum moving distance: {max_distance:.1f}m")
            summary.append(f"Machine maximum speed: {max_speed:.1f}m/s")
            summary.append(f"Machine average speed per second: {avg_speed:.1f}m/s")
        
        # Add selective feedback
        selective = feedback.get("selective", [])
        for query in selective:
            block_id = query.get("block_id")
            block_type = query.get("block_type", "Unknown")
            query_types = query.get("query_types", [])
            time_window = query.get("time_window", [0.0, 0.0])
            data = query.get("data", [])
            
            if "integrity" in query_types and len(data) > 0:
                # Check if block broke
                broken = any(d.get("integrity", 1.0) < 1.0 for d in data)
                if broken:
                    # Find when it broke
                    first_broken_time = None
                    for d in data:
                        if d.get("integrity", 1.0) < 1.0:
                            first_broken_time = d.get("t", 0.0)
                            break
                    if first_broken_time is not None:
                        summary.append(f"Block {block_id} ({block_type}) broke at t={first_broken_time:.1f}s")
            
            if "length" in query_types and block_type == "Spring" and len(data) > 0:
                lengths = [d.get("length", 0.0) for d in data]
                min_length = min(lengths)
                max_length = max(lengths)
                if min_length < 0.2:
                    summary.append(f"Spring {block_id} compressed below minimum length: {min_length:.1f}m")
                if max_length > 2.0:
                    summary.append(f"Spring {block_id} extended beyond maximum length: {max_length:.1f}m")
                    
            if "position" in query_types and block_type == "Container" and len(data) > 0:
                y_positions = [d.get("position", [0,0,0])[1] for d in data]
                avg_y = sum(y_positions) / len(y_positions)
                if avg_y < 0.5:
                    summary.append(f"Container {block_id} positioned too low: average y={avg_y:.1f}m")
        
        return "\n".join(summary) if summary else "No specific feedback from simulation."

    def _extract_json_arrays(self, raw_output: str) -> List[List[Dict[str, Any]]]:
        """
        Extract multiple JSON arrays from raw LLM output.
        
        Args:
            raw_output (str): Raw text output from LLM
            
        Returns:
            List[List[Dict[str, Any]]]: List of parsed JSON arrays
        """
        results = []
        
        if not raw_output:
            return results
            
        # Remove markdown code blocks
        cleaned = re.sub(r'^\s*```json\s*', '', raw_output, flags=re.IGNORECASE)
        cleaned = re.sub(r'^\s*```.*?\s*', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.DOTALL)
        
        # Find all potential JSON arrays using regex
        json_pattern = r'\[\s*[\s\S]*?\s*\]'
        matches = re.findall(json_pattern, cleaned)
        
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, list):
                    results.append(parsed)
            except json.JSONDecodeError:
                continue
                
        # If no matches found, try direct parsing
        if not results:
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, list):
                    results.append(parsed)
            except json.JSONDecodeError:
                pass
                
        return results

    def _validate_and_filter_candidates(self, candidates: List[List[Dict[str, Any]]]) -> List[ConstructionTree]:
        """
        Validate candidate designs and filter out invalid or duplicate ones.
        
        Args:
            candidates (List[List[Dict[str, Any]]]): List of raw JSON arrays
            
        Returns:
            List[ConstructionTree]: List of valid, unique ConstructionTree objects
        """
        valid_trees = []
        seen_structures = set()
        
        for i, candidate in enumerate(candidates):
            # Skip empty candidates
            if not isinstance(candidate, list) or len(candidate) == 0:
                self.logger.debug(f"Skipping empty candidate {i}")
                continue
                
            # Ensure root block is first and has id=0
            if candidate[0].get("type") != "Starting Block" or candidate[0].get("id") != 0:
                # Try to fix: prepend root block if missing
                if candidate[0].get("type") != "Starting Block":
                    root_block = {"type": "Starting Block", "id": 0, "parent": None, "face_id": None}
                    # Adjust all IDs to be sequential starting from 0
                    for j, block in enumerate(candidate):
                        block["id"] = j + 1
                        if "parent" in block and block["parent"] is not None:
                            block["parent"] += 1
                        if "parent_a" in block and block["parent_a"] is not None:
                            block["parent_a"] += 1
                        if "parent_b" in block and block["parent_b"] is not None:
                            block["parent_b"] += 1
                    candidate.insert(0, root_block)
                    self.logger.warning(f"Fixed missing Starting Block in candidate {i}")
                else:
                    # ID is not 0, adjust
                    candidate[0]["id"] = 0
                    self.logger.warning(f"Fixed ID of first block to 0 in candidate {i}")
            
            # Validate all blocks are in allowed list
            valid_candidate = True
            for j, block in enumerate(candidate):
                block_type = block.get("type")
                if block_type not in self.valid_block_types:
                    self.logger.error(f"Invalid block type '{block_type}' at index {j} in candidate {i}")
                    valid_candidate = False
                    break
                    
            if not valid_candidate:
                continue
                
            # Validate IDs are sequential integers starting from 0
            expected_ids = list(range(len(candidate)))
            actual_ids = [block.get("id") for block in candidate]
            if actual_ids != expected_ids:
                # Try to fix sequential IDs
                for j, block in enumerate(candidate):
                    block["id"] = j
                self.logger.warning(f"Fixed non-sequential block IDs in candidate {i}")
            
            # Create ConstructionTree and validate
            try:
                tree = ConstructionTree(candidate)
                is_valid, error_msg = tree.validate()
                if not is_valid:
                    self.logger.error(f"Construction tree validation failed for candidate {i}: {error_msg}")
                    continue
                    
                # Check for duplicates by serializing the tree structure
                structure_signature = []
                for block in candidate:
                    signature = (
                        block.get("type"),
                        block.get("parent"),
                        block.get("parent_a"),
                        block.get("parent_b"),
                        block.get("face_id"),
                        block.get("face_id_a"),
                        block.get("face_id_b")
                    )
                    structure_signature.append(signature)
                
                # Convert to hashable tuple
                structure_tuple = tuple(structure_signature)
                
                if structure_tuple in seen_structures:
                    self.logger.debug(f"Duplicate candidate detected (candidate {i})")
                    continue
                    
                seen_structures.add(structure_tuple)
                valid_trees.append(tree)
                
            except Exception as e:
                self.logger.error(f"Failed to create ConstructionTree for candidate {i}: {str(e)}")
                continue
                
        return valid_trees

    def _call_llm(self, prompt: str) -> str:
        """
        Simulate LLM call. In production, this would be replaced with actual API integration.
        Uses the configured parameters (temperature, top_p) for generation.
        
        Args:
            prompt (str): Complete prompt string
            
        Returns:
            str: Simulated LLM response
        """
        # This is a placeholder for the actual LLM API call
        # In a real implementation, this would make an API call to the LLM service
        # For example: response = llm_api(prompt, temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_output_length)
        
        # For simulation purposes, return a deterministic response
        # In production, replace with actual LLM call
        return """[
  [
    {"type": "Starting Block", "id": 0, "parent": null, "face_id": null},
    {"type": "Ballast", "id": 1, "parent": 0, "face_id": 0},
    {"type": "Wooden Rod", "id": 2, "parent": 1, "face_id": 0},
    {"type": "Hinge", "id": 3, "parent": 2, "face_id": 0},
    {"type": "Rotating Block", "id": 4, "parent": 3, "face_id": 0},
    {"type": "Container", "id": 5, "parent": 2, "face_id": 4}
  ],
  [
    {"type": "Starting Block", "id": 0, "parent": null, "face_id": null},
    {"type": "Ballast", "id": 1, "parent": 0, "face_id": 0},
    {"type": "Wooden Rod", "id": 2, "parent": 1, "face_id": 0},
    {"type": "Hinge", "id": 3, "parent": 2, "face_id": 0},
    {"type": "Rotating Block", "id": 4, "parent": 3, "face_id": 0},
    {"type": "Container", "id": 5, "parent": 4, "face_id": 0}
  ],
  [
    {"type": "Starting Block", "id": 0, "parent": null, "face_id": null},
    {"type": "Ballast", "id": 1, "parent": 0, "face_id": 0},
    {"type": "Wooden Rod", "id": 2, "parent": 1, "face_id": 0},
    {"type": "Hinge", "id": 3, "parent": 2, "face_id": 0},
    {"type": "Rotating Block", "id": 4, "parent": 3, "face_id": 0},
    {"type": "Container", "id": 5, "parent": 1, "face_id": 4}
  ],
  [
    {"type": "Starting Block", "id": 0, "parent": null, "face_id": null},
    {"type": "Ballast", "id": 1, "parent": 0, "face_id": 0},
    {"type": "Wooden Rod", "id": 2, "parent": 1, "face_id": 0},
    {"type": "Hinge", "id": 3, "parent": 2, "face_id": 0},
    {"type": "Rotating Block", "id": 4, "parent": 3, "face_id": 0},
    {"type": "Container", "id": 5, "parent": 0, "face_id": 4}
  ],
  [
    {"type": "Starting Block", "id": 0, "parent": null, "face_id": null},
    {"type": "Ballast", "id": 1, "parent": 0, "face_id": 0},
    {"type": "Wooden Rod", "id": 2, "parent": 1, "face_id": 0},
    {"type": "Hinge", "id": 3, "parent": 2, "face_id": 0},
    {"type": "Rotating Block", "id": 4, "parent": 3, "face_id": 0},
    {"type": "Container", "id": 5, "parent": 2, "face_id": 4},
    {"type": "Brace", "id": 6, "parent": 5, "face_id": 0}
  ]
]"""

    def critique(self, design: ConstructionTree, feedback: Optional[Dict[str, Any]] = None) -> List[ConstructionTree]:
        """
        Perform abstract critique (if no feedback) or refinement (if feedback provided).
        Returns a list of ≥3 revised machine designs.
        
        Args:
            design (ConstructionTree): The current draft machine to be revised (must be valid)
            feedback (Dict[str, Any], optional): Feedback from ActiveEnvQuerier. 
                                               If None, perform abstract critique only.
        
        Returns:
            List[ConstructionTree]: List of revised machine designs (≥3 candidates)
        """
        # Validate input design
        if not design.is_valid():
            self.logger.error("InspectorRefiner received invalid design")
            return []
            
        # Convert design to JSON string
        construction_tree_json = json.dumps(design.to_json(), indent=2)
        
        # Initialize critique summary
        critique_summary = ""
        
        # Step 1: Abstract Critique (if no feedback or empty feedback)
        if feedback is None or not feedback:
            # Build critique prompt
            critique_prompt = self.critique_prompt_template.replace("{construction_tree_json}", construction_tree_json)
            
            # Truncate if too long
            if len(critique_prompt) > self.max_input_length:
                critique_prompt = critique_prompt[:self.max_input_length]
                self.logger.warning(f"Abstract critique prompt truncated to {self.max_input_length} characters")
                
            # Call LLM for critique
            raw_critique_output = self._call_llm(critique_prompt)
            
            # Parse critique
            critique_points = self._parse_critique_json(raw_critique_output)
            critique_summary = "\n".join(f"- {point}" for point in critique_points) if critique_points else "No critique generated."
            
            self.logger.info(f"Abstract critique completed: {len(critique_points)} issues identified")
            
            # If no feedback, return original design as candidate (fallback)
            # This is a fallback - in practice, we'd want to generate revisions even without feedback
            # But the paper's workflow requires refinement after critique
            # So we'll generate revisions using the critique summary as context
            # We'll use the same refinement prompt with critique only
            
        # Step 2: Refinement (always perform, even if only critique is available)
        if feedback is None:
            feedback = {}
            
        # Format feedback summary
        feedback_summary = self._format_feedback_summary(feedback)
        
        # Build refinement prompt
        refinement_prompt = self.refinement_prompt_template \
            .replace("{construction_tree_json}", construction_tree_json) \
            .replace("{critique_summary}", critique_summary) \
            .replace("{feedback_summary}", feedback_summary)
            
        # Truncate if too long
        if len(refinement_prompt) > self.max_input_length:
            refinement_prompt = refinement_prompt[:self.max_input_length]
            self.logger.warning(f"Refinement prompt truncated to {self.max_input_length} characters")
            
        # Call LLM for refinement (via Refiner)
        # Note: We're using Refiner's refine method which expects feedback dict
        # But we need to pass the critique as part of feedback for context
        # We'll create a synthetic feedback dict that includes critique
        synthetic_feedback = feedback.copy()
        synthetic_feedback["critique_summary"] = critique_summary
        
        # Call Refiner to generate candidates
        candidates = self.refiner.refine(design, synthetic_feedback)
        
        # Validate and filter candidates
        valid_candidates = self._validate_and_filter_candidates(candidates)
        
        # Ensure at least 3 valid candidates
        if len(valid_candidates) < 3:
            self.logger.warning(f"Only {len(valid_candidates)} valid candidates generated, attempting fallback")
            
            # Fallback: Generate additional candidates by perturbing the original design
            original_json = design.to_json()
            remaining = 3 - len(valid_candidates)
            
            for i in range(remaining):
                # Create a copy of the original design
                new_json = [block.copy() for block in original_json]
                
                # Make a small change: modify the attachment of the last block
                if len(new_json) > 1:
                    last_block = new_json[-1]
                    # Change parent to a different block (avoid root if possible)
                    possible_parents = [k for k in range(len(new_json)) if k != 0 and k != last_block.get("parent", -1)]
                    if possible_parents:
                        new_parent = possible_parents[i % len(possible_parents)]
                        last_block["parent"] = new_parent
                        # Change face_id to a different value
                        if "face_id" in last_block:
                            last_block["face_id"] = (last_block["face_id"] + 1) % 6
                    else:
                        # If no other parents, change face_id only
                        if "face_id" in last_block:
                            last_block["face_id"] = (last_block["face_id"] + 1) % 6
                
                # Create new ConstructionTree
                try:
                    new_tree = ConstructionTree(new_json)
                    is_valid, error_msg = new_tree.validate()
                    if is_valid:
                        # Check for duplicates
                        structure_signature = []
                        for block in new_json:
                            signature = (
                                block.get("type"),
                                block.get("parent"),
                                block.get("parent_a"),
                                block.get("parent_b"),
                                block.get("face_id"),
                                block.get("face_id_a"),
                                block.get("face_id_b")
                            )
                            structure_signature.append(signature)
                        structure_tuple = tuple(structure_signature)
                        
                        exists = False
                        for existing in valid_candidates:
                            existing_json = existing.to_json()
                            existing_signature = []
                            for b in existing_json:
                                sig = (
                                    b.get("type"),
                                    b.get("parent"),
                                    b.get("parent_a"),
                                    b.get("parent_b"),
                                    b.get("face_id"),
                                    b.get("face_id_a"),
                                    b.get("face_id_b")
                                )
                                existing_signature.append(sig)
                            existing_tuple = tuple(existing_signature)
                            
                            if structure_tuple == existing_tuple:
                                exists = True
                                break
                                
                        if not exists:
                            valid_candidates.append(new_tree)
                            if len(valid_candidates) >= 3:
                                break
                except Exception as e:
                    self.logger.warning(f"Failed to create fallback candidate: {str(e)}")
                    continue
                    
        # Final validation
        if len(valid_candidates) < 3:
            self.logger.error(f"Failed to generate 3 valid candidates after fallback. Returning {len(valid_candidates)} candidates.")
            
        self.logger.info(f"InspectorRefiner generated {len(valid_candidates)} valid revisions")
        
        return valid_candidates
