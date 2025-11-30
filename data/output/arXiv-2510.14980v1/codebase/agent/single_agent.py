# agent/single_agent.py
import json
import re
from typing import Optional, List, Dict, Any
from utils.config import Config
from utils.logger import Logger
from env.block_registry import BlockRegistry
from representation.construction_tree import ConstructionTree
from utils.validator import JSONValidator


class SingleAgent:
    """
    Single-agent for compositional machine design: generates a machine design in one shot
    from a natural language task prompt using Chain-of-Thought (CoT) reasoning.
    Implements the baseline agent workflow described in the paper.
    """

    def __init__(self, llm_model: str, prompt_template: Optional[str] = None):
        """
        Initialize the SingleAgent with LLM model identifier and optional custom prompt template.
        
        Args:
            llm_model (str): Identifier for the LLM to use (e.g., "gemini-2.5-pro", "qwen-2.5-14b")
            prompt_template (str, optional): Custom prompt template. If None, uses default template from paper.
        """
        self.llm_model = llm_model
        self.logger = Logger(__name__)
        self.config = Config("config.yaml")
        
        # Get block list from registry (27 blocks as per paper)
        self.block_registry = BlockRegistry()
        self.block_list = sorted(list(self.block_registry._valid_block_names))
        
        # Default prompt template as specified in paper's "Single-agent setting"
        self.prompt_template = prompt_template or self._build_default_prompt_template()
        
        # LLM API parameters (hardcoded per paper's cold-start use case)
        self.max_output_length = self.config.get("model.max_output_length", 1168)
        self.temperature = 0.0  # Deterministic generation for dataset curation
        self.top_p = 1.0        # Full sampling space
        
        # Validate template contains required placeholders
        if "{task}" not in self.prompt_template:
            raise ValueError("Prompt template must contain '{task}' placeholder for task injection")
        if "{block_list}" not in self.prompt_template:
            raise ValueError("Prompt template must contain '{block_list}' placeholder for block list injection")

    def _build_default_prompt_template(self) -> str:
        """
        Build the default prompt template as specified in the paper's "Single-agent setting".
        This template enforces Chain-of-Thought reasoning and strict JSON output format.
        
        Returns:
            str: Formatted prompt template with placeholders
        """
        # Get block list as comma-separated string
        block_list_str = ", ".join([f'"{block}"' for block in self.block_list])
        
        return f"""You are an expert mechanical designer. Your task is to build a machine using only the following 27 blocks: {block_list_str}.

Construction Rules:
- Start with the "Starting Block" (ID=0). Every other block must be attached to an existing block.
- Each block has exactly one attachable face, except "Spring", which can attach to two blocks.
- Blocks cannot be scaled or rotated after attachment.
- Output must be a JSON list of block dictionaries in construction order.
- Each block must include: "type", "id", "parent", "face_id". For "Spring", use "parent_a", "parent_b", "face_id_a", "face_id_b".
- First block must be: {{"type": "Starting Block", "id": 0, "parent": null, "face_id": null}}.
- Do not include any text outside the JSON list.

Task: {{task}}

First, reason step-by-step about the design in a Chain-of-Thought (CoT) inside triple backticks:
