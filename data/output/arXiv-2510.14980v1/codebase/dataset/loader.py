# dataset/loader.py
import json
import jsonlines
from typing import List, Dict, Any, Tuple, Optional
from utils.config import Config
from utils.validator import JSONValidator
from utils.logger import Logger


class DatasetLoader:
    """
    Loads and validates the cold-start dataset for RL fine-tuning and test prompts for evaluation.
    Implements strict schema validation and partitioning as specified in the paper.
    """

    def __init__(self, cold_start_dataset_path: str, test_prompts_path: str, config: Optional[Config] = None):
        """
        Initialize the DatasetLoader with file paths and optional config.
        
        Args:
            cold_start_dataset_path (str): Path to JSONL file with 9,984 (prompt, cot, machine) triples
            test_prompts_path (str): Path to JSONL file with 100 pure prompt strings
            config (Config, optional): Configuration loader. If None, loads from "config.yaml"
        """
        self.config = config or Config("config.yaml")
        self.logger = Logger(__name__)
        self.json_validator = JSONValidator()
        
        # Load configuration values
        self.cold_start_dataset_path = cold_start_dataset_path
        self.test_prompts_path = test_prompts_path
        self.num_test_prompts = self.config.get("dataset.num_test_prompts", 100)
        
        # Validate inputs
        if not isinstance(self.cold_start_dataset_path, str) or not self.cold_start_dataset_path:
            raise ValueError("cold_start_dataset_path must be a non-empty string")
        if not isinstance(self.test_prompts_path, str) or not self.test_prompts_path:
            raise ValueError("test_prompts_path must be a non-empty string")
        if not isinstance(self.num_test_prompts, int) or self.num_test_prompts <= 0:
            raise ValueError("dataset.num_test_prompts must be a positive integer")
            
        # Ensure files exist
        if not os.path.exists(self.cold_start_dataset_path):
            raise FileNotFoundError(f"Cold-start dataset file not found: {self.cold_start_dataset_path}")
        if not os.path.exists(self.test_prompts_path):
            raise FileNotFoundError(f"Test prompts file not found: {self.test_prompts_path}")

    def load_train_val_test(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        """
        Load and validate the training dataset and test prompts.
        
        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
                - train_data: List of 9,984 samples, each with keys: "prompt", "cot", "machine"
                - val_data: Empty list (paper does not specify validation set)
                - test_prompts: List of 100 prompt strings (no machines, only text)
                
        Raises:
            ValueError: If schema validation fails or file formats are incorrect
        """
        self.logger.info(f"Loading training data from: {self.cold_start_dataset_path}")
        self.logger.info(f"Loading test prompts from: {self.test_prompts_path}")
        
        train_data = []
        test_prompts = []
        
        # Load training data (9,984 triples)
        try:
            with jsonlines.open(self.cold_start_dataset_path, 'r') as reader:
                for line_num, obj in enumerate(reader, 1):
                    # Validate schema
                    if not isinstance(obj, dict):
                        self.logger.warning(f"Line {line_num}: Expected dict, got {type(obj)} - skipping")
                        continue
                        
                    required_keys = {"prompt", "cot", "machine"}
                    missing_keys = required_keys - set(obj.keys())
                    if missing_keys:
                        self.logger.warning(f"Line {line_num}: Missing keys {missing_keys} - skipping")
                        continue
                        
                    prompt = obj["prompt"]
                    cot = obj["cot"]
                    machine = obj["machine"]
                    
                    # Validate prompt
                    if not isinstance(prompt, str) or not prompt.strip():
                        self.logger.warning(f"Line {line_num}: Invalid prompt - skipping")
                        continue
                        
                    # Validate cot
                    if not isinstance(cot, str) or not cot.strip():
                        self.logger.warning(f"Line {line_num}: Invalid CoT - skipping")
                        continue
                        
                    # Validate machine: must be list
                    if not isinstance(machine, list):
                        self.logger.warning(f"Line {line_num}: Machine must be a list - skipping")
                        continue
                        
                    # Validate machine structure using ConstructionTree schema
                    is_valid, error_msg = self.json_validator.validate_construction_tree(machine)
                    if not is_valid:
                        self.logger.error(f"Line {line_num}: Invalid machine structure: {error_msg} - skipping")
                        continue
                        
                    # Add valid sample
                    train_data.append({
                        "prompt": prompt,
                        "cot": cot,
                        "machine": machine
                    })
                    
        except jsonlines.InvalidLineError as e:
            raise ValueError(f"Invalid JSONL format in {self.cold_start_dataset_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to read cold-start dataset: {str(e)}")
            
        # Validate training data size
        if len(train_data) == 0:
            raise ValueError(f"No valid training samples found in {self.cold_start_dataset_path}")
        if len(train_data) != 9984:
            self.logger.warning(f"Expected 9984 training samples, but found {len(train_data)}")
            
        # Load test prompts (100 pure prompts)
        try:
            with jsonlines.open(self.test_prompts_path, 'r') as reader:
                for line_num, obj in enumerate(reader, 1):
                    # Each line should be a dict with "prompt" key
                    if not isinstance(obj, dict):
                        self.logger.warning(f"Test prompts line {line_num}: Expected dict, got {type(obj)} - skipping")
                        continue
                        
                    if "prompt" not in obj:
                        self.logger.warning(f"Test prompts line {line_num}: Missing 'prompt' key - skipping")
                        continue
                        
                    prompt = obj["prompt"]
                    if not isinstance(prompt, str) or not prompt.strip():
                        self.logger.warning(f"Test prompts line {line_num}: Invalid prompt string - skipping")
                        continue
                        
                    test_prompts.append(prompt)
                    
        except jsonlines.InvalidLineError as e:
            raise ValueError(f"Invalid JSONL format in {self.test_prompts_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Failed to read test prompts: {str(e)}")
            
        # Validate test prompts size
        if len(test_prompts) != self.num_test_prompts:
            raise ValueError(f"Expected {self.num_test_prompts} test prompts, but found {len(test_prompts)} in {self.test_prompts_path}")
            
        # Log summary
        self.logger.info(f"Successfully loaded {len(train_data)} training samples and {len(test_prompts)} test prompts")
        
        # Return train, val (empty), test
        return train_data, [], test_prompts
