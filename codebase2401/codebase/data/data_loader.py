## data/data_loader.py
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from dataclasses import dataclass
import datasets
from transformers import AutoTokenizer, BatchEncoding
from config import CONFIG

logger = logging.getLogger(__name__)

@dataclass
class TaskConfig:
    """Configuration specific to a dataset task."""
    name: str
    dataset_name: str
    subset: Optional[str] = None
    is_classification: bool = False
    is_question_answering: bool = False
    is_summarization: bool = False
    is_instruction_tuning: bool = False
    label_column: str = "label"
    text_columns: List[str] = None

class DatasetLoader:
    """
    Dataset loader for Adaptive Pruning and Tuning (APT) framework.
    
    Handles loading and preprocessing of all datasets used in the paper:
    - GLUE tasks: SST-2, MNLI
    - Question Answering: SQuAD v2.0
    - Summarization: CNN/DailyMail
    - Instruction Tuning: Alpaca (GPT-4 generated)
    
    Key features:
    - Unified interface across multiple tasks
    - Task-specific preprocessing and tokenization
    - Support for dynamic batch sizes (train vs eval)
    - Proper handling of special formats (e.g., Alpaca prompts)
    - Integration with HuggingFace datasets library
    
    Based on paper's experimental setup in sec:experiments::Tasks.
    """
    
    def __init__(self, 
                 tokenizer: AutoTokenizer,
                 max_length: int = None,
                 batch_size: int = None,
                 eval_batch_size: int = None):
        """
        Initialize the dataset loader with tokenizer and configuration.
        
        Args:
            tokenizer: Pretrained tokenizer matching the base model
            max_length: Maximum sequence length for tokenization (default from config)
            batch_size: Training batch size per GPU (default from config)
            eval_batch_size: Evaluation batch size (default from config)
            
        Raises:
            ValueError: If required configuration is invalid or missing
        """
        # Use defaults from global config if not provided
        self.max_length = max_length or CONFIG.data.max_length
        self.batch_size = batch_size or CONFIG.data.batch_size
        self.eval_batch_size = eval_batch_size or CONFIG.data.eval_batch_size
        self.task = CONFIG.data.task.lower()
        self.tokenizer = tokenizer
        
        # Validate inputs
        if self.max_length <= 0:
            raise ValueError(f"max_length must be positive, got {self.max_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.eval_batch_size <= 0:
            raise ValueError(f"eval_batch_size must be positive, got {self.eval_batch_size}")
        
        # Define task configurations
        self.task_configs: Dict[str, TaskConfig] = {
            "sst2": TaskConfig(
                name="sst2",
                dataset_name="glue",
                subset="sst2",
                is_classification=True,
                text_columns=["sentence"]
            ),
            "mnli": TaskConfig(
                name="mnli",
                dataset_name="glue",
                subset="mnli",
                is_classification=True,
                text_columns=["premise", "hypothesis"]
            ),
            "squad_v2": TaskConfig(
                name="squad_v2",
                dataset_name="squad_v2",
                is_question_answering=True,
                text_columns=["context", "question"]
            ),
            "cnn_dm": TaskConfig(
                name="cnn_dm",
                dataset_name="cnn_dailymail",
                subset="3.0.0",
                is_summarization=True,
                text_columns=["article"],
                label_column="highlights"
            ),
            "alpaca": TaskConfig(
                name="alpaca",
                dataset_name="yahma/alpaca-cleaned",  # Common cleaned version
                is_instruction_tuning=True,
                text_columns=["instruction", "input"]
            )
        }
        
        # Get current task config
        if self.task not in self.task_configs:
            raise ValueError(f"Unsupported task: {self.task}. Supported: {list(self.task_configs.keys())}")
            
        self.current_task_config = self.task_configs[self.task]
        
        # Load raw dataset
        self.raw_datasets = self._load_dataset()
        
        logger.info(f"DatasetLoader initialized for task='{self.task}', "
                   f"max_length={self.max_length}, batch_size={self.batch_size}")
    
    def _load_dataset(self) -> datasets.DatasetDict:
        """
        Load the raw dataset based on task configuration.
        
        Returns:
            DatasetDict containing train/validation/test splits
            
        Raises:
            RuntimeError: If dataset cannot be loaded
        """
        try:
            if self.current_task_config.subset:
                dataset = datasets.load_dataset(
                    self.current_task_config.dataset_name,
                    self.current_task_config.subset,
                    cache_dir=CONFIG.output.output_dir
                )
            else:
                dataset = datasets.load_dataset(
                    self.current_task_config.dataset_name,
                    cache_dir=CONFIG.output.output_dir
                )
                
            logger.info(f"Loaded dataset '{self.current_task_config.name}' with splits: {list(dataset.keys())}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.current_task_config.dataset_name}: {e}")
            raise RuntimeError(f"Could not load dataset: {e}")
    
    def _preprocess_glue(self, examples: Dict[str, Any]) -> BatchEncoding:
        """
        Preprocess GLUE classification tasks (SST-2, MNLI).
        
        Tokenizes input texts and prepares labels.
        
        Args:
            examples: Dictionary of lists from dataset
            
        Returns:
            BatchEncoding with input_ids, attention_mask, and labels
        """
        text_cols = self.current_task_config.text_columns
        
        if len(text_cols) == 1:
            # Single sentence task (e.g., SST-2)
            result = self.tokenizer(
                examples[text_cols[0]],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        else:
            # Sentence pair task (e.g., MNLI)
            result = self.tokenizer(
                examples[text_cols[0]],
                examples[text_cols[1]],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
        
        # Add labels
        result["labels"] = examples[self.current_task_config.label_column]
        
        return result
    
    def _preprocess_squad_v2(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess SQuAD v2.0 question answering dataset.
        
        Handles both answerable and unanswerable questions by setting start/end positions.
        Uses offset mapping to align tokens with original text.
        
        Args:
            examples: Dictionary of lists from dataset
            
        Returns:
            Dictionary with input_ids, attention_mask, start_positions, end_positions
        """
        questions = [q.strip() for q in examples["question"]]
        contexts = examples["context"]
        
        # Tokenize contexts and questions together
        tokenized_examples = self.tokenizer(
            questions,
            contexts,
            truncation="only_second",  # Truncate context if too long
            max_length=self.max_length,
            stride=self.max_length // 4,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Map back to original examples
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # Initialize labels
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossible"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            # One example can give several spans, so we need to identify the one
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            # Start/end position are set to cls_index when no answer exists
            start_char = answers["answer_start"][0] if len(answers["answer_start"]) > 0 else -1
            answer_text = answers["text"][0] if len(answers["text"]) > 0 else ""
            
            if start_char >= 0:
                end_char = start_char + len(answer_text)
                
                # Find start and end tokens
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
                
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
                
                # Check if answer within context
                if not (offsets[token_start_index][0] <= start_char and 
                        offsets[token_end_index][1] >= end_char):
                    # Answer not fully in context
                    token_start_index = cls_index
                    token_end_index = cls_index
                else:
                    # Move start index to the start of the answer
                    while token_start_index < len(offsets) and \
                          offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    token_start_index -= 1
                    
                    # Move end index to the end of the answer
                    while token_end_index >= 0 and \
                          offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    token_end_index += 1
                    
            else:
                # No answer found
                token_start_index = cls_index
                token_end_index = cls_index
            
            tokenized_examples["start_positions"].append(token_start_index)
            tokenized_examples["end_positions"].append(token_end_index)
            tokenized_examples["is_impossible"].append(float(token_start_index == cls_index))
        
        return tokenized_examples
    
    def _preprocess_summarization(self, examples: Dict[str, Any]) -> BatchEncoding:
        """
        Preprocess summarization tasks (CNN/DailyMail).
        
        Formats inputs with T5-style prefix and tokenizes both source and target.
        
        Args:
            examples: Dictionary of lists from dataset
            
        Returns:
            BatchEncoding with input_ids, attention_mask, and labels
        """
        # Add prefix for T5 models
        inputs = [f"summarize: {doc}" for doc in examples["article"]]
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples[self.current_task_config.label_column],
                max_length=self.max_length // 2,  # Targets are shorter
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def _preprocess_instruction_tuning(self, examples: Dict[str, Any]) -> BatchEncoding:
        """
        Preprocess instruction tuning data (Alpaca format).
        
        Constructs prompt using template:
        ### Instruction:
        {instruction}
        
        ### Input:
        {input}
        
        ### Response:
        {output}
        
        Then tokenizes the full prompt.
        
        Args:
            examples: Dictionary of lists from dataset
            
        Returns:
            BatchEncoding with input_ids, attention_mask, and labels
        """
        INSTRUCTION_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
        
        # Format each example
        prompts = []
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            input_text = examples.get("input", [""])[i] if "input" in examples else ""
            output_text = examples["output"][i]
            
            prompt = INSTRUCTION_TEMPLATE.format(
                instruction=instruction,
                input=input_text,
                output=output_text
            )
            prompts.append(prompt)
        
        # Tokenize full prompts
        tokenized = self.tokenizer(
            prompts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # For causal LM training, labels should be same as input_ids
        # The model will compute loss only over the response part during training
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def _get_preprocessing_fn(self):
        """Get the appropriate preprocessing function based on task."""
        if self.current_task_config.is_classification:
            return self._preprocess_glue
        elif self.current_task_config.is_question_answering:
            return self._preprocess_squad_v2
        elif self.current_task_config.is_summarization:
            return self._preprocess_summarization
        elif self.current_task_config.is_instruction_tuning:
            return self._preprocess_instruction_tuning
        else:
            raise ValueError(f"No preprocessing function for task {self.task}")
    
    def load_split(self, split: str) -> DataLoader:
        """
        Load and preprocess a dataset split into a PyTorch DataLoader.
        
        Args:
            split: Split name ('train', 'validation', 'test')
            
        Returns:
            DataLoader object ready for training/evaluation
            
        Raises:
            ValueError: If split does not exist or cannot be processed
        """
        # Map common split names
        split_map = {
            "train": ["train", "training"],
            "validation": ["validation", "val", "dev"],
            "test": ["test", "testing"]
        }
        
        actual_split = None
        for key, variants in split_map.items():
            if split in variants or (key == split):
                for variant in variants:
                    if variant in self.raw_datasets:
                        actual_split = variant
                        break
                break
        
        if actual_split is None or actual_split not in self.raw_datasets:
            available = list(self.raw_datasets.keys())
            raise ValueError(f"Split '{split}' not found. Available splits: {available}")
        
        dataset = self.raw_datasets[actual_split]
        
        # Get preprocessing function
        preprocess_fn = self._get_preprocessing_fn()
        
        # Apply preprocessing
        try:
            # Disable tqdm in multiprocessing
            old_level = datasets.logging.get_verbosity()
            datasets.logging.set_verbosity(datasets.logging.ERROR)
            
            processed_dataset = dataset.map(
                preprocess_fn,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Preprocessing {split} dataset"
            )
            
            datasets.logging.set_verbosity(old_level)
            
        except Exception as e:
            logger.error(f"Failed to preprocess {split} split: {e}")
            raise ValueError(f"Preprocessing failed: {e}")
        
        # Determine batch size
        batch_size = self.batch_size if split == "train" else self.eval_batch_size
        
        # Create DataLoader
        dataloader = DataLoader(
            processed_dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,  # Avoid issues with tokenizers in subprocesses
            pin_memory=True
        )
        
        logger.info(f"Loaded {split} split with {len(processed_dataset)} examples, "
                   f"batch_size={batch_size}")
        
        return dataloader
    
    def get_num_labels(self) -> Optional[int]:
        """
        Get the number of labels for classification tasks.
        
        Returns:
            Number of unique labels, or None if not applicable
        """
        if not self.current_task_config.is_classification:
            return None
            
        # For GLUE tasks, get label list
        if self.task == "sst2":
            return 2  # Positive/Negative
        elif self.task == "mnli":
            return 3  # Entailment/Neutral/Contradiction
            
        # Try to infer from dataset
        try:
            if "train" in self.raw_datasets:
                train_dataset = self.raw_datasets["train"]
                if self.current_task_config.label_column in train_dataset.features:
                    feature = train_dataset.features[self.current_task_config.label_column]
                    if hasattr(feature, 'names'):
                        return len(feature.names)
        except:
            pass
            
        return None
    
    def get_vocab_size(self) -> int:
        """
        Get the tokenizer vocabulary size.
        
        Returns:
            Size of tokenizer vocab
        """
        return len(self.tokenizer)
    
    def get_max_length(self) -> int:
        """
        Get the maximum sequence length used for tokenization.
        
        Returns:
            Maximum length value
        """
        return self.max_length

__all__ = ['DatasetLoader']
