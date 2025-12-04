# utils/config.py
import yaml
import os
from typing import Any, Dict, Union
from utils.logger import Logger


class Config:
    """
    Centralized configuration loader and accessor for the entire system.
    Loads all hyperparameters, environmental settings, and experimental parameters from config.yaml.
    Provides type-safe, dot-notation access to nested configuration values.
    All configuration is immutable after initialization.
    """

    def __init__(self, config_file: str):
        """
        Initialize the Config instance by loading and parsing the YAML configuration file.
        
        Args:
            config_file (str): Path to the configuration file (e.g., "config.yaml")
            
        Raises:
            FileNotFoundError: If the config file does not exist
            yaml.YAMLError: If the config file is malformed
            ValueError: If required top-level keys are missing
        """
        self.logger = Logger(__name__)
        self._config_file = config_file
        self._config: Dict[str, Any] = {}
        
        # Validate config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        # Load and parse YAML
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse configuration file {config_file}: {str(e)}")
            
        # Validate that required top-level keys exist
        required_keys = [
            "training", "model", "simulation", "agent", "dataset", 
            "tasks", "logging", "output"
        ]
        
        missing_keys = [key for key in required_keys if key not in self._config]
        if missing_keys:
            raise ValueError(f"Missing required top-level configuration keys: {missing_keys}")
            
        self.logger.debug(f"Configuration loaded successfully from {config_file}")

    def get(self, key: str) -> Any:
        """
        Retrieve a configuration value using dot-notation path.
        
        Args:
            key (str): Dot-separated path to the configuration value (e.g., "training.rl_finetune.learning_rate")
            
        Returns:
            Any: The configuration value at the specified path
            
        Raises:
            KeyError: If the key path does not exist in the configuration
            ValueError: If the key is empty or None
        """
        if not key or not isinstance(key, str):
            raise ValueError("Configuration key must be a non-empty string")
            
        # Split key by dots and traverse nested dict
        keys = key.split('.')
        current = self._config
        
        for k in keys:
            if not isinstance(current, dict) or k not in current:
                raise KeyError(f"Missing required config key: '{key}'")
            current = current[k]
            
        return current

    def get_all(self) -> Dict[str, Any]:
        """
        Return a deep copy of the entire configuration dictionary.
        
        Returns:
            Dict[str, Any]: Complete configuration as a dictionary
        """
        # Return a deep copy to prevent external modification
        import copy
        return copy.deepcopy(self._config)

    def get_int(self, key: str) -> int:
        """
        Retrieve a configuration value as an integer.
        
        Args:
            key (str): Dot-separated path to the configuration value
            
        Returns:
            int: The configuration value cast to integer
            
        Raises:
            KeyError: If the key path does not exist
            ValueError: If the value cannot be converted to integer
        """
        value = self.get(key)
        if not isinstance(value, (int, float, str)):
            raise ValueError(f"Configuration value at '{key}' is not a number: {value}")
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Configuration value at '{key}' cannot be converted to int: {value}")

    def get_float(self, key: str) -> float:
        """
        Retrieve a configuration value as a float.
        
        Args:
            key (str): Dot-separated path to the configuration value
            
        Returns:
            float: The configuration value cast to float
            
        Raises:
            KeyError: If the key path does not exist
            ValueError: If the value cannot be converted to float
        """
        value = self.get(key)
        if not isinstance(value, (int, float, str)):
            raise ValueError(f"Configuration value at '{key}' is not a number: {value}")
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Configuration value at '{key}' cannot be converted to float: {value}")

    def get_bool(self, key: str) -> bool:
        """
        Retrieve a configuration value as a boolean.
        
        Args:
            key (str): Dot-separated path to the configuration value
            
        Returns:
            bool: The configuration value cast to boolean
            
        Raises:
            KeyError: If the key path does not exist
            ValueError: If the value cannot be converted to boolean
        """
        value = self.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.lower() in ('true', '1', 'yes', 'on'):
                return True
            elif value.lower() in ('false', '0', 'no', 'off'):
                return False
            else:
                raise ValueError(f"Configuration value at '{key}' cannot be converted to bool: {value}")
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"Configuration value at '{key}' cannot be converted to bool: {value}")

    def get_str(self, key: str) -> str:
        """
        Retrieve a configuration value as a string.
        
        Args:
            key (str): Dot-separated path to the configuration value
            
        Returns:
            str: The configuration value cast to string
            
        Raises:
            KeyError: If the key path does not exist
            ValueError: If the value is not a string or cannot be converted to string
        """
        value = self.get(key)
        if not isinstance(value, str):
            raise ValueError(f"Configuration value at '{key}' is not a string: {value}")
        return value

    def get_list(self, key: str) -> list:
        """
        Retrieve a configuration value as a list.
        
        Args:
            key (str): Dot-separated path to the configuration value
            
        Returns:
            list: The configuration value cast to list
            
        Raises:
            KeyError: If the key path does not exist
            ValueError: If the value is not a list
        """
        value = self.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Configuration value at '{key}' is not a list: {value}")
        return value

    def get_dict(self, key: str) -> dict:
        """
        Retrieve a configuration value as a dictionary.
        
        Args:
            key (str): Dot-separated path to the configuration value
            
        Returns:
            dict: The configuration value cast to dict
            
        Raises:
            KeyError: If the key path does not exist
            ValueError: If the value is not a dictionary
        """
        value = self.get(key)
        if not isinstance(value, dict):
            raise ValueError(f"Configuration value at '{key}' is not a dictionary: {value}")
        return value
