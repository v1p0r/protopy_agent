# env/block_registry.py
from typing import Dict, Any, FrozenSet


class BlockRegistry:
    """
    Centralized registry for all 27 mechanical blocks in BesiegeField.
    Provides metadata lookup for block types including attachment rules, physical properties, and special behaviors.
    This class is read-only and thread-safe for use in multi-process simulation.
    """

    def __init__(self):
        """
        Initialize the block registry with predefined metadata for all 27 blocks.
        All values are hardcoded based on paper descriptions and logical inference.
        """
        self._BLOCK_METADATA = {
            "Starting Block": {
                "attachable_faces": 0,
                "is_special": False,
                "mass": 10.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Small Wooden Block": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 5.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Wooden Block": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 10.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Wooden Rod": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 1.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Log": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 15.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Steering Hinge": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 8.0,
                "friction": 0.5,
                "is_powered": True
            },
            "Steering Block": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 8.0,
                "friction": 0.5,
                "is_powered": True
            },
            "Powered Wheel": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 12.0,
                "friction": 0.8,
                "is_powered": True
            },
            "Unpowered Wheel": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 12.0,
                "friction": 0.8,
                "is_powered": False
            },
            "Large Powered Wheel": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 36.0,
                "friction": 0.8,
                "is_powered": True
            },
            "Large Unpowered Wheel": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 36.0,
                "friction": 0.8,
                "is_powered": False
            },
            "Small Wheel": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 6.0,
                "friction": 0.7,
                "is_powered": False
            },
            "Roller Wheel": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 4.0,
                "friction": 0.7,
                "is_powered": False
            },
            "Universal Joint": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 5.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Hinge": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 5.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Ball Joint": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 5.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Axle Connector": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 5.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Suspension": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 8.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Rotating Block": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 10.0,
                "friction": 0.5,
                "is_powered": True
            },
            "Grabber": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 8.0,
                "friction": 0.5,
                "is_powered": True
            },
            "Boulder": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 50.0,
                "friction": 0.6,
                "is_powered": False
            },
            "Grip Pad": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 5.0,
                "friction": 1.0,
                "is_powered": False
            },
            "Elastic Pad": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 5.0,
                "friction": 0.3,
                "is_powered": False
            },
            "Container": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 10.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Spring": {
                "attachable_faces": 0,
                "is_special": True,
                "mass": 2.0,
                "friction": 0.4,
                "is_powered": False
            },
            "Brace": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 8.0,
                "friction": 0.5,
                "is_powered": False
            },
            "Ballast": {
                "attachable_faces": 6,
                "is_special": False,
                "mass": 50.0,
                "friction": 0.5,
                "is_powered": False
            }
        }

        # Validate that all 27 blocks are present
        if len(self._BLOCK_METADATA) != 27:
            raise ValueError(f"BlockRegistry initialized with {len(self._BLOCK_METADATA)} blocks, expected 27")

        # Precompute sets for O(1) lookups
        self._valid_block_names: FrozenSet[str] = frozenset(self._BLOCK_METADATA.keys())
        self._special_blocks: FrozenSet[str] = frozenset(
            block_name for block_name, info in self._BLOCK_METADATA.items() if info["is_special"]
        )

        # Ensure Spring is the only special block
        if self._special_blocks != {"Spring"}:
            raise ValueError(f"Expected only 'Spring' as special block, got {self._special_blocks}")

    def get_block_info(self, block_type: str) -> Dict[str, Any]:
        """
        Retrieve full metadata for a block type.
        
        Args:
            block_type (str): Exact name of the block (e.g., "Powered Wheel")
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "attachable_faces": int (number of faces available for attachment)
                - "is_special": bool (True if block has non-standard attachment rules)
                - "mass": float (mass in kg)
                - "friction": float (coefficient of friction, 0.0-1.0)
                - "is_powered": bool (True if block receives control commands)
                
        Raises:
            ValueError: If block_type is not in the registry
        """
        if block_type not in self._valid_block_names:
            raise ValueError(f"Unknown block type: '{block_type}'. Valid blocks: {sorted(self._valid_block_names)}")
        return self._BLOCK_METADATA[block_type].copy()  # Return copy to prevent external mutation

    def is_special_block(self, block_type: str) -> bool:
        """
        Check if a block is special (i.e., violates standard attachment rules).
        
        Args:
            block_type (str): Exact name of the block
            
        Returns:
            bool: True if the block is special (currently only "Spring"), False otherwise
            
        Raises:
            ValueError: If block_type is not in the registry
        """
        if block_type not in self._valid_block_names:
            raise ValueError(f"Unknown block type: '{block_type}'")
        return block_type in self._special_blocks

    def get_attachable_faces(self, block_type: str) -> int:
        """
        Get the number of attachable faces for a block.
        
        Args:
            block_type (str): Exact name of the block
            
        Returns:
            int: Number of attachable faces (0-6)
            
        Raises:
            ValueError: If block_type is not in the registry
        """
        if block_type not in self._valid_block_names:
            raise ValueError(f"Unknown block type: '{block_type}'")
        return self._BLOCK_METADATA[block_type]["attachable_faces"]
