from typing import List, Dict, Any, Tuple, Set
from utils.block_registry import BlockRegistry


class JSONValidator:
    """
    Validates the structural and syntactic correctness of a ConstructionTree JSON representation.
    Enforces the exact schema defined in the paper for machine construction trees.
    Does not perform physics or spatial validation — only JSON structure and field constraints.
    """

    def __init__(self, valid_block_types: Set[str]):
        """
        Initialize the JSONValidator with the set of valid block types.
        
        Args:
            valid_block_types (Set[str]): Set of 27 valid block type names from BlockRegistry.
        """
        self.valid_blocks = valid_block_types
        self.required_fields = {
            "Spring": {"parent_a", "parent_b", "face_id_a", "face_id_b"},
            "default": {"parent", "face_id"}
        }
        self.forbidden_fields = {
            "Spring": {"parent", "face_id"},
            "default": {"parent_a", "parent_b", "face_id_a", "face_id_b"}
        }
        self.face_id_range = set(range(6))  # Valid face IDs: 0, 1, 2, 3, 4, 5

    def validate_construction_tree(self, json_data: List[Dict[str, Any]]) -> Tuple[bool, str]:
        """
        Validate a construction tree JSON list against the BesiegeField specification.
        
        Validation rules (as per paper):
        1. Must be a non-empty list of dictionaries.
        2. First block must be "Starting Block" with id=0 and no parent/face_id fields.
        3. All block IDs must be unique, consecutive integers from 0 to N-1.
        4. Each block must have a valid type from the 27-block registry.
        5. Non-Spring blocks must have exactly "parent" and "face_id" (both integers).
        6. Spring blocks must have exactly "parent_a", "parent_b", "face_id_a", "face_id_b" (all integers).
        7. Parent IDs must reference existing blocks with id < current block id.
        8. Face IDs must be integers in range [0, 5].
        9. Forbidden fields must not be present (e.g., parent on Spring blocks).
        
        Args:
            json_data (List[Dict[str, Any]]): List of block dictionaries in construction order.
            
        Returns:
            Tuple[bool, str]: (is_valid: bool, error_message: str)
        """
        # Phase 1: Top-level structure
        if not isinstance(json_data, list) or len(json_data) == 0:
            return False, "Construction tree must be a non-empty list."
        
        # Phase 2: Root block validation
        root_block = json_data[0]
        if not isinstance(root_block, dict):
            return False, "Root block must be a dictionary."
        
        root_type = root_block.get("type")
        if root_type != "Starting Block":
            return False, f"Root block must be 'Starting Block', got '{root_type}'"
        
        root_id = root_block.get("id")
        if not isinstance(root_id, int) or root_id != 0:
            return False, "Root block must have id=0"
        
        # Root block must not have any parent/face fields
        for forbidden_field in ["parent", "face_id", "parent_a", "parent_b", "face_id_a", "face_id_b"]:
            if forbidden_field in root_block:
                return False, f"Root block must not have '{forbidden_field}' field"
        
        # Phase 3: ID uniqueness and ordering
        ids = []
        for i, block in enumerate(json_data):
            if not isinstance(block, dict):
                return False, f"Block at index {i} is not a dictionary"
            
            block_id = block.get("id")
            if not isinstance(block_id, int):
                return False, f"Block at index {i} has invalid id type: {type(block_id)}"
            ids.append(block_id)
        
        # Check uniqueness
        if len(set(ids)) != len(ids):
            return False, "All block IDs must be unique"
        
        # Check consecutive integers from 0 to N-1
        expected_ids = list(range(len(json_data)))
        if ids != expected_ids:
            return False, f"Block IDs must be consecutive integers starting from 0: expected {expected_ids}, got {ids}"
        
        # Phase 4: Block type validation
        for i, block in enumerate(json_data):
            block_type = block.get("type")
            if not isinstance(block_type, str):
                return False, f"Block at index {i} has invalid type: must be string, got {type(block_type)}"
            if block_type not in self.valid_blocks:
                return False, f"Invalid block type '{block_type}' at index {i}. Valid types: {sorted(self.valid_blocks)}"
        
        # Phase 5: Attachment field validation per block
        for i, block in enumerate(json_data):
            block_type = block["type"]
            expected_fields = self.required_fields["Spring"] if block_type == "Spring" else self.required_fields["default"]
            forbidden_fields = self.forbidden_fields["Spring"] if block_type == "Spring" else self.forbidden_fields["default"]
            
            # Check required fields are present
            for field in expected_fields:
                if field not in block:
                    return False, f"Missing required field '{field}' in block at index {i} of type '{block_type}'"
            
            # Check forbidden fields are absent
            for field in forbidden_fields:
                if field in block:
                    return False, f"Forbidden field '{field}' present in block at index {i} of type '{block_type}'"
            
            # Validate parent and face_id values based on block type
            if block_type == "Spring":
                # Validate parent_a and parent_b
                parent_a = block["parent_a"]
                parent_b = block["parent_b"]
                if not isinstance(parent_a, int) or not isinstance(parent_b, int):
                    return False, f"parent_a and parent_b in block at index {i} must be integers"
                if parent_a >= i or parent_b >= i:
                    return False, f"parent_a and parent_b in block at index {i} must be less than current block id ({i})"
                if parent_a == parent_b:
                    return False, f"parent_a and parent_b in block at index {i} must be distinct"
                
                # Validate face_id_a and face_id_b
                face_id_a = block["face_id_a"]
                face_id_b = block["face_id_b"]
                if not isinstance(face_id_a, int) or not isinstance(face_id_b, int):
                    return False, f"face_id_a and face_id_b in block at index {i} must be integers"
                if face_id_a not in self.face_id_range or face_id_b not in self.face_id_range:
                    return False, f"face_id_a and face_id_b in block at index {i} must be in range [0,5]"
                
            else:
                # Validate parent and face_id for non-Spring blocks
                parent = block["parent"]
                face_id = block["face_id"]
                if not isinstance(parent, int):
                    return False, f"parent in block at index {i} must be an integer"
                if parent >= i:
                    return False, f"parent in block at index {i} must be less than current block id ({i})"
                if not isinstance(face_id, int):
                    return False, f"face_id in block at index {i} must be an integer"
                if face_id not in self.face_id_range:
                    return False, f"face_id in block at index {i} must be in range [0,5]"
        
        return True, "Valid construction tree."
