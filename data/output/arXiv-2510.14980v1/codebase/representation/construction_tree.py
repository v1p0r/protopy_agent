# representation/construction_tree.py
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from utils.config import Config
from env.block_registry import BlockRegistry


class ConstructionTree:
    """
    Represents a machine design as a construction tree based on BesiegeField's attachment rules.
    Enforces structural validity, converts between relative (construction tree) and absolute (global position) representations.
    All blocks are assumed to be placed at default scale and orientation; no post-attachment scaling or rotation.
    """

    def __init__(self, json_data: List[Dict[str, Any]], block_registry: Optional[BlockRegistry] = None):
        """
        Initialize the ConstructionTree from a JSON list of block dictionaries.
        
        Args:
            json_data (List[Dict[str, Any]]): Ordered list of block dictionaries in construction sequence.
                Each dict must contain:
                - "type": str (block type name)
                - "id": int (0-indexed position in list)
                - "parent": int or None (parent block ID; for non-Spring blocks)
                - "face_id": int (attachment face on parent; 0-5)
                - For Spring blocks only: "parent_a", "parent_b", "face_id_a", "face_id_b"
            block_registry (BlockRegistry, optional): Block metadata registry. If None, uses singleton instance.
        """
        self._json_data = json_data
        self._block_registry = block_registry or BlockRegistry()
        self._validated = False
        self._is_valid = False
        self._validation_error = ""
        self._global_position_cache = None
        self._root_block = None
        self._id_to_block = {}
        self._children = defaultdict(list)
        self._parents = {}  # block_id -> parent_id
        self._spring_parents = {}  # block_id -> (parent_a_id, parent_b_id)
        
        # Initialize block lookup by ID for fast access
        for block in self._json_data:
            block_id = block.get("id")
            if block_id is not None:
                self._id_to_block[block_id] = block
        
        # Validate immediately on construction
        self.validate()

    def validate(self) -> Tuple[bool, str]:
        """
        Validate the construction tree against BesiegeField's rules.
        Returns (is_valid: bool, error_message: str).
        Caches result to avoid recomputation.
        
        Validation rules:
        1. Root block must be first, type="Starting Block", id=0, parent=None, face_id=None
        2. All IDs must be unique and form a contiguous sequence 0,1,...,N-1
        3. Parent ID must be < current block ID (sequential construction)
        4. Spring blocks must have exactly two parents (parent_a, parent_b) and no parent/face_id
        5. Non-Spring blocks must have exactly one parent and one face_id
        6. Face IDs must be integers in [0,5]
        7. Block types must be in BlockRegistry
        8. No cycles in parent-child graph
        9. All blocks must be reachable from root
        """
        if self._validated:
            return self._is_valid, self._validation_error

        # Reset validation state
        self._is_valid = True
        self._validation_error = ""
        self._children.clear()
        self._parents.clear()
        self._spring_parents.clear()
        self._root_block = None

        # Phase 1: Basic structural checks
        if not isinstance(self._json_data, list) or len(self._json_data) == 0:
            self._is_valid = False
            self._validation_error = "Construction tree must be a non-empty list"
            self._validated = True
            return self._is_valid, self._validation_error

        # Check root block (first block)
        root_block = self._json_data[0]
        if root_block.get("type") != "Starting Block":
            self._is_valid = False
            self._validation_error = f"Root block must be 'Starting Block', got '{root_block.get('type')}'"
            self._validated = True
            return self._is_valid, self._validation_error

        if root_block.get("id") != 0:
            self._is_valid = False
            self._validation_error = f"Root block must have id=0, got {root_block.get('id')}"
            self._validated = True
            return self._is_valid, self._validation_error

        if root_block.get("parent") is not None:
            self._is_valid = False
            self._validation_error = "Root block must have parent=None"
            self._validated = True
            return self._is_valid, self._validation_error

        if root_block.get("face_id") is not None:
            self._is_valid = False
            self._validation_error = "Root block must have face_id=None"
            self._validated = True
            return self._is_valid, self._validation_error

        # Check all IDs are integers and unique
        ids = set()
        for i, block in enumerate(self._json_data):
            block_id = block.get("id")
            if not isinstance(block_id, int):
                self._is_valid = False
                self._validation_error = f"Block at index {i} has invalid id type: {type(block_id)}"
                self._validated = True
                return self._is_valid, self._validation_error
            if block_id in ids:
                self._is_valid = False
                self._validation_error = f"Duplicate block id: {block_id}"
                self._validated = True
                return self._is_valid, self._validation_error
            ids.add(block_id)

        # Check IDs form contiguous sequence 0,1,...,N-1
        expected_ids = set(range(len(self._json_data)))
        if ids != expected_ids:
            self._is_valid = False
            self._validation_error = f"Block IDs must form contiguous sequence 0 to {len(self._json_data)-1}, got {sorted(ids)}"
            self._validated = True
            return self._is_valid, self._validation_error

        # Validate each block
        for i, block in enumerate(self._json_data):
            block_id = block["id"]
            block_type = block.get("type")
            
            # Check block type is valid
            if block_type not in self._block_registry._valid_block_names:
                self._is_valid = False
                self._validation_error = f"Invalid block type '{block_type}' at id={block_id}"
                self._validated = True
                return self._is_valid, self._validation_error

            # Check if block is special (Spring)
            is_special = self._block_registry.is_special_block(block_type)
            
            # Handle Spring blocks
            if is_special:
                # Spring must have exactly two parents and no single parent
                if "parent" in block and block["parent"] is not None:
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} cannot have 'parent' field"
                    self._validated = True
                    return self._is_valid, self._validation_error
                if "face_id" in block and block["face_id"] is not None:
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} cannot have 'face_id' field"
                    self._validated = True
                    return self._is_valid, self._validation_error
                if "parent_a" not in block or "parent_b" not in block:
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} must have both 'parent_a' and 'parent_b'"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                parent_a_id = block["parent_a"]
                parent_b_id = block["parent_b"]
                
                # Validate parent IDs
                if not isinstance(parent_a_id, int) or not isinstance(parent_b_id, int):
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} has non-integer parent_a or parent_b"
                    self._validated = True
                    return self._is_valid, self._validation_error
                if parent_a_id < 0 or parent_b_id < 0:
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} has negative parent ID"
                    self._validated = True
                    return self._is_valid, self._validation_error
                if parent_a_id >= block_id or parent_b_id >= block_id:
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} has parent(s) with ID >= current ID"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                # Validate face IDs for Spring
                if "face_id_a" not in block or "face_id_b" not in block:
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} must have both 'face_id_a' and 'face_id_b'"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                face_id_a = block["face_id_a"]
                face_id_b = block["face_id_b"]
                
                if not isinstance(face_id_a, int) or not isinstance(face_id_b, int):
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} has non-integer face_id_a or face_id_b"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                if face_id_a < 0 or face_id_a > 5 or face_id_b < 0 or face_id_b > 5:
                    self._is_valid = False
                    self._validation_error = f"Spring block at id={block_id} has face_id_a or face_id_b out of [0,5]"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                # Store Spring parent relationships
                self._spring_parents[block_id] = (parent_a_id, parent_b_id)
                self._parents[parent_a_id] = block_id
                self._parents[parent_b_id] = block_id
                self._children[parent_a_id].append(block_id)
                self._children[parent_b_id].append(block_id)
                
            else:
                # Non-Spring blocks: must have single parent and face_id
                if "parent" not in block or block["parent"] is None:
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} must have 'parent' field"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                parent_id = block["parent"]
                if not isinstance(parent_id, int):
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} has non-integer parent"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                if parent_id < 0:
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} has negative parent ID"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                if parent_id >= block_id:
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} has parent ID {parent_id} >= current ID"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                if "face_id" not in block:
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} must have 'face_id' field"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                face_id = block["face_id"]
                if not isinstance(face_id, int):
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} has non-integer face_id"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                if face_id < 0 or face_id > 5:
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} has face_id out of [0,5]"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                # Check if block type has attachable faces
                attachable_faces = self._block_registry.get_attachable_faces(block_type)
                if attachable_faces == 0:
                    self._is_valid = False
                    self._validation_error = f"Block type '{block_type}' has no attachable faces but is not Spring"
                    self._validated = True
                    return self._is_valid, self._validation_error
                
                # Record parent-child relationship
                self._parents[block_id] = parent_id
                self._children[parent_id].append(block_id)
                
                # Spring blocks cannot have single parent fields
                if "parent_a" in block or "parent_b" in block:
                    self._is_valid = False
                    self._validation_error = f"Non-Spring block at id={block_id} cannot have parent_a or parent_b"
                    self._validated = True
                    return self._is_valid, self._validation_error

        # Phase 2: Cycle detection and reachability
        # Build graph and check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def dfs(node_id: int) -> bool:
            if node_id not in visited:
                visited.add(node_id)
                rec_stack.add(node_id)
                
                # Traverse children
                for child_id in self._children[node_id]:
                    if child_id not in visited:
                        if not dfs(child_id):
                            return False
                    elif child_id in rec_stack:
                        return False  # Cycle detected
                
                rec_stack.remove(node_id)
                return True
            return True
        
        # Start DFS from root (id=0)
        if not dfs(0):
            self._is_valid = False
            self._validation_error = "Cycle detected in construction tree"
            self._validated = True
            return self._is_valid, self._validation_error
        
        # Check all blocks are reachable from root
        if len(visited) != len(self._json_data):
            unreachable = set(range(len(self._json_data))) - visited
            self._is_valid = False
            self._validation_error = f"Disconnected components: unreachable blocks {sorted(unreachable)}"
            self._validated = True
            return self._is_valid, self._validation_error
        
        # All validations passed
        self._is_valid = True
        self._validation_error = ""
        self._validated = True
        self._root_block = root_block
        return self._is_valid, self._validation_error

    def to_json(self) -> List[Dict[str, Any]]:
        """
        Return the construction tree as a JSON list of block dictionaries.
        Returns a deep copy of the original data, normalized (no modifications).
        
        Returns:
            List[Dict[str, Any]]: The original JSON data list
        """
        return [block.copy() for block in self._json_data]

    def to_global_position(self) -> List[Dict[str, Any]]:
        """
        Convert the construction tree to global position representation.
        Each block is assigned absolute position and orientation based on attachment rules.
        Uses recursive transformation from root.
        
        Returns:
            List[Dict[str, Any]]: List of block dicts with added "position" and "orientation" fields.
                Format: 
                {
                  "type": str,
                  "id": int,
                  "parent": int or None,
                  "face_id": int or None,
                  "parent_a": int (optional, for Spring),
                  "parent_b": int (optional, for Spring),
                  "face_id_a": int (optional, for Spring),
                  "face_id_b": int (optional, for Spring),
                  "position": [x, y, z] (float),
                  "orientation": [qx, qy, qz, qw] (quaternion, float)
                }
        """
        if self._global_position_cache is not None:
            return self._global_position_cache

        if not self._validated:
            self.validate()

        if not self._is_valid:
            raise ValueError("Cannot convert invalid construction tree to global position")

        # Initialize result list
        result = []
        
        # Define default attachment offsets for each block type and face
        # These are inferred from standard cuboid geometry: 1m unit blocks
        # Offsets are relative to parent's attachment face
        # Assumption: blocks are 1m cubes except where specified
        # We define offset as: from parent's attachment face center to child's local origin
        # For a cube attached to front face (face_id=0): offset = [0, 0, 0.5] (assuming parent's front is +z)
        # But since we assume default scale and no rotation, we use unit offsets
        # Note: The paper states "no post-construction scaling", so we assume default size
        # We use a mapping from block_type to face_id -> [dx, dy, dz] offset
        # This is a simplified model based on common game mechanics
        
        # Define default attachment offsets for each block type
        # For simplicity, assume all blocks are 1m cubes except:
        # - Wooden Rod: length 1m, diameter negligible -> offset along axis
        # - Log: length 3m -> offset along length
        # - Spring: length varies, we use average offset
        # We define offsets based on face_id:
        # face_id: 0=front(+z), 1=back(-z), 2=left(-x), 3=right(+x), 4=top(+y), 5=bottom(-y)
        # For a block attached to parent's front face (0), child's center is 0.5m ahead of parent's front face
        # So if parent's front face is at (px, py, pz), child's center is at (px, py, pz + 0.5)
        # But we need to consider the block's own size: child's center is offset by half its size along the attachment axis
        
        # Precompute default offsets based on block type
        # We'll use a dictionary: block_type -> face_id -> [dx, dy, dz]
        # Default: for cube blocks, offset is 0.5 along the attachment axis
        # For elongated blocks, we use half their length
        
        # Define block size defaults (half-length along primary axis)
        # We assume:
        # - Small Wooden Block: 1m cube -> half_size = 0.5
        # - Wooden Block: 2m long (2x1x1) -> half_size = 1.0 along length (assume along z)
        # - Log: 3m long -> half_size = 1.5
        # - Wooden Rod: 1m long -> half_size = 0.5
        # - Wheels: radius given in paper -> use radius as offset
        # - Spring: length not specified, use 0.5m as average
        
        # We'll define a size map for each block type
        block_size_map = {
            "Starting Block": 0.5,
            "Small Wooden Block": 0.5,
            "Wooden Block": 1.0,  # 2x1x1, length along z
            "Wooden Rod": 0.5,    # 1m long
            "Log": 1.5,           # 3m long
            "Steering Hinge": 0.5,
            "Steering Block": 0.5,
            "Powered Wheel": 1.0,  # radius=1m
            "Unpowered Wheel": 1.0,
            "Large Powered Wheel": 1.5,  # radius=3m
            "Large Unpowered Wheel": 1.5,
            "Small Wheel": 0.6,   # 1.2m long? But it's a caster, we assume offset along axis
            "Roller Wheel": 0.4,  # 0.8m long
            "Universal Joint": 0.5,
            "Hinge": 0.5,
            "Ball Joint": 0.5,
            "Axle Connector": 0.5,
            "Suspension": 0.5,
            "Rotating Block": 0.5,
            "Grabber": 0.5,
            "Boulder": 0.5,       # assumed cube
            "Grip Pad": 0.5,
            "Elastic Pad": 0.5,
            "Container": 0.5,
            "Spring": 0.5,        # average length
            "Brace": 0.5,
            "Ballast": 0.5
        }
        
        # Define face-to-axis mapping and direction
        # face_id: 0=front(+z), 1=back(-z), 2=left(-x), 3=right(+x), 4=top(+y), 5=bottom(-y)
        # For each face, the offset direction is along the normal vector
        face_directions = {
            0: [0, 0, 1],   # +z
            1: [0, 0, -1],  # -z
            2: [-1, 0, 0],  # -x
            3: [1, 0, 0],   # +x
            4: [0, 1, 0],   # +y
            5: [0, -1, 0]   # -y
        }
        
        # Define rotation matrices for child orientation based on attachment face
        # Child's local coordinate system is rotated to align with parent's face normal
        # We assume child's forward axis (z) points along the attachment normal
        # For face_id=0 (front): child's z aligns with parent's z -> identity
        # For face_id=4 (top): child's z aligns with parent's y -> rotate 90° around x
        # We use quaternion representation: [x, y, z, w]
        # Identity: [0,0,0,1]
        # Rotate 90° around x: [sin(45°), 0, 0, cos(45°)] = [0.707, 0, 0, 0.707]
        # Rotate 180° around x: [1,0,0,0]
        # Rotate 90° around y: [0, sin(45°), 0, cos(45°)] = [0, 0.707, 0, 0.707]
        # Rotate 90° around z: [0,0,sin(45°),cos(45°)] = [0,0,0.707,0.707]
        
        # Define rotation quaternions for each face
        # The child's local coordinate system is rotated so that its local +z aligns with the attachment direction
        # We define the rotation from parent's coordinate system to child's coordinate system
        # For face_id=0: child's +z = parent's +z -> no rotation: [0,0,0,1]
        # For face_id=1: child's +z = parent's -z -> rotate 180° around x: [1,0,0,0]
        # For face_id=2: child's +z = parent's -x -> rotate 90° around y: [0, 0.707, 0, 0.707]
        # For face_id=3: child's +z = parent's +x -> rotate -90° around y: [0, -0.707, 0, 0.707]
        # For face_id=4: child's +z = parent's +y -> rotate -90° around x: [0, 0, 0.707, 0.707]
        # For face_id=5: child's +z = parent's -y -> rotate 90° around x: [0, 0, -0.707, 0.707]
        
        # But note: we also need to define child's local x and y axes
        # We assume: child's local x = parent's local x, child's local y = parent's local y
        # This means we only rotate around the axis perpendicular to the attachment face
        
        # Define rotation quaternions for each face (qx, qy, qz, qw)
        face_rotations = {
            0: [0.0, 0.0, 0.0, 1.0],  # identity
            1: [1.0, 0.0, 0.0, 0.0],  # 180° around x
            2: [0.0, 0.7071, 0.0, 0.7071],  # 90° around y
            3: [0.0, -0.7071, 0.0, 0.7071],  # -90° around y
            4: [0.0, 0.0, -0.7071, 0.7071],  # -90° around x
            5: [0.0, 0.0, 0.7071, 0.7071]   # 90° around x
        }
        
        # Initialize root block
        root_block = self._json_data[0]
        root_pos = [0.0, 0.0, 0.0]  # as per paper: root at (0,0,0)
        root_ori = [0.0, 0.0, 0.0, 1.0]  # identity quaternion (aligned to +z)
        
        # Create result entry for root
        root_result = {
            "type": root_block["type"],
            "id": root_block["id"],
            "parent": root_block["parent"],
            "face_id": root_block["face_id"],
            "position": root_pos,
            "orientation": root_ori
        }
        
        # Copy all other fields from root block
        for key, value in root_block.items():
            if key not in root_result:
                root_result[key] = value
                
        result.append(root_result)
        
        # For each subsequent block, compute position and orientation
        for i in range(1, len(self._json_data)):
            block = self._json_data[i]
            block_id = block["id"]
            block_type = block["type"]
            half_size = block_size_map.get(block_type, 0.5)
            
            # Get parent information
            if block_id in self._spring_parents:
                # Spring block: two parents
                parent_a_id, parent_b_id = self._spring_parents[block_id]
                parent_a_block = result[parent_a_id]
                parent_b_block = result[parent_b_id]
                
                face_id_a = block["face_id_a"]
                face_id_b = block["face_id_b"]
                
                # Get attachment offset from parent_a
                offset_a = np.array(face_directions[face_id_a]) * half_size
                # Get parent_a's global position and orientation
                parent_a_pos = np.array(parent_a_block["position"])
                parent_a_ori = np.array(parent_a_block["orientation"])
                
                # Transform offset_a to global coordinates
                # Apply rotation to offset: R * offset
                # We use quaternion rotation: v' = q * v * q_conj
                # We'll use numpy for quaternion rotation
                offset_a_global = self._rotate_vector(offset_a, parent_a_ori)
                pos_a_global = parent_a_pos + offset_a_global
                
                # Similarly for parent_b
                offset_b = np.array(face_directions[face_id_b]) * half_size
                parent_b_pos = np.array(parent_b_block["position"])
                parent_b_ori = np.array(parent_b_block["orientation"])
                offset_b_global = self._rotate_vector(offset_b, parent_b_ori)
                pos_b_global = parent_b_pos + offset_b_global
                
                # Spring position: midpoint between the two attachment points
                spring_pos = (pos_a_global + pos_b_global) / 2.0
                
                # Spring orientation: align with the vector from parent_a to parent_b attachment points
                # Vector from attachment point A to attachment point B
                attachment_vector = pos_b_global - pos_a_global
                if np.linalg.norm(attachment_vector) < 1e-6:
                    # Degenerate case: same point, use default
                    spring_ori = [0.0, 0.0, 0.0, 1.0]
                else:
                    # Normalize vector
                    attachment_vector = attachment_vector / np.linalg.norm(attachment_vector)
                    # We want child's z-axis to align with this vector
                    # We need to construct a rotation matrix that rotates [0,0,1] to attachment_vector
                    # We'll use the "look at" method
                    # Let z_axis = attachment_vector
                    # We need two orthogonal vectors for x and y
                    # We'll assume the world up vector is [0,1,0]
                    up = np.array([0.0, 1.0, 0.0])
                    if np.abs(np.dot(attachment_vector, up)) > 0.99:
                        # Special case: attachment_vector is nearly vertical
                        right = np.array([1.0, 0.0, 0.0])
                    else:
                        right = np.cross(up, attachment_vector)
                        right = right / np.linalg.norm(right)
                    up = np.cross(attachment_vector, right)
                    up = up / np.linalg.norm(up)
                    
                    # Construct rotation matrix
                    rotation_matrix = np.column_stack([right, up, attachment_vector])
                    # Convert rotation matrix to quaternion
                    spring_ori = self._rotation_matrix_to_quaternion(rotation_matrix)
                
                # Store the two parent positions for visualization
                spring_result = {
                    "type": block_type,
                    "id": block_id,
                    "parent_a": parent_a_id,
                    "parent_b": parent_b_id,
                    "face_id_a": face_id_a,
                    "face_id_b": face_id_b,
                    "position": spring_pos.tolist(),
                    "orientation": spring_ori.tolist()
                }
                
                # Copy all other fields from original block
                for key, value in block.items():
                    if key not in spring_result:
                        spring_result[key] = value
                        
                result.append(spring_result)
                
            else:
                # Regular block: single parent
                parent_id = block["parent"]
                face_id = block["face_id"]
                
                # Get parent block result
                parent_block = result[parent_id]
                parent_pos = np.array(parent_block["position"])
                parent_ori = np.array(parent_block["orientation"])
                
                # Compute offset from parent's attachment face
                offset = np.array(face_directions[face_id]) * half_size
                # Rotate offset by parent's orientation
                offset_global = self._rotate_vector(offset, parent_ori)
                # Compute child's global position
                child_pos = parent_pos + offset_global
                
                # Compute child's orientation
                # The child's local z-axis aligns with the attachment direction
                # We use the predefined rotation for this face
                child_ori = face_rotations[face_id]
                
                # Create result entry
                child_result = {
                    "type": block_type,
                    "id": block_id,
                    "parent": parent_id,
                    "face_id": face_id,
                    "position": child_pos.tolist(),
                    "orientation": child_ori
                }
                
                # Copy all other fields from original block
                for key, value in block.items():
                    if key not in child_result:
                        child_result[key] = value
                        
                result.append(child_result)
        
        self._global_position_cache = result
        return result

    def _rotate_vector(self, vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        """
        Rotate a 3D vector by a quaternion.
        
        Args:
            vector: np.ndarray of shape (3,)
            quaternion: np.ndarray of shape (4,) [qx, qy, qz, qw]
            
        Returns:
            np.ndarray: rotated vector of shape (3,)
        """
        # Normalize quaternion
        q = quaternion / np.linalg.norm(quaternion)
        qx, qy, qz, qw = q
        
        # Quaternion rotation formula: v' = q * v * q_conj
        # q = [qx, qy, qz, qw], v = [vx, vy, vz, 0]
        # q_conj = [-qx, -qy, -qz, qw]
        # Result: v' = [vx', vy', vz']
        
        # Using formula:
        # v' = v + 2 * cross(q_vec, cross(q_vec, v) + qw * v)
        # where q_vec = [qx, qy, qz]
        
        q_vec = q[:3]
        v = vector
        
        # First cross product: cross(q_vec, v)
        cross1 = np.cross(q_vec, v)
        # Second cross product: cross(q_vec, cross1) + qw * v
        cross2 = np.cross(q_vec, cross1) + qw * v
        # Final result
        result = v + 2.0 * cross2
        
        return result

    def _rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> List[float]:
        """
        Convert a 3x3 rotation matrix to a quaternion [qx, qy, qz, qw].
        
        Args:
            rotation_matrix: np.ndarray of shape (3,3)
            
        Returns:
            List[float]: quaternion [qx, qy, qz, qw]
        """
        # From: https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        m = rotation_matrix
        trace = np.trace(m)
        
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2  # s = 4 * qx
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2  # s = 4 * qy
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2  # s = 4 * qz
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
            
        return [qx, qy, qz, qw]

    def get_root_block(self) -> Dict[str, Any]:
        """
        Get the root block dictionary.
        
        Returns:
            Dict[str, Any]: The root block (id=0) dictionary
        """
        if self._root_block is None:
            self.validate()
        return self._root_block

    def is_valid(self) -> bool:
        """
        Check if the construction tree is valid.
        
        Returns:
            bool: True if valid, False otherwise
        """
        if not self._validated:
            self.validate()
        return self._is_valid

    def get_validation_error(self) -> str:
        """
        Get the last validation error message.
        
        Returns:
            str: Error message if validation failed, empty string otherwise
        """
        if not self._validated:
            self.validate()
        return self._validation_error

    def get_block_by_id(self, block_id: int) -> Dict[str, Any]:
        """
        Get a block by its ID.
        
        Args:
            block_id (int): The ID of the block to retrieve
            
        Returns:
            Dict[str, Any]: The block dictionary
            
        Raises:
            KeyError: If block_id is not found
        """
        if block_id not in self._id_to_block:
            raise KeyError(f"Block with id={block_id} not found in construction tree")
        return self._id_to_block[block_id]

    def get_parent(self, block_id: int) -> Optional[int]:
        """
        Get the parent ID of a block.
        
        Args:
            block_id (int): The ID of the block
            
        Returns:
            Optional[int]: Parent ID, or None if block is root or Spring (has two parents)
        """
        if block_id == 0:
            return None
        if block_id in self._spring_parents:
            return None  # Spring has two parents
        return self._parents.get(block_id)

    def get_parents(self, block_id: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the parent IDs of a block (for Spring blocks, returns both; for others, returns (parent, None)).
        
        Args:
            block_id (int): The ID of the block
            
        Returns:
            Tuple[Optional[int], Optional[int]]: (parent_a, parent_b)
        """
        if block_id in self._spring_parents:
            return self._spring_parents[block_id]
        elif block_id == 0:
            return None, None
        else:
            parent = self._parents.get(block_id)
            return parent, None

    def get_children(self, block_id: int) -> List[int]:
        """
        Get the list of child block IDs.
        
        Args:
            block_id (int): The ID of the parent block
            
        Returns:
            List[int]: List of child block IDs
        """
        return self._children[block_id].copy()

    def __len__(self) -> int:
        """
        Get the number of blocks in the construction tree.
        
        Returns:
            int: Number of blocks
        """
        return len(self._json_data)

    def __str__(self) -> str:
        """
        String representation of the construction tree.
        
        Returns:
            str: Human-readable string
        """
        return f"ConstructionTree(valid={self._is_valid}, blocks={len(self._json_data)}, error='{self._validation_error}')"
