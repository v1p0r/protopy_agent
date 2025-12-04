# env/besiegefield.py
import pybullet as p
import pybullet_data
import numpy as np
import multiprocessing
import time
from typing import List, Dict, Any, Optional, Tuple
from utils.config import Config
from env.block_registry import BlockRegistry
from representation.construction_tree import ConstructionTree
import logging


class BesiegeFieldSimulator:
    """
    Physics simulation engine for BesiegeField environment.
    Simulates rigid-body dynamics with gravity and elastic collisions.
    Builds machines from ConstructionTree, logs state at 0.2s intervals, and detects collisions/breakage.
    Designed for parallel execution (8 workers) in RL training.
    """

    def __init__(self, block_list: List[str], physics_config: Dict[str, Any]):
        """
        Initialize the BesiegeField simulator with block configuration and physics parameters.
        
        Args:
            block_list (List[str]): List of 27 valid block types (must match paper's specification)
            physics_config (Dict[str, Any]): Configuration from config.yaml simulation section
                Must contain: duration_seconds, state_log_interval, gravity, collision_threshold
        """
        # Validate block_list against registry
        registry = BlockRegistry()
        valid_blocks = registry._valid_block_names
        invalid_blocks = [b for b in block_list if b not in valid_blocks]
        if invalid_blocks:
            raise ValueError(f"Invalid block types in block_list: {invalid_blocks}")
        
        # Store configuration
        self.block_list = block_list
        self.physics_config = physics_config
        
        # Extract config values with defaults (though all should be provided)
        self.duration_seconds = physics_config.get("duration_seconds", 5.0)
        self.state_log_interval = physics_config.get("state_log_interval", 0.2)
        self.gravity = physics_config.get("gravity", 9.81)
        self.collision_threshold = physics_config.get("collision_threshold", 0.01)
        self.catapult_height_threshold = physics_config.get("catapult_height_threshold", 3.0)
        
        # Simulation state
        self.blocks: Dict[int, Dict[str, Any]] = {}  # block_id -> block info
        self.state_log: List[Dict[str, Any]] = []    # List of timestep snapshots
        self.simulation_started = False
        self.simulation_ended = False
        self.timestep = 0
        self.pybullet_initialized = False
        self.block_registry = BlockRegistry()
        
        # Define break thresholds per block type (N) - inferred from paper's block properties
        self.break_thresholds = {
            "Starting Block": 50.0,
            "Small Wooden Block": 20.0,
            "Wooden Block": 30.0,
            "Wooden Rod": 5.0,      # Fragile
            "Log": 25.0,
            "Steering Hinge": 25.0,
            "Steering Block": 25.0,
            "Powered Wheel": 40.0,
            "Unpowered Wheel": 40.0,
            "Large Powered Wheel": 60.0,
            "Large Unpowered Wheel": 60.0,
            "Small Wheel": 30.0,
            "Roller Wheel": 25.0,
            "Universal Joint": 25.0,
            "Hinge": 25.0,
            "Ball Joint": 25.0,
            "Axle Connector": 25.0,
            "Suspension": 35.0,
            "Rotating Block": 50.0,
            "Grabber": 30.0,
            "Boulder": 100.0,
            "Grip Pad": 20.0,
            "Elastic Pad": 20.0,
            "Container": 30.0,
            "Spring": 15.0,         # Spring is flexible but can break under tension
            "Brace": 80.0,          # Strong reinforcement
            "Ballast": 120.0        # Very heavy, high threshold
        }
        
        # Define power parameters for powered blocks (N or N·m)
        self.power_parameters = {
            "Powered Wheel": {"force": 50.0},           # Forward force in N
            "Large Powered Wheel": {"force": 150.0},    # Larger force for larger wheel
            "Rotating Block": {"torque": 20.0},         # Torque around local y-axis in N·m
            "Steering Hinge": {"torque": 15.0},         # Rotational torque
            "Steering Block": {"torque": 15.0},         # Rotational torque
            "Grabber": {"force": 30.0}                  # Grabbing force
        }
        
        # Face normals in local coordinate system (standard: Y=front, X=right, Z=top)
        # face_id: 0=front(+Y), 1=back(-Y), 2=right(+X), 3=left(-X), 4=top(+Z), 5=bottom(-Z)
        self.face_normals = {
            0: np.array([0.0, 1.0, 0.0]),   # +Y (front)
            1: np.array([0.0, -1.0, 0.0]),  # -Y (back)
            2: np.array([1.0, 0.0, 0.0]),   # +X (right)
            3: np.array([-1.0, 0.0, 0.0]),  # -X (left)
            4: np.array([0.0, 0.0, 1.0]),   # +Z (top)
            5: np.array([0.0, 0.0, -1.0])   # -Z (bottom)
        }
        
        # Block dimensions (half-sizes for AABB) in meters
        self.block_half_sizes = {
            "Starting Block": 0.5,
            "Small Wooden Block": 0.5,
            "Wooden Block": 1.0,     # 2x1x1, length along Y
            "Wooden Rod": 0.5,       # 1m long, assume 0.5m half-size
            "Log": 1.5,              # 3m long
            "Steering Hinge": 0.5,
            "Steering Block": 0.5,
            "Powered Wheel": 1.0,    # radius=1m
            "Unpowered Wheel": 1.0,
            "Large Powered Wheel": 1.5,  # radius=3m
            "Large Unpowered Wheel": 1.5,
            "Small Wheel": 0.6,      # 1.2m long? Use as half-size
            "Roller Wheel": 0.4,     # 0.8m long
            "Universal Joint": 0.5,
            "Hinge": 0.5,
            "Ball Joint": 0.5,
            "Axle Connector": 0.5,
            "Suspension": 0.5,
            "Rotating Block": 0.5,
            "Grabber": 0.5,
            "Boulder": 0.5,
            "Grip Pad": 0.5,
            "Elastic Pad": 0.5,
            "Container": 0.5,
            "Brace": 0.5,
            "Ballast": 0.5
        }
        
        # Spring properties
        self.spring_length = 1.0  # Default length when not attached
        
        # Initialize PyBullet in DIRECT mode (headless)
        self._init_pybullet()

    def _init_pybullet(self):
        """Initialize PyBullet in headless mode."""
        if self.pybullet_initialized:
            return
            
        # Connect to PyBullet in DIRECT mode (no GUI)
        p.connect(p.DIRECT)
        
        # Set gravity
        p.setGravity(0, 0, -self.gravity)
        
        # Set simulation timestep (100 Hz)
        p.setTimeStep(0.01)
        
        # Load ground plane
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Enable collision detection
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        
        self.pybullet_initialized = True

    def _get_block_half_size(self, block_type: str) -> float:
        """Get the half-size (for AABB) of a block type."""
        return self.block_half_sizes.get(block_type, 0.5)

    def _get_block_mass(self, block_type: str) -> float:
        """Get the mass of a block type from block registry."""
        return self.block_registry.get_block_info(block_type)["mass"]

    def _get_block_friction(self, block_type: str) -> float:
        """Get the friction coefficient of a block type."""
        return self.block_registry.get_block_info(block_type)["friction"]

    def _get_block_restitution(self, block_type: str) -> float:
        """Get the restitution (elasticity) of a block type."""
        # Paper doesn't specify, use default 0.5 for most, 0.8 for elastic pad
        if block_type == "Elastic Pad":
            return 0.8
        elif block_type == "Grip Pad":
            return 0.2
        else:
            return 0.5

    def _create_box_body(self, position: List[float], orientation: List[float], 
                        half_size: float, block_type: str) -> int:
        """
        Create a box-shaped body in PyBullet with specified properties.
        
        Args:
            position: [x, y, z] in world coordinates
            orientation: [qx, qy, qz, qw] quaternion
            half_size: Half the size of the cube (for a cube, this is 0.5 for 1m side)
            block_type: Type of block for material properties
            
        Returns:
            body_id: PyBullet body ID
        """
        # Create collision shape (box)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size, half_size]
        )
        
        # Create visual shape (same as collision)
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[half_size, half_size, half_size],
            rgbaColor=[0.6, 0.4, 0.2, 1.0]  # Wooden brown
        )
        
        # Get mass and friction
        mass = self._get_block_mass(block_type)
        friction = self._get_block_friction(block_type)
        restitution = self._get_block_restitution(block_type)
        
        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        # Set friction and restitution
        p.changeDynamics(body_id, -1, lateralFriction=friction, restitution=restitution)
        
        return body_id

    def _create_wheel_body(self, position: List[float], orientation: List[float], 
                          radius: float, block_type: str) -> int:
        """
        Create a wheel-shaped body (cylinder) in PyBullet.
        
        Args:
            position: [x, y, z] in world coordinates
            orientation: [qx, qy, qz, qw] quaternion
            radius: Radius of the wheel in meters
            block_type: Type of wheel (e.g., "Powered Wheel")
            
        Returns:
            body_id: PyBullet body ID
        """
        # For wheels, use cylinder shape
        # Height is small (0.2m) for realistic wheels
        height = 0.2
        half_height = height / 2.0
        
        # Create collision shape (cylinder)
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=height
        )
        
        # Create visual shape
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=height,
            rgbaColor=[0.3, 0.3, 0.3, 1.0]  # Dark gray
        )
        
        # Get mass and friction
        mass = self._get_block_mass(block_type)
        friction = self._get_block_friction(block_type)
        restitution = self._get_block_restitution(block_type)
        
        # Create multi-body
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation
        )
        
        # Set friction and restitution
        p.changeDynamics(body_id, -1, lateralFriction=friction, restitution=restitution)
        
        return body_id

    def _create_spring_body(self, start_pos: List[float], end_pos: List[float], 
                           block_type: str) -> int:
        """
        Create a spring as a capped cylinder between two points.
        Spring has no volume and is represented as a line segment.
        
        Args:
            start_pos: [x, y, z] of first endpoint
            end_pos: [x, y, z] of second endpoint
            block_type: "Spring"
            
        Returns:
            body_id: PyBullet body ID
        """
        # Calculate length and direction
        start_vec = np.array(start_pos)
        end_vec = np.array(end_pos)
        direction = end_vec - start_vec
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            length = self.spring_length
            direction = np.array([0.0, 0.0, 1.0])
            end_vec = start_vec + direction * length
            
        # Normalize direction
        direction = direction / length
        
        # Find perpendicular vectors for cylinder orientation
        if np.abs(direction[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
            
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)
        
        # Create rotation matrix
        rotation_matrix = np.column_stack([right, up, direction])
        
        # Convert to quaternion
        quat = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        # Position is midpoint
        mid_pos = (start_vec + end_vec) / 2.0
        
        # Create cylinder with radius 0.05 and length matching distance
        radius = 0.05
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=length
        )
        
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            height=length,
            rgbaColor=[0.7, 0.7, 0.7, 1.0]  # Light gray
        )
        
        mass = self._get_block_mass(block_type)
        friction = self._get_block_friction(block_type)
        restitution = self._get_block_restitution(block_type)
        
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=mid_pos,
            baseOrientation=quat
        )
        
        p.changeDynamics(body_id, -1, lateralFriction=friction, restitution=restitution)
        
        return body_id

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
        # Using formula: v' = v + 2 * cross(q_vec, cross(q_vec, v) + qw * v)
        q_vec = q[:3]
        v = vector
        
        # First cross product: cross(q_vec, v)
        cross1 = np.cross(q_vec, v)
        # Second cross product: cross(q_vec, cross1) + qw * v
        cross2 = np.cross(q_vec, cross1) + qw * v
        # Final result
        result = v + 2.0 * cross2
        
        return result

    def _get_aabb(self, position: List[float], orientation: List[float], 
                  half_size: float, block_type: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the Axis-Aligned Bounding Box (AABB) for a block.
        
        Args:
            position: [x, y, z] center position
            orientation: [qx, qy, qz, qw] quaternion
            half_size: Half the size of the cube (for a cube, this is 0.5 for 1m side)
            block_type: Type of block for special handling
            
        Returns:
            min_point: [x_min, y_min, z_min]
            max_point: [x_max, y_max, z_max]
        """
        # For wheels, use different dimensions
        if "Wheel" in block_type:
            radius = self.block_half_sizes.get(block_type, 0.5)
            height = 0.2
            half_height = height / 2.0
            # AABB for cylinder
            min_point = np.array([position[0] - radius, position[1] - radius, position[2] - half_height])
            max_point = np.array([position[0] + radius, position[1] + radius, position[2] + half_height])
        elif block_type == "Spring":
            # Spring has no volume, so we return a tiny AABB
            min_point = np.array(position) - np.array([0.001, 0.001, 0.001])
            max_point = np.array(position) + np.array([0.001, 0.001, 0.001])
        else:
            # For boxes, compute rotated AABB
            corners = []
            half_vec = np.array([half_size, half_size, half_size])
            # All 8 corners of the box in local coordinates
            for i in range(8):
                corner_local = np.array([
                    (-1 if (i & 1) else 1) * half_vec[0],
                    (-1 if (i & 2) else 1) * half_vec[1],
                    (-1 if (i & 4) else 1) * half_vec[2]
                ])
                # Rotate corner by quaternion
                corner_world = self._rotate_vector(corner_local, orientation)
                corner_world += np.array(position)
                corners.append(corner_world)
            
            corners = np.array(corners)
            min_point = np.min(corners, axis=0)
            max_point = np.max(corners, axis=0)
            
        return min_point, max_point

    def build_from_tree(self, construction_tree: ConstructionTree) -> bool:
        """
        Build a machine from a validated ConstructionTree.
        Places blocks in 3D space according to attachment relationships.
        
        Args:
            construction_tree (ConstructionTree): Validated construction tree
            
        Returns:
            bool: True if all blocks placed successfully, False otherwise
        """
        # Ensure PyBullet is initialized
        self._init_pybullet()
        
        # Clear previous state
        self.blocks = {}
        self.state_log = []
        self.simulation_started = False
        self.simulation_ended = False
        self.timestep = 0
        
        # Get construction tree data
        blocks_data = construction_tree.to_json()
        
        # Map block_id to PyBullet body_id
        body_map = {}
        
        # Place root block (id=0)
        root_block = blocks_data[0]
        if root_block["type"] != "Starting Block":
            return False
            
        # Place at origin with identity orientation
        position = [0.0, 0.0, 0.0]
        orientation = [0.0, 0.0, 0.0, 1.0]
        half_size = self._get_block_half_size("Starting Block")
        
        # Create body
        body_id = self._create_box_body(position, orientation, half_size, "Starting Block")
        body_map[0] = body_id
        
        # Store block info
        self.blocks[0] = {
            "type": "Starting Block",
            "body_id": body_id,
            "position": position,
            "orientation": orientation,
            "parent": None,
            "child_ids": [],
            "attach_face": None,
            "is_powered": False,
            "is_special": False,
            "integrity": 1.0,
            "velocity": (0.0, 0.0, 0.0),
            "angular_velocity": (0.0, 0.0, 0.0)
        }
        
        # Process remaining blocks
        for block in blocks_data[1:]:
            block_id = block["id"]
            block_type = block["type"]
            is_special = self.block_registry.is_special_block(block_type)
            
            # Get parent information
            if is_special and block_type == "Spring":
                # Spring has two parents
                parent_a_id = block["parent_a"]
                parent_b_id = block["parent_b"]
                
                # Get parent positions and orientations
                parent_a_info = self.blocks[parent_a_id]
                parent_b_info = self.blocks[parent_b_id]
                
                parent_a_pos = np.array(parent_a_info["position"])
                parent_a_ori = np.array(parent_a_info["orientation"])
                parent_b_pos = np.array(parent_b_info["position"])
                parent_b_ori = np.array(parent_b_info["orientation"])
                
                # Compute attachment points
                face_id_a = block["face_id_a"]
                face_id_b = block["face_id_b"]
                
                # Get offset from parent's attachment face
                parent_a_half_size = self._get_block_half_size(parent_a_info["type"])
                parent_b_half_size = self._get_block_half_size(parent_b_info["type"])
                
                # Get attachment face normal in parent's local space
                face_normal_a = self.face_normals[face_id_a]
                face_normal_b = self.face_normals[face_id_b]
                
                # Transform to world space
                face_offset_a = self._rotate_vector(face_normal_a * parent_a_half_size, parent_a_ori)
                face_offset_b = self._rotate_vector(face_normal_b * parent_b_half_size, parent_b_ori)
                
                # Attachment points in world space
                attach_point_a = parent_a_pos + face_offset_a
                attach_point_b = parent_b_pos + face_offset_b
                
                # Create spring body between two points
                spring_body_id = self._create_spring_body(attach_point_a.tolist(), attach_point_b.tolist(), "Spring")
                body_map[block_id] = spring_body_id
                
                # Spring has no orientation (we'll use the line direction)
                spring_midpoint = (attach_point_a + attach_point_b) / 2.0
                
                # Store spring info
                self.blocks[block_id] = {
                    "type": "Spring",
                    "body_id": spring_body_id,
                    "position": spring_midpoint.tolist(),
                    "orientation": [0.0, 0.0, 0.0, 1.0],  # Default orientation
                    "parent_a": parent_a_id,
                    "parent_b": parent_b_id,
                    "face_id_a": face_id_a,
                    "face_id_b": face_id_b,
                    "is_powered": False,
                    "is_special": True,
                    "integrity": 1.0,
                    "velocity": (0.0, 0.0, 0.0),
                    "angular_velocity": (0.0, 0.0, 0.0)
                }
                
                # Update parent's child lists
                self.blocks[parent_a_id]["child_ids"].append(block_id)
                self.blocks[parent_b_id]["child_ids"].append(block_id)
                
            else:
                # Regular block with single parent
                parent_id = block["parent"]
                face_id = block["face_id"]
                
                # Get parent info
                parent_info = self.blocks[parent_id]
                parent_pos = np.array(parent_info["position"])
                parent_ori = np.array(parent_info["orientation"])
                
                # Get half-size of current block
                half_size = self._get_block_half_size(block_type)
                
                # Compute attachment point: from parent's attachment face
                face_normal = self.face_normals[face_id]
                face_offset = self._rotate_vector(face_normal * half_size, parent_ori)
                attach_point = parent_pos + face_offset
                
                # Compute orientation: align child's attachment face with parent's face
                # Child's local +Y should align with parent's attachment face normal
                # We need to rotate the child so that its local +Y becomes the attachment direction
                # The default orientation has +Y as front
                # So we need a rotation that maps [0,1,0] to face_normal
                target = face_normal
                source = np.array([0.0, 1.0, 0.0])
                
                # Compute rotation to align source with target
                if np.dot(source, target) > 0.9999:
                    # Already aligned
                    child_ori = [0.0, 0.0, 0.0, 1.0]
                elif np.dot(source, target) < -0.9999:
                    # Opposite direction: rotate 180° around any perpendicular axis
                    if np.abs(source[0]) < 0.9:
                        axis = np.cross(source, [1.0, 0.0, 0.0])
                    else:
                        axis = np.cross(source, [0.0, 1.0, 0.0])
                    axis = axis / np.linalg.norm(axis)
                    angle = np.pi
                    child_ori = self._axis_angle_to_quaternion(axis, angle)
                else:
                    # General case
                    axis = np.cross(source, target)
                    axis = axis / np.linalg.norm(axis)
                    angle = np.arccos(np.dot(source, target))
                    child_ori = self._axis_angle_to_quaternion(axis, angle)
                
                # Create body
                if "Wheel" in block_type:
                    radius = self._get_block_half_size(block_type)
                    body_id = self._create_wheel_body(attach_point.tolist(), child_ori, radius, block_type)
                else:
                    body_id = self._create_box_body(attach_point.tolist(), child_ori, half_size, block_type)
                
                body_map[block_id] = body_id
                
                # Store block info
                self.blocks[block_id] = {
                    "type": block_type,
                    "body_id": body_id,
                    "position": attach_point.tolist(),
                    "orientation": child_ori,
                    "parent": parent_id,
                    "child_ids": [],
                    "attach_face": face_id,
                    "is_powered": self.block_registry.get_block_info(block_type)["is_powered"],
                    "is_special": is_special,
                    "integrity": 1.0,
                    "velocity": (0.0, 0.0, 0.0),
                    "angular_velocity": (0.0, 0.0, 0.0)
                }
                
                # Update parent's child list
                self.blocks[parent_id]["child_ids"].append(block_id)
        
        # Store body_map for reference
        self.body_map = body_map
        
        return True

    def _axis_angle_to_quaternion(self, axis: np.ndarray, angle: float) -> List[float]:
        """
        Convert axis-angle representation to quaternion.
        
        Args:
            axis: np.ndarray of shape (3,) - rotation axis
            angle: float - rotation angle in radians
            
        Returns:
            List[float]: quaternion [qx, qy, qz, qw]
        """
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)
        
        qx = axis[0] * sin_half
        qy = axis[1] * sin_half
        qz = axis[2] * sin_half
        qw = cos_half
        
        return [qx, qy, qz, qw]

    def check_self_collision(self) -> bool:
        """
        Check for self-collision between blocks before simulation starts.
        Uses AABB overlap detection with collision_threshold expansion.
        
        Returns:
            bool: True if no self-collision detected, False otherwise
        """
        # Get all blocks
        block_ids = list(self.blocks.keys())
        
        # For each pair of blocks
        for i in range(len(block_ids)):
            for j in range(i + 1, len(block_ids)):
                block_id_i = block_ids[i]
                block_id_j = block_ids[j]
                
                block_i = self.blocks[block_id_i]
                block_j = self.blocks[block_id_j]
                
                # Skip Spring blocks (no volume)
                if block_i["type"] == "Spring" or block_j["type"] == "Spring":
                    continue
                
                # Get AABB for both blocks
                half_size_i = self._get_block_half_size(block_i["type"])
                half_size_j = self._get_block_half_size(block_j["type"])
                
                min_point_i, max_point_i = self._get_aabb(
                    block_i["position"], block_i["orientation"], half_size_i, block_i["type"]
                )
                min_point_j, max_point_j = self._get_aabb(
                    block_j["position"], block_j["orientation"], half_size_j, block_j["type"]
                )
                
                # Expand AABB by collision_threshold to account for numerical precision
                expand = self.collision_threshold
                min_point_i -= expand
                max_point_i += expand
                min_point_j -= expand
                max_point_j += expand
                
                # Check AABB overlap
                if (min_point_i[0] < max_point_j[0] and max_point_i[0] > min_point_j[0] and
                    min_point_i[1] < max_point_j[1] and max_point_i[1] > min_point_j[1] and
                    min_point_i[2] < max_point_j[2] and max_point_i[2] > min_point_j[2]):
                    return False  # Collision detected
                    
        return True  # No collision

    def simulate(self) -> None:
        """
        Run physics simulation for duration_seconds at 100Hz (0.01s timestep).
        Log state at state_log_interval (0.2s) intervals.
        Activate powered blocks at t=2s.
        Detect and record breakage.
        """
        if not self.pybullet_initialized:
            self._init_pybullet()
            
        # Reset simulation state
        self.state_log = []
        self.simulation_started = True
        self.simulation_ended = False
        self.timestep = 0
        
        # Total steps: duration_seconds / timestep
        total_steps = int(self.duration_seconds / 0.01)
        log_interval_steps = int(self.state_log_interval / 0.01)  # 0.2s = 20 steps
        
        # Start simulation
        for step in range(total_steps):
            p.stepSimulation()
            self.timestep += 1
            
            # Check for breakage every step
            for block_id, block_info in self.blocks.items():
                if block_info["integrity"] <= 0:
                    continue  # Already broken
                    
                # Get contact points
                contact_points = p.getContactPoints(bodyA=block_info["body_id"])
                
                # Check if any contact force exceeds threshold
                max_force = 0.0
                for contact in contact_points:
                    # Contact force is the normal force
                    force = contact[9]  # Normal force
                    max_force = max(max_force, force)
                
                # If force exceeds threshold, break the block
                if max_force > self.break_thresholds.get(block_info["type"], 50.0):
                    block_info["integrity"] = 0.0
                    
                    # Apply a small impulse to simulate breakage
                    # This is not in the paper but helps with simulation stability
                    p.applyExternalForce(
                        block_info["body_id"], 
                        -1, 
                        [0.0, 0.0, 10.0], 
                        [0.0, 0.0, 0.0], 
                        p.WORLD_FRAME
                    )
            
            # Activate powered blocks at t=2s (step 200)
            if step == 200:  # 2s = 200 * 0.01s
                for block_id, block_info in self.blocks.items():
                    if block_info["is_powered"]:
                        body_id = block_info["body_id"]
                        block_type = block_info["type"]
                        
                        if block_type in self.power_parameters:
                            params = self.power_parameters[block_type]
                            
                            if "force" in params:
                                # Apply force in forward direction (local +Z)
                                # Get current orientation
                                _, orientation = p.getBasePositionAndOrientation(body_id)
                                # Forward direction in world space
                                forward = self._rotate_vector(np.array([0.0, 0.0, 1.0]), orientation)
                                force = params["force"] * forward
                                p.applyExternalForce(
                                    body_id, 
                                    -1, 
                                    force.tolist(), 
                                    [0.0, 0.0, 0.0], 
                                    p.WORLD_FRAME
                                )
                                
                            elif "torque" in params:
                                # Apply torque around local y-axis
                                # Get current orientation
                                _, orientation = p.getBasePositionAndOrientation(body_id)
                                # Y-axis in world space
                                y_axis = self._rotate_vector(np.array([0.0, 1.0, 0.0]), orientation)
                                torque = params["torque"] * y_axis
                                p.applyExternalTorque(
                                    body_id, 
                                    -1, 
                                    torque.tolist(), 
                                    p.WORLD_FRAME
                                )
            
            # Log state at intervals
            if step % log_interval_steps == 0:
                timestep_snapshot = {
                    "timestep": step * 0.01,
                    "blocks": []
                }
                
                for block_id, block_info in self.blocks.items():
                    # Get position and orientation
                    pos, ori = p.getBasePositionAndOrientation(block_info["body_id"])
                    # Get velocity
                    lin_vel, ang_vel = p.getBaseVelocity(block_info["body_id"])
                    
                    # Create block state
                    block_state = {
                        "block_id": block_id,
                        "type": block_info["type"],
                        "position": pos,
                        "orientation": ori,
                        "velocity": lin_vel,
                        "angular_velocity": ang_vel,
                        "integrity": block_info["integrity"]
                    }
                    
                    timestep_snapshot["blocks"].append(block_state)
                
                self.state_log.append(timestep_snapshot)
        
        # Mark simulation as ended
        self.simulation_ended = True

    def get_state_log(self) -> List[Dict[str, Any]]:
        """
        Get the simulation state log.
        
        Returns:
            List[Dict[str, Any]]: List of timestep snapshots, each containing block states.
            Format: [
                {
                    "timestep": 0.0,
                    "blocks": [
                        {
                            "block_id": 0,
                            "type": "Starting Block",
                            "position": [x,y,z],
                            "orientation": [qx,qy,qz,qw],
                            "velocity": [vx,vy,vz],
                            "angular_velocity": [wx,wy,wz],
                            "integrity": 1.0
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        return self.state_log.copy()

    def get_block_by_id(self, block_id: int) -> Optional[Dict[str, Any]]:
        """
        Get block information by ID.
        
        Args:
            block_id (int): Block ID
            
        Returns:
            Dict[str, Any] or None: Block info if exists, None otherwise
        """
        return self.blocks.get(block_id)

    def get_all_blocks(self) -> Dict[int, Dict[str, Any]]:
        """
        Get all blocks in the simulation.
        
        Returns:
            Dict[int, Dict[str, Any]]: Dictionary mapping block_id to block info
        """
        return self.blocks.copy()

    def __del__(self):
        """Clean up PyBullet connection."""
        if hasattr(self, 'pybullet_initialized') and self.pybullet_initialized:
            p.disconnect()
