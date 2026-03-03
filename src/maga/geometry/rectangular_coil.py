"""
Rectangular coil geometry generator for MAGA library.

This module implements the RectangularCoil class for generating rectangular
current loops. This provides a clean interface for creating single rectangular
coils with arbitrary dimensions and orientations.

Key features:
- Arbitrary center position and orientation
- Configurable width, height, and discretization
- Constant current distribution around the perimeter
- Integration with the MAGA geometry system
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

from .base import BaseGeometry, GeometryParameters

logger = logging.getLogger(__name__)


class RectangularCoil(BaseGeometry):
    """
    Single rectangular current loop geometry generator.
    
    Creates a rectangular coil discretized into line elements along its
    perimeter. The coil lies in a plane with configurable center, dimensions,
    orientation, and current distribution.
    """
    
    def __init__(self,
                 center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 width: float = 2.0,
                 height: float = 1.0,
                 current: float = 1.0,
                 num_elements: int = 100,
                 normal_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 name: str = "rectangular_coil"):
        """
        Initialize rectangular coil geometry.
        
        Args:
            center: Center position of the coil (x, y, z)
            width: Width of the rectangle (along u-direction)
            height: Height of the rectangle (along v-direction)  
            current: Current flowing through the coil (positive for CCW when viewed along +normal)
            num_elements: Total number of line elements for discretization
            normal_vector: Normal vector to the coil plane (defines orientation)
            name: Name for this geometry
        """
        # Create parameter container
        parameters = GeometryParameters(
            center=center,
            width=width,
            height=height,
            current=current,
            num_elements=num_elements,
            normal_vector=normal_vector
        )
        
        super().__init__(name, parameters)
        
        # Validate inputs
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
        if num_elements < 4:
            raise ValueError("Need at least 4 elements for rectangular discretization")
            
        # Store geometry parameters
        self.center = np.asarray(center, dtype=float)
        self.width = float(width)
        self.height = float(height)
        self.current = float(current)
        self.num_elements_param = int(num_elements)
        self.normal_vector = np.asarray(normal_vector, dtype=float)
        
        # Normalize normal vector
        norm = np.linalg.norm(self.normal_vector)
        if norm < 1e-10:
            raise ValueError("Normal vector cannot be zero")
        self.normal_vector = self.normal_vector / norm
        
        # Compute local coordinate system for the coil plane
        self._compute_local_coordinates()
        
        # Calculate element distribution among sides
        self._calculate_element_distribution()
        
        logger.debug(f"Created rectangular coil: center={self.center}, "
                    f"size={self.width}×{self.height}, current={self.current}A, "
                    f"elements={self.num_elements_param}")
        
    def _compute_local_coordinates(self):
        """Compute local coordinate system for the coil plane."""
        # Choose an arbitrary vector not parallel to normal
        if abs(self.normal_vector[2]) < 0.9:
            temp_vec = np.array([0, 0, 1], dtype=float)
        else:
            temp_vec = np.array([1, 0, 0], dtype=float)
            
        # Compute two orthogonal vectors in the coil plane
        self.u_vector = np.cross(self.normal_vector, temp_vec)
        self.u_vector = self.u_vector / np.linalg.norm(self.u_vector)
        
        self.v_vector = np.cross(self.normal_vector, self.u_vector)
        self.v_vector = self.v_vector / np.linalg.norm(self.v_vector)
        
        logger.debug(f"Local coordinates: u={self.u_vector}, v={self.v_vector}, n={self.normal_vector}")
        
    def _calculate_element_distribution(self):
        """Calculate how to distribute elements among the four sides."""
        perimeter = 2 * (self.width + self.height)
        total_elements = self.num_elements_param
        
        # Distribute elements proportional to side lengths
        width_fraction = self.width / perimeter
        height_fraction = self.height / perimeter
        
        # Elements per side (ensuring at least 1 element per side)
        self.elements_width = max(1, int(total_elements * width_fraction))
        self.elements_height = max(1, int(total_elements * height_fraction))
        
        # Adjust to match total exactly
        current_total = 2 * (self.elements_width + self.elements_height)
        if current_total != total_elements:
            # Distribute remaining elements to longer sides
            remaining = total_elements - current_total
            if self.width >= self.height:
                self.elements_width += remaining // 2
                if remaining % 2 == 1:
                    self.elements_width += 1
            else:
                self.elements_height += remaining // 2
                if remaining % 2 == 1:
                    self.elements_height += 1
                    
        logger.debug(f"Element distribution: width sides={self.elements_width} each, "
                    f"height sides={self.elements_height} each, "
                    f"total={2*(self.elements_width + self.elements_height)}")
        
    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate line element geometry for the rectangular coil.
        
        Args:
            time: Simulation time (not used for static coils)
            
        Returns:
            Tuple of (A, B, I) arrays where:
            - A: Start points, shape (num_elements, 3)
            - B: End points, shape (num_elements, 3)
            - I: Current values, shape (num_elements,)
        """
        # Calculate total elements
        total_elements = 2 * (self.elements_width + self.elements_height)
        
        # Initialize arrays
        A = np.zeros((total_elements, 3), dtype=float)
        B = np.zeros((total_elements, 3), dtype=float)
        I = np.full(total_elements, self.current, dtype=float)
        
        # Half dimensions for corner calculations
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        
        element_idx = 0
        
        # Bottom side (left to right)
        for i in range(self.elements_width):
            u_start = -half_width + i * (self.width / self.elements_width)
            u_end = -half_width + (i + 1) * (self.width / self.elements_width)
            v_pos = -half_height
            
            A[element_idx] = self._local_to_global(u_start, v_pos)
            B[element_idx] = self._local_to_global(u_end, v_pos)
            element_idx += 1
            
        # Right side (bottom to top)
        for i in range(self.elements_height):
            u_pos = half_width
            v_start = -half_height + i * (self.height / self.elements_height)
            v_end = -half_height + (i + 1) * (self.height / self.elements_height)
            
            A[element_idx] = self._local_to_global(u_pos, v_start)
            B[element_idx] = self._local_to_global(u_pos, v_end)
            element_idx += 1
            
        # Top side (right to left)
        for i in range(self.elements_width):
            u_start = half_width - i * (self.width / self.elements_width)
            u_end = half_width - (i + 1) * (self.width / self.elements_width)
            v_pos = half_height
            
            A[element_idx] = self._local_to_global(u_start, v_pos)
            B[element_idx] = self._local_to_global(u_end, v_pos)
            element_idx += 1
            
        # Left side (top to bottom)
        for i in range(self.elements_height):
            u_pos = -half_width
            v_start = half_height - i * (self.height / self.elements_height)
            v_end = half_height - (i + 1) * (self.height / self.elements_height)
            
            A[element_idx] = self._local_to_global(u_pos, v_start)
            B[element_idx] = self._local_to_global(u_pos, v_end)
            element_idx += 1
            
        logger.debug(f"Generated {total_elements} rectangular coil elements")
        
        return A, B, I
        
    def _local_to_global(self, u: float, v: float) -> np.ndarray:
        """
        Convert local coordinates to global coordinates.
        
        Args:
            u: Coordinate along width direction
            v: Coordinate along height direction
            
        Returns:
            Global 3D coordinates
        """
        return self.center + u * self.u_vector + v * self.v_vector
        
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get spatial bounds of the rectangular coil.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        # Corner points of the rectangle
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        
        corners = [
            self._local_to_global(-half_width, -half_height),
            self._local_to_global(half_width, -half_height),
            self._local_to_global(half_width, half_height),
            self._local_to_global(-half_width, half_height)
        ]
        
        corners = np.array(corners)
        
        return {
            'x': (corners[:, 0].min(), corners[:, 0].max()),
            'y': (corners[:, 1].min(), corners[:, 1].max()),
            'z': (corners[:, 2].min(), corners[:, 2].max())
        }
        
    def get_perimeter(self) -> float:
        """Get the perimeter of the coil."""
        return 2 * (self.width + self.height)
        
    def get_area(self) -> float:
        """Get the area enclosed by the coil."""
        return self.width * self.height
        
    def get_aspect_ratio(self) -> float:
        """Get the aspect ratio (width/height) of the rectangle."""
        return self.width / self.height
        
    def get_magnetic_dipole_moment(self) -> np.ndarray:
        """
        Get the magnetic dipole moment vector.
        
        Returns:
            Dipole moment vector pointing along the normal direction
        """
        # m = I * A * n_hat
        area = self.get_area()
        return self.current * area * self.normal_vector
        
    def get_corner_positions(self) -> np.ndarray:
        """
        Get the positions of the four corners.
        
        Returns:
            Array of shape (4, 3) with corner positions
        """
        half_width = self.width / 2.0
        half_height = self.height / 2.0
        
        corners = np.array([
            self._local_to_global(-half_width, -half_height),  # Bottom-left
            self._local_to_global(half_width, -half_height),   # Bottom-right
            self._local_to_global(half_width, half_height),    # Top-right
            self._local_to_global(-half_width, half_height)    # Top-left
        ])
        
        return corners
        
    def set_current(self, current: float):
        """
        Set the current value for the coil.
        
        Args:
            current: New current value
        """
        self.current = float(current)
        self.parameters['current'] = self.current
        
        # Clear cached geometry to force regeneration
        self._geometry_A = None
        self._geometry_B = None
        self._geometry_I = None
        
        logger.debug(f"Updated coil current to {self.current}A")
        
    def set_dimensions(self, width: float, height: float):
        """
        Set the dimensions of the rectangle.
        
        Args:
            width: New width value
            height: New height value
        """
        if width <= 0 or height <= 0:
            raise ValueError("Width and height must be positive")
            
        self.width = float(width)
        self.height = float(height)
        self.parameters['width'] = self.width
        self.parameters['height'] = self.height
        
        # Recalculate element distribution
        self._calculate_element_distribution()
        
        # Clear cached geometry to force regeneration
        self._geometry_A = None
        self._geometry_B = None
        self._geometry_I = None
        
        logger.debug(f"Updated coil dimensions to {self.width}×{self.height}")
        
    def set_center(self, center: Tuple[float, float, float]):
        """
        Set the center position of the coil.
        
        Args:
            center: New center position (x, y, z)
        """
        self.center = np.asarray(center, dtype=float)
        self.parameters['center'] = center
        
        # Clear cached geometry to force regeneration
        self._geometry_A = None
        self._geometry_B = None
        self._geometry_I = None
        
        logger.debug(f"Updated coil center to {self.center}")
        
    def set_orientation(self, normal_vector: Tuple[float, float, float]):
        """
        Set the orientation of the coil.
        
        Args:
            normal_vector: New normal vector defining coil plane
        """
        self.normal_vector = np.asarray(normal_vector, dtype=float)
        
        # Normalize
        norm = np.linalg.norm(self.normal_vector)
        if norm < 1e-10:
            raise ValueError("Normal vector cannot be zero")
        self.normal_vector = self.normal_vector / norm
        
        self.parameters['normal_vector'] = normal_vector
        
        # Recompute local coordinate system
        self._compute_local_coordinates()
        
        # Clear cached geometry to force regeneration
        self._geometry_A = None
        self._geometry_B = None
        self._geometry_I = None
        
        logger.debug(f"Updated coil orientation to normal={self.normal_vector}")
        
    def __str__(self) -> str:
        """String representation of rectangular coil."""
        return (f"RectangularCoil('{self.name}', center={self.center}, "
                f"size={self.width}×{self.height}, current={self.current}A, "
                f"{2*(self.elements_width + self.elements_height)} elements)")