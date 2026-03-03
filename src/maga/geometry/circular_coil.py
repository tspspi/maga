"""
Circular coil geometry generator for MAGA library.

This module implements the CircularCoil class for generating circular current
loops. Based on the CircleGenerator from the original Jupyter notebook,
this provides a clean interface for creating single circular coils.

Key features:
- Arbitrary center position and orientation
- Configurable radius and discretization
- Constant current distribution
- Integration with the MAGA geometry system
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

from .base import BaseGeometry, GeometryParameters

logger = logging.getLogger(__name__)


class CircularCoil(BaseGeometry):
    """
    Single circular current loop geometry generator.
    
    Creates a circular coil discretized into line elements, suitable
    for magnetic field calculations. The coil lies in a plane with
    configurable center, radius, orientation, and current.
    """
    
    def __init__(self,
                 center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 radius: float = 1.0,
                 current: float = 1.0,
                 windings: int = 1,
                 num_elements: int = 100,
                 normal_vector: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 name: str = "circular_coil"):
        """
        Initialize circular coil geometry.
        
        Args:
            center: Center position of the coil (x, y, z)
            radius: Radius of the coil
            current: Current flowing through the coil (positive for CCW when viewed along +normal)
            windings: Number of windings in our coil
            num_elements: Number of line elements for discretization
            normal_vector: Normal vector to the coil plane (defines orientation)
            name: Name for this geometry
        """
        # Create parameter container
        parameters = GeometryParameters(
            center=center,
            radius=radius,
            current=current,
            windings=windings,
            num_elements=num_elements,
            normal_vector=normal_vector
        )
        
        super().__init__(name, parameters)
        
        # Validate inputs
        if radius <= 0:
            raise ValueError("Radius must be positive")
        if num_elements < 3:
            raise ValueError("Need at least 3 elements for circular discretization")
        if int(windings) != windings:
            raise ValueError("Windings has to be an integer")
        if windings < 1:
            raise ValueError("Coil needs to have at least one winding")
            
        # Store geometry parameters
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.current = float(current)
        self.windings = int(windings)
        self.num_elements_param = int(num_elements)
        self.normal_vector = np.asarray(normal_vector, dtype=float)
        
        # Normalize normal vector
        norm = np.linalg.norm(self.normal_vector)
        if norm < 1e-10:
            raise ValueError("Normal vector cannot be zero")
        self.normal_vector = self.normal_vector / norm
        
        # Compute local coordinate system for the coil plane
        self._compute_local_coordinates()
        
        logger.debug(f"Created circular coil: center={self.center}, radius={self.radius}, "
                    f"current={self.current}, windings={self.windings}, elements={self.num_elements_param}")
        
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
        
    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate line element geometry for the circular coil.
        
        Args:
            time: Simulation time (not used for static coils)
            
        Returns:
            Tuple of (A, B, I) arrays where:
            - A: Start points, shape (num_elements, 3)
            - B: End points, shape (num_elements, 3)
            - I: Current values, shape (num_elements,)
        """
        num_elements = self.num_elements_param
        
        # Angular step size
        angular_step = 2 * np.pi / num_elements
        
        # Initialize arrays
        A = np.zeros((num_elements, 3), dtype=float)
        B = np.zeros((num_elements, 3), dtype=float)
        I = np.full(num_elements, self.current*self.windings, dtype=float)
        
        # Generate circle points in local coordinate system
        for i in range(num_elements):
            # Current angle and next angle
            angle1 = i * angular_step
            angle2 = (i + 1) * angular_step
            
            # Points on circle in local coordinates
            p1_local = self.radius * (np.cos(angle1) * self.u_vector + np.sin(angle1) * self.v_vector)
            p2_local = self.radius * (np.cos(angle2) * self.u_vector + np.sin(angle2) * self.v_vector)
            
            # Transform to global coordinates
            A[i] = self.center + p1_local
            B[i] = self.center + p2_local
            
        logger.debug(f"Generated {num_elements} circular coil elements")
        
        return A, B, I
        
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get spatial bounds of the circular coil.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        # Coil extends radius distance in all directions within its plane
        # We need to consider all possible orientations
        
        # Extreme points along each axis
        extremes = []
        
        # Sample points around the circle to find extremes
        angles = np.linspace(0, 2*np.pi, 100)
        for angle in angles:
            point_local = self.radius * (np.cos(angle) * self.u_vector + np.sin(angle) * self.v_vector)
            point_global = self.center + point_local
            extremes.append(point_global)
            
        extremes = np.array(extremes)
        
        return {
            'x': (extremes[:, 0].min(), extremes[:, 0].max()),
            'y': (extremes[:, 1].min(), extremes[:, 1].max()),
            'z': (extremes[:, 2].min(), extremes[:, 2].max())
        }
        
    def get_circumference(self) -> float:
        """Get the circumference of the coil."""
        return 2 * np.pi * self.radius
        
    def get_area(self) -> float:
        """Get the area enclosed by the coil."""
        return np.pi * self.radius**2
        
    def get_magnetic_dipole_moment(self) -> np.ndarray:
        """
        Get the magnetic dipole moment vector.
        
        Returns:
            Dipole moment vector pointing along the normal direction
        """
        # m = I * A * n_hat
        area = self.get_area()
        return self.current * area * self.normal_vector
        
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
        
    def set_radius(self, radius: float):
        """
        Set the radius of the coil.
        
        Args:
            radius: New radius value
        """
        if radius <= 0:
            raise ValueError("Radius must be positive")
            
        self.radius = float(radius)
        self.parameters['radius'] = self.radius
        
        # Clear cached geometry to force regeneration
        self._geometry_A = None
        self._geometry_B = None
        self._geometry_I = None
        
        logger.debug(f"Updated coil radius to {self.radius}")
        
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
        """String representation of circular coil."""
        return (f"CircularCoil('{self.name}', center={self.center}, "
                f"radius={self.radius}, current={self.current}A, "
                f"windings={self.windings}, "
                f"{self.num_elements_param} elements)")
