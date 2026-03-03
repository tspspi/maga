"""
Coil pair geometries for MAGA library.

This module implements coil pair configurations like Helmholtz and Anti-Helmholtz
coils. These are commonly used configurations that combine two circular coils
to create uniform fields or field gradients.

Key features:
- Helmholtz coils for uniform magnetic fields
- Anti-Helmholtz coils for magnetic field gradients
- Automatic optimal spacing calculations
- Combined geometry generation
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

from .base import BaseGeometry, GeometryParameters
from .circular_coil import CircularCoil

logger = logging.getLogger(__name__)


class HelmholtzCoils(BaseGeometry):
    """
    Helmholtz coil pair geometry generator.
    
    Creates a pair of circular coils separated by their radius distance
    to produce a uniform magnetic field in the region between them.
    Both coils carry current in the same direction.
    """
    
    def __init__(self,
                 center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 radius: float = 1.0,
                 current: float = 1.0,
                 separation: Optional[float] = None,
                 windings: int = 1,
                 num_elements_per_coil: int = 100,
                 axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 name: str = "helmholtz_coils"):
        """
        Initialize Helmholtz coil pair.
        
        Args:
            center: Center position between the two coils
            radius: Radius of both coils
            current: Current through both coils (same direction)
            separation: Distance between coils (default: radius for optimal uniformity)
            num_elements_per_coil: Number of elements per individual coil
            axis: Axis along which coils are separated (coil normal direction)
            name: Name for this geometry
        """
        # Set default separation for Helmholtz condition
        if separation is None:
            separation = radius
            
        # Create parameter container
        parameters = GeometryParameters(
            center=center,
            radius=radius,
            current=current,
            separation=separation,
            windings=windings,
            num_elements_per_coil=num_elements_per_coil,
            axis=axis
        )
        
        super().__init__(name, parameters)
        
        # Validate inputs
        if radius <= 0:
            raise ValueError("Radius must be positive")
        if separation <= 0:
            raise ValueError("Separation must be positive")
        if num_elements_per_coil < 3:
            raise ValueError("Need at least 3 elements per coil")
        if int(windings) != windings:
            raise ValueError("Winding number has to be an integer")
        if windings < 1:
            raise ValueError("At least one winding has to be present per coil")
            
        # Store parameters
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.current = float(current)
        self.separation = float(separation)
        self.num_elements_per_coil = int(num_elements_per_coil)
        self.axis = np.asarray(axis, dtype=float)
        self.windings = int(windings)
        
        # Normalize axis
        norm = np.linalg.norm(self.axis)
        if norm < 1e-10:
            raise ValueError("Axis vector cannot be zero")
        self.axis = self.axis / norm
        
        # Create individual coils
        self._create_coils()
        
        logger.debug(f"Created Helmholtz coils: radius={self.radius}, "
                    f"separation={self.separation}, current={self.current}A")
        
    def _create_coils(self):
        """Create the two individual circular coils."""
        # Coil positions
        offset = self.separation / 2.0
        coil1_center = self.center - offset * self.axis
        coil2_center = self.center + offset * self.axis
        
        # Create coils with same current direction
        self.coil1 = CircularCoil(
            center=coil1_center,
            radius=self.radius,
            current=self.current,
            num_elements=self.num_elements_per_coil,
            normal_vector=self.axis,
            windings=self.windings,
            name=f"{self.name}_coil1"
        )
        
        self.coil2 = CircularCoil(
            center=coil2_center,
            radius=self.radius,
            current=self.current,  # Same direction
            num_elements=self.num_elements_per_coil,
            normal_vector=self.axis,
            windings=self.windings,
            name=f"{self.name}_coil2"
        )
        
    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate combined geometry for both coils.
        
        Args:
            time: Simulation time (not used for static coils)
            
        Returns:
            Tuple of (A, B, I) arrays with combined geometry
        """
        # Get geometry from both coils
        A1, B1, I1 = self.coil1.get_geometry(time)
        A2, B2, I2 = self.coil2.get_geometry(time)
        
        # Combine geometries
        A = np.vstack([A1, A2])
        B = np.vstack([B1, B2])
        I = np.hstack([I1, I2])
        
        logger.debug(f"Generated {len(I)} elements for Helmholtz coils")
        
        return A, B, I
        
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get combined spatial bounds of both coils."""
        bounds1 = self.coil1.get_bounds()
        bounds2 = self.coil2.get_bounds()
        
        return {
            'x': (min(bounds1['x'][0], bounds2['x'][0]), 
                  max(bounds1['x'][1], bounds2['x'][1])),
            'y': (min(bounds1['y'][0], bounds2['y'][0]), 
                  max(bounds1['y'][1], bounds2['y'][1])),
            'z': (min(bounds1['z'][0], bounds2['z'][0]), 
                  max(bounds1['z'][1], bounds2['z'][1]))
        }
        
    def get_field_uniformity_region(self) -> Dict[str, float]:
        """
        Get the region of good field uniformity.
        
        Returns:
            Dictionary with uniformity region parameters
        """
        # For Helmholtz coils, good uniformity is typically within
        # about 0.2 * radius from center along each axis
        uniformity_radius = 0.2 * self.radius
        
        return {
            'center_x': self.center[0],
            'center_y': self.center[1], 
            'center_z': self.center[2],
            'radius_x': uniformity_radius,
            'radius_y': uniformity_radius,
            'radius_z': uniformity_radius * 0.5  # Smaller along coil axis
        }
        
    def get_optimal_separation(self) -> float:
        """Get the theoretical optimal separation for maximum uniformity."""
        return self.radius  # R = a for Helmholtz condition
        
    def set_current(self, current: float):
        """Set current for both coils."""
        self.current = float(current)
        self.coil1.set_current(current)
        self.coil2.set_current(current)
        self.parameters['current'] = self.current
        
    def set_separation(self, separation: float):
        """Set separation between coils."""
        if separation <= 0:
            raise ValueError("Separation must be positive")
            
        self.separation = float(separation)
        self.parameters['separation'] = self.separation
        
        # Recreate coils with new separation
        self._create_coils()
        
    def __str__(self) -> str:
        return (f"HelmholtzCoils('{self.name}', radius={self.radius}, "
                f"separation={self.separation}, current={self.current}A, windings={self.windings})")


class AntiHelmholtzCoils(BaseGeometry):
    """
    Anti-Helmholtz coil pair geometry generator.
    
    Creates a pair of circular coils with opposite current directions
    to produce a linear magnetic field gradient. Commonly used in
    atom traps and magnetic focusing applications.
    """
    
    def __init__(self,
                 center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 radius: float = 1.0,
                 current: float = 1.0,
                 separation: Optional[float] = None,
                 num_elements_per_coil: int = 100,
                 windings: int = 1,
                 axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 name: str = "anti_helmholtz_coils"):
        """
        Initialize Anti-Helmholtz coil pair.
        
        Args:
            center: Center position between the two coils (field zero point)
            radius: Radius of both coils
            current: Current magnitude (opposite directions in the two coils)
            separation: Distance between coils (default: radius)
            num_elements_per_coil: Number of elements per individual coil
            axis: Axis along which coils are separated and gradient is created
            name: Name for this geometry
        """
        # Set default separation
        if separation is None:
            separation = radius
            
        # Create parameter container
        parameters = GeometryParameters(
            center=center,
            radius=radius,
            current=current,
            separation=separation,
            windings=windings,
            num_elements_per_coil=num_elements_per_coil,
            axis=axis
        )
        
        super().__init__(name, parameters)
        
        # Validate inputs
        if radius <= 0:
            raise ValueError("Radius must be positive")
        if separation <= 0:
            raise ValueError("Separation must be positive")
        if num_elements_per_coil < 3:
            raise ValueError("Need at least 3 elements per coil")
        if int(windings) != windings:
            raise ValueError("Winding number has to be a integer")
        if windings < 1:
            raise ValueError("Coils have to have at least one winding")
            
        # Store parameters
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.current = float(current)
        self.separation = float(separation)
        self.num_elements_per_coil = int(num_elements_per_coil)
        self.axis = np.asarray(axis, dtype=float)
        self.windings = int(windings)
        
        # Normalize axis
        norm = np.linalg.norm(self.axis)
        if norm < 1e-10:
            raise ValueError("Axis vector cannot be zero")
        self.axis = self.axis / norm
        
        # Create individual coils
        self._create_coils()
        
        logger.debug(f"Created Anti-Helmholtz coils: radius={self.radius}, "
                    f"separation={self.separation}, current=±{self.current}A")
        
    def _create_coils(self):
        """Create the two individual circular coils with opposite currents."""
        # Coil positions
        offset = self.separation / 2.0
        coil1_center = self.center - offset * self.axis
        coil2_center = self.center + offset * self.axis
        
        # Create coils with opposite current directions
        self.coil1 = CircularCoil(
            center=coil1_center,
            radius=self.radius,
            current=self.current,   # Positive current
            num_elements=self.num_elements_per_coil,
            windings=windings,
            normal_vector=self.axis,
            name=f"{self.name}_coil1"
        )
        
        self.coil2 = CircularCoil(
            center=coil2_center,
            radius=self.radius,
            current=-self.current,  # Opposite direction
            num_elements=self.num_elements_per_coil,
            windings=windings,
            normal_vector=self.axis,
            name=f"{self.name}_coil2"
        )
        
    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate combined geometry for both coils.
        
        Args:
            time: Simulation time (not used for static coils)
            
        Returns:
            Tuple of (A, B, I) arrays with combined geometry
        """
        # Get geometry from both coils
        A1, B1, I1 = self.coil1.get_geometry(time)
        A2, B2, I2 = self.coil2.get_geometry(time)
        
        # Combine geometries
        A = np.vstack([A1, A2])
        B = np.vstack([B1, B2])
        I = np.hstack([I1, I2])
        
        logger.debug(f"Generated {len(I)} elements for Anti-Helmholtz coils")
        
        return A, B, I
        
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get combined spatial bounds of both coils."""
        bounds1 = self.coil1.get_bounds()
        bounds2 = self.coil2.get_bounds()
        
        return {
            'x': (min(bounds1['x'][0], bounds2['x'][0]), 
                  max(bounds1['x'][1], bounds2['x'][1])),
            'y': (min(bounds1['y'][0], bounds2['y'][0]), 
                  max(bounds1['y'][1], bounds2['y'][1])),
            'z': (min(bounds1['z'][0], bounds2['z'][0]), 
                  max(bounds1['z'][1], bounds2['z'][1]))
        }
        
    def get_gradient_strength(self) -> float:
        """
        Estimate the magnetic field gradient strength at the center.
        
        Returns:
            Field gradient in T/m (approximate)
        """
        # Approximate formula for Anti-Helmholtz gradient
        # dB/dz ≈ μ₀ * I * R² / (R² + (d/2)²)^(3/2)
        mu_0 = 4e-7 * np.pi
        R = self.radius
        d = self.separation
        
        gradient = mu_0 * self.current * R**2 / (R**2 + (d/2)**2)**(3/2)
        return gradient
        
    def get_zero_field_position(self) -> np.ndarray:
        """
        Get the position where the magnetic field is zero.
        
        Returns:
            Position of magnetic field zero (typically the center)
        """
        return self.center.copy()
        
    def get_trap_frequencies(self, mass: float, magnetic_moment: float) -> Dict[str, float]:
        """
        Calculate trap frequencies for magnetic trapping.
        
        Args:
            mass: Particle mass in kg
            magnetic_moment: Magnetic moment in J/T
            
        Returns:
            Dictionary with trap frequencies in Hz
        """
        gradient = self.get_gradient_strength()
        
        # Trap frequency: ω = sqrt(μ * |dB/dx| / m)
        # For Anti-Helmholtz: radial trapping, axial anti-trapping
        omega_radial = np.sqrt(magnetic_moment * gradient / mass) / (2 * np.pi)
        omega_axial = omega_radial  # Same magnitude, opposite sign (anti-trapping)
        
        return {
            'radial_frequency': omega_radial,
            'axial_frequency': omega_axial,  # Note: anti-trapping direction
            'gradient': gradient
        }
        
    def set_current(self, current: float):
        """Set current magnitude (maintains opposite directions)."""
        self.current = float(current)
        self.coil1.set_current(current)
        self.coil2.set_current(-current)  # Opposite direction
        self.parameters['current'] = self.current
        
    def set_separation(self, separation: float):
        """Set separation between coils."""
        if separation <= 0:
            raise ValueError("Separation must be positive")
            
        self.separation = float(separation)
        self.parameters['separation'] = self.separation
        
        # Recreate coils with new separation
        self._create_coils()
        
    def __str__(self) -> str:
        return (f"AntiHelmholtzCoils('{self.name}', radius={self.radius}, "
                f"separation={self.separation}, current=±{self.current}A, windings={self.windings})")
