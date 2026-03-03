"""
Base geometry classes for MAGA library.

This module provides the foundational classes for all geometry generators.
The base classes define the interface for generating line element representations
of current-carrying structures for Biot-Savart calculations.

Key concepts:
- Line elements: Short current segments with start/end points and current
- Discretization: Converting continuous current paths into discrete elements
- Time dependence: Support for time-varying currents and positions
- CPU/GPU generation: Option for host or device-side geometry calculation
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class GeometryParameters:
    """
    Container for geometry parameters and metadata.
    
    Stores configuration parameters, physical properties, and generation
    settings for a geometry. Provides validation and serialization.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize geometry parameters.
        
        Args:
            **kwargs: Arbitrary parameter name-value pairs
        """
        self._params = {}
        self._metadata = {
            'creation_time': None,
            'generator_type': None,
            'num_elements': 0
        }
        
        # Store all parameters
        for key, value in kwargs.items():
            self._params[key] = value
            
    def get(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name."""
        return self._params.get(name, default)
        
    def set(self, name: str, value: Any):
        """Set parameter value."""
        self._params[name] = value
        
    def update(self, **kwargs):
        """Update multiple parameters."""
        self._params.update(kwargs)
        
    def get_metadata(self, name: str, default: Any = None) -> Any:
        """Get metadata value by name."""
        return self._metadata.get(name, default)
        
    def set_metadata(self, name: str, value: Any):
        """Set metadata value."""
        self._metadata[name] = value
        
    @property
    def parameters(self) -> Dict[str, Any]:
        """Dictionary of all parameters."""
        return self._params.copy()
        
    @property
    def metadata(self) -> Dict[str, Any]:
        """Dictionary of all metadata."""
        return self._metadata.copy()
        
    def __getitem__(self, key: str) -> Any:
        """Dictionary-style access to parameters."""
        return self._params[key]
        
    def __setitem__(self, key: str, value: Any):
        """Dictionary-style setting of parameters."""
        self._params[key] = value
        
    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self._params
        
    def __str__(self) -> str:
        """String representation of parameters."""
        param_strs = [f"{k}={v}" for k, v in self._params.items()]
        return f"GeometryParameters({', '.join(param_strs)})"


class BaseGeometry(ABC):
    """
    Abstract base class for all geometry generators.
    
    Defines the interface for generating current-carrying geometries
    as collections of line elements suitable for Biot-Savart calculations.
    
    All geometries produce three arrays:
    - A: Start points of line elements, shape (N, 3)
    - B: End points of line elements, shape (N, 3)  
    - I: Current values for each element, shape (N,)
    """
    
    def __init__(self, name: str = "geometry", parameters: Optional[GeometryParameters] = None):
        """
        Initialize base geometry.
        
        Args:
            name: Human-readable name for this geometry
            parameters: Geometry parameters (created if None)
        """
        self.name = name
        self.parameters = parameters or GeometryParameters()
        
        # Generated geometry data (cached after first generation)
        self._geometry_A = None
        self._geometry_B = None 
        self._geometry_I = None
        self._num_elements = 0
        
        # Set metadata
        self.parameters.set_metadata('generator_type', self.__class__.__name__)
        
        logger.debug(f"Created {self.__class__.__name__} geometry: '{name}'")
        
    @abstractmethod
    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate line element geometry for given time.
        
        Args:
            time: Simulation time (used for time-dependent geometries)
            
        Returns:
            Tuple of (A, B, I) arrays where:
            - A: Start points, shape (N, 3)
            - B: End points, shape (N, 3)
            - I: Current values, shape (N,)
            
        Must be implemented by subclasses.
        """
        pass
        
    @abstractmethod 
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get spatial bounds of the geometry.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
            
        Must be implemented by subclasses.
        """
        pass
        
    def get_geometry(self, time: float = 0.0, force_regenerate: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get cached geometry or generate if needed.
        
        Args:
            time: Simulation time
            force_regenerate: Force regeneration even if cached
            
        Returns:
            Tuple of (A, B, I) geometry arrays
        """
        # Check if generation needed
        if (self._geometry_A is None or 
            self._geometry_B is None or 
            self._geometry_I is None or
            force_regenerate or
            self.is_time_dependent()):
            
            # Generate geometry
            A, B, I = self.generate_geometry(time)
            
            # Validate and cache
            self._validate_geometry_arrays(A, B, I)
            
            if not self.is_time_dependent():
                # Cache for time-independent geometries
                self._geometry_A = A.copy()
                self._geometry_B = B.copy()
                self._geometry_I = I.copy()
                self._num_elements = len(I)
                self.parameters.set_metadata('num_elements', self._num_elements)
                
            logger.debug(f"Generated geometry: {len(I)} elements")
            return A, B, I
        else:
            # Return cached geometry
            return self._geometry_A.copy(), self._geometry_B.copy(), self._geometry_I.copy()
            
    def _validate_geometry_arrays(self, A: np.ndarray, B: np.ndarray, I: np.ndarray):
        """Validate geometry arrays for consistency and correctness."""
        # Convert to numpy arrays
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        I = np.asarray(I, dtype=np.float64)
        
        # Check shapes
        if A.ndim != 2 or A.shape[1] != 3:
            raise ValueError("A array must have shape (N, 3)")
        if B.ndim != 2 or B.shape[1] != 3:
            raise ValueError("B array must have shape (N, 3)")
        if I.ndim != 1:
            raise ValueError("I array must have shape (N,)")
            
        # Check consistent sizes
        if len(A) != len(B) or len(A) != len(I):
            raise ValueError("A, B, I arrays must have consistent sizes")
            
        # Check for invalid values
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)):
            raise ValueError("Coordinate arrays contain invalid values")
        if not np.all(np.isfinite(I)):
            raise ValueError("Current array contains invalid values")
            
        # Check for zero-length elements (may indicate discretization issues)
        element_lengths = np.linalg.norm(B - A, axis=1)
        if np.any(element_lengths == 0):
            logger.warning("Some line elements have zero length")
            
    def is_time_dependent(self) -> bool:
        """
        Check if this geometry varies with time.
        
        Returns:
            True if geometry changes with simulation time
            
        Default implementation returns False. Override for time-dependent geometries.
        """
        return False
        
    @property
    def num_elements(self) -> int:
        """Number of line elements in this geometry."""
        if self._num_elements == 0 and self._geometry_I is not None:
            self._num_elements = len(self._geometry_I)
        return self._num_elements
        
    def get_total_current(self) -> float:
        """
        Get total current in the geometry.
        
        Returns:
            Sum of all element currents
        """
        if self._geometry_I is not None:
            return np.sum(self._geometry_I)
        else:
            # Need to generate geometry to calculate
            _, _, I = self.get_geometry()
            return np.sum(I)
            
    def get_center_of_mass(self) -> np.ndarray:
        """
        Get center of mass of the current distribution.
        
        Returns:
            Center of mass coordinates as [x, y, z]
        """
        A, B, I = self.get_geometry()
        
        # Element centers weighted by current
        centers = 0.5 * (A + B)
        total_current = np.sum(np.abs(I))
        
        if total_current == 0:
            # No current, return geometric center
            return np.mean(centers, axis=0)
        else:
            # Current-weighted center
            return np.average(centers, axis=0, weights=np.abs(I))
            
    def translate(self, offset: Union[np.ndarray, Tuple[float, float, float]]):
        """
        Translate geometry by offset vector.
        
        Args:
            offset: Translation vector [dx, dy, dz]
        """
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,):
            raise ValueError("Offset must be 3-element vector")
            
        # Apply translation to cached geometry if present
        if self._geometry_A is not None:
            self._geometry_A += offset
            self._geometry_B += offset
            
        # Update parameters if they contain position information
        if 'center' in self.parameters:
            current_center = np.asarray(self.parameters['center'])
            self.parameters['center'] = current_center + offset
            
        logger.debug(f"Translated geometry by {offset}")
        
    def rotate(self, rotation_matrix: np.ndarray, center: Optional[np.ndarray] = None):
        """
        Rotate geometry about center point.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            center: Rotation center (geometry center if None)
        """
        rotation_matrix = np.asarray(rotation_matrix, dtype=float)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
            
        if center is None:
            center = self.get_center_of_mass()
        else:
            center = np.asarray(center, dtype=float)
            
        # Apply rotation to cached geometry if present
        if self._geometry_A is not None:
            # Translate to origin, rotate, translate back
            A_centered = self._geometry_A - center
            B_centered = self._geometry_B - center
            
            A_rotated = A_centered @ rotation_matrix.T
            B_rotated = B_centered @ rotation_matrix.T
            
            self._geometry_A = A_rotated + center
            self._geometry_B = B_rotated + center
            
        logger.debug("Applied rotation to geometry")
        
    def scale(self, scale_factor: Union[float, np.ndarray], preserve_current: bool = True):
        """
        Scale geometry by factor.
        
        Args:
            scale_factor: Uniform scale factor or [sx, sy, sz] vector
            preserve_current: If True, scale current to preserve total current
        """
        if np.isscalar(scale_factor):
            scale_vector = np.full(3, scale_factor, dtype=float)
        else:
            scale_vector = np.asarray(scale_factor, dtype=float)
            if scale_vector.shape != (3,):
                raise ValueError("Scale factor must be scalar or 3-element vector")
                
        # Apply scaling to cached geometry if present
        if self._geometry_A is not None:
            center = self.get_center_of_mass()
            
            # Scale relative to center
            A_centered = self._geometry_A - center
            B_centered = self._geometry_B - center
            
            A_scaled = A_centered * scale_vector
            B_scaled = B_centered * scale_vector
            
            self._geometry_A = A_scaled + center * scale_vector
            self._geometry_B = B_scaled + center * scale_vector
            
            # Scale current to preserve total current if requested
            if preserve_current:
                volume_scale = np.prod(scale_vector)
                self._geometry_I *= (1.0 / volume_scale)
                
        logger.debug(f"Scaled geometry by {scale_vector}")
        
    def __str__(self) -> str:
        """String representation of geometry."""
        return f"{self.__class__.__name__}('{self.name}', {self.num_elements} elements)"
        
    def __repr__(self) -> str:
        """Detailed representation of geometry."""
        return f"{self.__class__.__name__}(name='{self.name}', parameters={self.parameters})"