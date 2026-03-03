"""
Grid configuration and generation for MAGA magnetic field calculations.

This module provides classes for defining and generating 3D coordinate grids
where magnetic field calculations will be performed. Supports various grid
types with memory-efficient layouts for GPU calculations.

Key features:
- Rectangular 3D grids with customizable spacing
- Cylindrical coordinate grids for axisymmetric problems
- Memory-efficient flattened coordinate arrays
- Grid metadata and bounds checking
- Support for time-dependent calculations
"""

import numpy as np
from typing import Tuple, Union, Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GridConfiguration:
    """
    Base configuration class for 3D coordinate grids.
    
    Defines the spatial region and resolution for magnetic field calculations.
    Provides methods for grid generation and coordinate transformation.
    """
    
    def __init__(self, name: str = "grid"):
        """
        Initialize base grid configuration.
        
        Args:
            name: Human-readable name for this grid
        """
        self.name = name
        self._coordinates = None
        self._num_points = 0
        
    @property
    def num_points(self) -> int:
        """Number of grid points."""
        return self._num_points
        
    @property
    def coordinates(self) -> Optional[np.ndarray]:
        """
        Grid coordinates as (N, 3) array.
        
        Returns:
            Array of shape (num_points, 3) with [x, y, z] coordinates
        """
        return self._coordinates
        
    @property
    def memory_size_bytes(self) -> int:
        """Estimate memory size in bytes for this grid."""
        return self.num_points * 3 * 8  # 3 coordinates * 8 bytes per double
        
    def generate_coordinates(self) -> np.ndarray:
        """
        Generate coordinate array for this grid.
        
        Returns:
            Array of shape (num_points, 3) with [x, y, z] coordinates
            
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_coordinates")
        
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get coordinate bounds for this grid.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        if self.coordinates is None:
            self.generate_coordinates()
            
        coords = self.coordinates
        return {
            'x': (coords[:, 0].min(), coords[:, 0].max()),
            'y': (coords[:, 1].min(), coords[:, 1].max()),
            'z': (coords[:, 2].min(), coords[:, 2].max())
        }
        
    def __str__(self) -> str:
        """String representation of grid configuration."""
        return f"{self.__class__.__name__}('{self.name}', {self.num_points} points)"


class RectangularGrid(GridConfiguration):
    """
    Rectangular 3D grid with uniform spacing.
    
    Creates a regular grid of points in a rectangular region,
    commonly used for field mapping and visualization.
    """
    
    def __init__(self, 
                 x_range: Tuple[float, float], 
                 y_range: Tuple[float, float], 
                 z_range: Tuple[float, float],
                 nx: int, 
                 ny: int, 
                 nz: int,
                 name: str = "rectangular_grid"):
        """
        Initialize rectangular grid.
        
        Args:
            x_range: (x_min, x_max) coordinate range
            y_range: (y_min, y_max) coordinate range  
            z_range: (z_min, z_max) coordinate range
            nx: Number of points along x-axis
            ny: Number of points along y-axis
            nz: Number of points along z-axis
            name: Grid name
        """
        super().__init__(name)
        
        # Validate inputs
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("Grid dimensions must be positive integers")
            
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1] or z_range[0] >= z_range[1]:
            raise ValueError("Invalid coordinate ranges: min must be less than max")
        
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._num_points = nx * ny * nz
        
        logger.debug(f"Created rectangular grid: {nx}×{ny}×{nz} = {self.num_points} points")
        
    def generate_coordinates(self) -> np.ndarray:
        """
        Generate rectangular grid coordinates.
        
        Returns:
            Array of shape (nx*ny*nz, 3) with coordinates
        """
        # Create 1D coordinate arrays
        x = np.linspace(self.x_range[0], self.x_range[1], self.nx)
        y = np.linspace(self.y_range[0], self.y_range[1], self.ny) 
        z = np.linspace(self.z_range[0], self.z_range[1], self.nz)
        
        # Create 3D meshgrid
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten and combine into coordinate array
        coordinates = np.column_stack([
            X.ravel(),
            Y.ravel(), 
            Z.ravel()
        ])
        
        self._coordinates = coordinates
        logger.debug(f"Generated {coordinates.shape[0]} grid coordinates")
        
        return coordinates
        
    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Shape of the 3D grid as (nx, ny, nz)."""
        return (self.nx, self.ny, self.nz)
        
    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Grid spacing as (dx, dy, dz)."""
        dx = (self.x_range[1] - self.x_range[0]) / (self.nx - 1) if self.nx > 1 else 0
        dy = (self.y_range[1] - self.y_range[0]) / (self.ny - 1) if self.ny > 1 else 0
        dz = (self.z_range[1] - self.z_range[0]) / (self.nz - 1) if self.nz > 1 else 0
        return (dx, dy, dz)
        
    def get_slice_coordinates(self, axis: str, value: float) -> np.ndarray:
        """
        Get coordinates for a 2D slice through the grid.
        
        Args:
            axis: Slice axis ('x', 'y', or 'z')
            value: Coordinate value for the slice
            
        Returns:
            Array of coordinates in the slice plane
        """
        if self.coordinates is None:
            self.generate_coordinates()
            
        axis_idx = {'x': 0, 'y': 1, 'z': 2}[axis.lower()]
        coords = self.coordinates
        
        # Find points close to the slice value
        tolerance = min(self.spacing) / 2
        mask = np.abs(coords[:, axis_idx] - value) <= tolerance
        
        return coords[mask]


class CylindricalGrid(GridConfiguration):
    """
    Cylindrical coordinate grid for axisymmetric problems.
    
    Creates points in cylindrical coordinates (r, phi, z) and converts
    to Cartesian for field calculations. Useful for problems with
    rotational symmetry like circular coils.
    """
    
    def __init__(self,
                 r_range: Tuple[float, float],
                 phi_range: Tuple[float, float],
                 z_range: Tuple[float, float], 
                 nr: int,
                 nphi: int,
                 nz: int,
                 name: str = "cylindrical_grid"):
        """
        Initialize cylindrical grid.
        
        Args:
            r_range: (r_min, r_max) radial range
            phi_range: (phi_min, phi_max) azimuthal range in radians
            z_range: (z_min, z_max) axial range
            nr: Number of points in radial direction
            nphi: Number of points in azimuthal direction
            nz: Number of points in axial direction
            name: Grid name
        """
        super().__init__(name)
        
        # Validate inputs
        if nr < 1 or nphi < 1 or nz < 1:
            raise ValueError("Grid dimensions must be positive integers")
            
        if r_range[0] < 0:
            raise ValueError("Radial coordinates must be non-negative")
            
        if r_range[0] >= r_range[1] or z_range[0] >= z_range[1]:
            raise ValueError("Invalid coordinate ranges")
            
        self.r_range = r_range
        self.phi_range = phi_range
        self.z_range = z_range
        self.nr = nr
        self.nphi = nphi
        self.nz = nz
        self._num_points = nr * nphi * nz
        
        logger.debug(f"Created cylindrical grid: {nr}×{nphi}×{nz} = {self.num_points} points")
        
    def generate_coordinates(self) -> np.ndarray:
        """
        Generate cylindrical grid coordinates in Cartesian form.
        
        Returns:
            Array of shape (nr*nphi*nz, 3) with [x, y, z] coordinates
        """
        # Create 1D coordinate arrays in cylindrical coordinates
        r = np.linspace(self.r_range[0], self.r_range[1], self.nr)
        phi = np.linspace(self.phi_range[0], self.phi_range[1], self.nphi)
        z = np.linspace(self.z_range[0], self.z_range[1], self.nz)
        
        # Create 3D meshgrid in cylindrical coordinates
        R, PHI, Z = np.meshgrid(r, phi, z, indexing='ij')
        
        # Convert to Cartesian coordinates
        X = R * np.cos(PHI)
        Y = R * np.sin(PHI)
        Z_cart = Z  # Z coordinate unchanged
        
        # Flatten and combine
        coordinates = np.column_stack([
            X.ravel(),
            Y.ravel(),
            Z_cart.ravel()
        ])
        
        self._coordinates = coordinates
        logger.debug(f"Generated {coordinates.shape[0]} cylindrical grid coordinates")
        
        return coordinates
        
    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Shape of the cylindrical grid as (nr, nphi, nz)."""
        return (self.nr, self.nphi, self.nz)


class PlaneGrid(GridConfiguration):
    """
    2D plane grid embedded in 3D space.
    
    Creates a regular grid of points in a plane, useful for
    visualizing field patterns on specific surfaces.
    """
    
    def __init__(self,
                 origin: Tuple[float, float, float],
                 u_vector: Tuple[float, float, float],
                 v_vector: Tuple[float, float, float],
                 u_range: Tuple[float, float],
                 v_range: Tuple[float, float],
                 nu: int,
                 nv: int,
                 name: str = "plane_grid"):
        """
        Initialize plane grid.
        
        Args:
            origin: (x, y, z) origin point of the plane
            u_vector: Direction vector for u-axis
            v_vector: Direction vector for v-axis  
            u_range: (u_min, u_max) range along u-axis
            v_range: (v_min, v_max) range along v-axis
            nu: Number of points along u-axis
            nv: Number of points along v-axis
            name: Grid name
        """
        super().__init__(name)
        
        # Validate inputs
        if nu < 1 or nv < 1:
            raise ValueError("Grid dimensions must be positive integers")
            
        # Normalize direction vectors
        u_vec = np.array(u_vector, dtype=float)
        v_vec = np.array(v_vector, dtype=float)
        u_vec = u_vec / np.linalg.norm(u_vec)
        v_vec = v_vec / np.linalg.norm(v_vec)
        
        # Check vectors are not parallel
        cross_product = np.cross(u_vec, v_vec)
        if np.linalg.norm(cross_product) < 1e-10:
            raise ValueError("u_vector and v_vector must not be parallel")
        
        self.origin = np.array(origin, dtype=float)
        self.u_vector = u_vec
        self.v_vector = v_vec
        self.u_range = u_range
        self.v_range = v_range
        self.nu = nu
        self.nv = nv
        self._num_points = nu * nv
        
        logger.debug(f"Created plane grid: {nu}×{nv} = {self.num_points} points")
        
    def generate_coordinates(self) -> np.ndarray:
        """
        Generate plane grid coordinates.
        
        Returns:
            Array of shape (nu*nv, 3) with [x, y, z] coordinates
        """
        # Create 1D parameter arrays
        u = np.linspace(self.u_range[0], self.u_range[1], self.nu)
        v = np.linspace(self.v_range[0], self.v_range[1], self.nv)
        
        # Create 2D meshgrid
        U, V = np.meshgrid(u, v, indexing='ij')
        
        # Calculate 3D positions
        positions = []
        for u_val, v_val in zip(U.ravel(), V.ravel()):
            point = self.origin + u_val * self.u_vector + v_val * self.v_vector
            positions.append(point)
            
        coordinates = np.array(positions)
        self._coordinates = coordinates
        
        logger.debug(f"Generated {coordinates.shape[0]} plane grid coordinates")
        return coordinates
        
    @property
    def grid_shape(self) -> Tuple[int, int]:
        """Shape of the plane grid as (nu, nv)."""
        return (self.nu, self.nv)
        
    @property
    def normal_vector(self) -> np.ndarray:
        """Normal vector to the plane."""
        return np.cross(self.u_vector, self.v_vector)


class CustomGrid(GridConfiguration):
    """
    Custom grid from user-provided coordinates.
    
    Allows arbitrary point distributions for specialized calculations
    or imported coordinate sets.
    """
    
    def __init__(self, coordinates: np.ndarray, name: str = "custom_grid"):
        """
        Initialize custom grid from coordinate array.
        
        Args:
            coordinates: Array of shape (N, 3) with [x, y, z] coordinates
            name: Grid name
        """
        super().__init__(name)
        
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError("Coordinates must be array of shape (N, 3)")
            
        self._coordinates = coordinates.copy()
        self._num_points = coordinates.shape[0]
        
        logger.debug(f"Created custom grid with {self.num_points} points")
        
    def generate_coordinates(self) -> np.ndarray:
        """Return the stored coordinates."""
        return self._coordinates.copy()