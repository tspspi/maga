"""
MAGA: Magnetic Analysis with GPU Acceleration

A Python library for GPU-accelerated magnetic field calculations using the Biot-Savart law.
Provides high-performance computation of magnetic fields from arbitrary current geometries
with support for various coil configurations and time-dependent simulations.

Key Features:
- GPU-accelerated Biot-Savart calculations using OpenCL
- Multiple geometry types: circular coils, rectangular coils, coil pairs, electron beams
- Flexible grid configurations for field mapping
- Time-dependent simulations for dynamic systems
- Automatic device selection with CPU fallback
- Memory-efficient batching for large calculations

Basic Usage:
    >>> from maga import CircularCoil, RectangularGrid, MagneticFieldCalculator
    >>> 
    >>> # Create a circular coil
    >>> coil = CircularCoil(center=(0,0,0), radius=1.0, current=10.0)
    >>> 
    >>> # Define calculation grid
    >>> grid = RectangularGrid(x_range=(-2,2), y_range=(-2,2), z_range=(-1,1),
    ...                       nx=21, ny=21, nz=11)
    >>> 
    >>> # Calculate magnetic field
    >>> calculator = MagneticFieldCalculator()
    >>> A, B, I = coil.get_geometry()
    >>> result = calculator.calculate_magnetic_field(A, B, I, grid)
    >>> 
    >>> print(f"Max field: {result.max_field:.2e} T")

Dependencies:
- numpy: Numerical computing
- pyopencl: OpenCL Python bindings for GPU acceleration
- Optional: matplotlib for visualization

Author: MAGA Development Team
License: Open Source
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "MAGA Development Team"
__license__ = "Open Source"

# Import core computational components
from .core import (
    KernelRegistry,
    DeviceManager,
    DeviceInfo,
    GridConfiguration,
    RectangularGrid,
    CylindricalGrid,
    PlaneGrid,
    CustomGrid,
    MagneticFieldCalculator,
    CalculationResult
)

# Import geometry components
from .geometry import (
    BaseGeometry,
    GeometryParameters,
    CircularCoil,
    RectangularCoil,
    OscillatingBeam,
    OscillatingBeam2D,
    ChoppedBeam,
    HelmholtzCoils,
    AntiHelmholtzCoils
)

# Main public API - commonly used classes
__all__ = [
    # Core calculation components
    'MagneticFieldCalculator',
    'CalculationResult',
    
    # Device and kernel management
    'DeviceManager',
    'DeviceInfo',
    'KernelRegistry',
    
    # Grid configurations
    'GridConfiguration',
    'RectangularGrid',
    'CylindricalGrid',
    'PlaneGrid',
    'CustomGrid',
    
    # Geometry base classes
    'BaseGeometry',
    'GeometryParameters',
    
    # Basic geometry types
    'CircularCoil',
    'RectangularCoil',
    'OscillatingBeam',
    'OscillatingBeam2D',
    'ChoppedBeam',
    
    # Coil pair configurations
    'HelmholtzCoils',
    'AntiHelmholtzCoils',
    
    # Version information
    '__version__',
    '__author__',
    '__license__'
]

# Convenience functions for common use cases
def quick_coil_calculation(radius=1.0, current=1.0, grid_size=21, grid_range=2.0):
    """
    Quick magnetic field calculation for a circular coil.
    
    Args:
        radius: Coil radius in meters
        current: Current in Amperes
        grid_size: Number of grid points per dimension
        grid_range: Grid extends from -grid_range to +grid_range
        
    Returns:
        CalculationResult object with magnetic field data
    """
    coil = CircularCoil(radius=radius, current=current)
    grid = RectangularGrid(
        x_range=(-grid_range, grid_range),
        y_range=(-grid_range, grid_range),
        z_range=(-grid_range/2, grid_range/2),
        nx=grid_size, ny=grid_size, nz=grid_size//2
    )
    calculator = MagneticFieldCalculator()
    A, B, I = coil.get_geometry()
    return calculator.calculate_magnetic_field(A, B, I, grid)

def list_opencl_devices():
    """
    List all available OpenCL devices.
    
    Returns:
        List of device information strings
    """
    try:
        manager = DeviceManager()
        devices = manager.list_available_devices()
        return [str(device) for device in devices]
    except Exception as e:
        return [f"Error listing devices: {e}"]

# Package metadata
def get_version_info():
    """Get detailed version information."""
    try:
        import pyopencl as cl
        opencl_version = cl.get_platforms()[0].version if cl.get_platforms() else "Not available"
    except:
        opencl_version = "Not available"
        
    try:
        import numpy as np
        numpy_version = np.__version__
    except:
        numpy_version = "Not available"
        
    return {
        'maga_version': __version__,
        'numpy_version': numpy_version,
        'opencl_version': opencl_version,
        'available_devices': len(list_opencl_devices())
    }

# Add convenience imports to global namespace
__all__.extend(['quick_coil_calculation', 'list_opencl_devices', 'get_version_info'])