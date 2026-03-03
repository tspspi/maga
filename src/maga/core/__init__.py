"""
Core computation modules for MAGA library.

This package contains the fundamental computational components:
- Kernel management and OpenCL kernel registry
- Device management and OpenCL context handling  
- Grid configuration for calculation domains
- Magnetic field calculator (main computation engine)

The core modules provide the low-level functionality needed for
GPU-accelerated magnetic field calculations using the Biot-Savart law.
"""

from .kernels import KernelRegistry
from .device_manager import DeviceManager, DeviceInfo
from .grid import GridConfiguration, RectangularGrid, CylindricalGrid, PlaneGrid, CustomGrid
from .field_calculator import MagneticFieldCalculator, CalculationResult

__all__ = [
    'KernelRegistry',
    'DeviceManager',
    'DeviceInfo',
    'GridConfiguration',
    'RectangularGrid',
    'CylindricalGrid', 
    'PlaneGrid',
    'CustomGrid',
    'MagneticFieldCalculator',
    'CalculationResult'
]