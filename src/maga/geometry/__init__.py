"""
Geometry generation module for MAGA library.

This module provides classes for generating various types of current-carrying
geometries that can be used in magnetic field calculations. All geometries
produce line element representations suitable for the Biot-Savart law.

Available geometry types:
- CircularCoil: Single circular current loop
- RectangularCoil: Rectangular current loop  
- HelmholtzCoils: Pair of circular coils for uniform fields
- AntiHelmholtzCoils: Pair of circular coils for gradient fields
- OscillatingBeam: Time-dependent electron beam simulation
- OscillatingBeam2D: Dual-axis oscillating beam for polarized trajectories
- ChoppedBeam: Straight chopped electron beam
"""

from .base import BaseGeometry, GeometryParameters
from .circular_coil import CircularCoil
from .rectangular_coil import RectangularCoil
from .oscillating_beam import OscillatingBeam
from .oscillating_beam_2d import OscillatingBeam2D
from .chopped_beam import ChoppedBeam
from .coil_pairs import HelmholtzCoils, AntiHelmholtzCoils

__all__ = [
    'BaseGeometry',
    'GeometryParameters', 
    'CircularCoil',
    'RectangularCoil',
    'OscillatingBeam',
    'OscillatingBeam2D',
    'ChoppedBeam',
    'HelmholtzCoils',
    'AntiHelmholtzCoils'
]
