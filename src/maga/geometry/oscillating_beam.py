"""
Oscillating electron beam geometry generator for MAGA library.

This module implements the OscillatingBeam class for generating time-dependent
electron beam geometries. Based on the oscillatingBeamGenerator from the original
Jupyter notebook, this simulates modulated electron beams as oscillating current paths.

Key features:
- Relativistic electron velocity calculation
- Time-dependent sinusoidal modulation
- Configurable beam parameters (voltage, current, frequency)
- Arbitrary propagation and modulation directions
- Integration with MAGA time-dependent calculations
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging

from .base import BaseGeometry, GeometryParameters

logger = logging.getLogger(__name__)

# Physical constants
ELECTRON_CHARGE = 1.60217663e-19  # Coulombs
ELECTRON_MASS = 9.1093837015e-31  # kg
SPEED_OF_LIGHT = 299792458.0      # m/s


class OscillatingBeam(BaseGeometry):
    """
    Time-dependent oscillating electron beam geometry generator.
    
    Models a modulated electron beam as an oscillating current path.
    The beam follows a sinusoidal trajectory with time-dependent amplitude,
    simulating the electromagnetic effects of modulated electron beams.
    
    Note: This is a simplified model treating the beam as a classical
    current distribution. Real electron beams have more complex near-field
    behavior that requires quantum mechanical treatment.
    """
    
    def __init__(self,
                 voltage: float,
                 current: float,
                 modulation_frequency: float,
                 modulation_amplitude: float,
                 start_position: Tuple[float, float, float],
                 propagation_direction: Tuple[float, float, float],
                 modulation_direction: Tuple[float, float, float],
                 length: float,
                 num_elements: int = 100,
                 name: str = "oscillating_beam"):
        """
        Initialize oscillating electron beam geometry.
        
        Args:
            voltage: Acceleration voltage in Volts
            current: Beam current in Amperes
            modulation_frequency: Modulation frequency in Hz
            modulation_amplitude: Modulation amplitude in meters
            start_position: Starting position of the beam (x, y, z)
            propagation_direction: Direction of beam propagation
            modulation_direction: Direction of beam modulation (perpendicular to propagation)
            length: Total length of the beam path in meters
            num_elements: Number of line elements for discretization
            name: Name for this geometry
        """
        # Create parameter container
        parameters = GeometryParameters(
            voltage=voltage,
            current=current,
            modulation_frequency=modulation_frequency,
            modulation_amplitude=modulation_amplitude,
            start_position=start_position,
            propagation_direction=propagation_direction,
            modulation_direction=modulation_direction,
            length=length,
            num_elements=num_elements
        )
        
        super().__init__(name, parameters)
        
        # Validate inputs
        if voltage <= 0:
            raise ValueError("Voltage must be positive")
        if modulation_frequency <= 0:
            raise ValueError("Modulation frequency must be positive")
        if length <= 0:
            raise ValueError("Beam length must be positive")
        if num_elements < 2:
            raise ValueError("Need at least 2 elements for beam discretization")
            
        # Store beam parameters
        self.voltage = float(voltage)
        self.current = float(current)
        self.modulation_frequency = float(modulation_frequency)
        self.modulation_amplitude = float(modulation_amplitude)
        self.length = float(length)
        self.num_elements_param = int(num_elements)
        
        # Position and direction vectors
        self.start_position = np.asarray(start_position, dtype=float)
        self.propagation_direction = np.asarray(propagation_direction, dtype=float)
        self.modulation_direction = np.asarray(modulation_direction, dtype=float)
        
        # Normalize direction vectors
        self._normalize_directions()
        
        # Calculate derived parameters
        self._calculate_beam_physics()
        
        logger.debug(f"Created oscillating beam: voltage={self.voltage}V, "
                    f"current={self.current}A, f_mod={self.modulation_frequency}Hz, "
                    f"length={self.length}m, velocity={self.velocity:.3e}m/s")
        
    def _normalize_directions(self):
        """Normalize and validate direction vectors."""
        # Normalize propagation direction
        prop_norm = np.linalg.norm(self.propagation_direction)
        if prop_norm < 1e-10:
            raise ValueError("Propagation direction cannot be zero")
        self.propagation_direction = self.propagation_direction / prop_norm
        
        # Normalize modulation direction
        mod_norm = np.linalg.norm(self.modulation_direction)
        if mod_norm < 1e-10:
            raise ValueError("Modulation direction cannot be zero")
        self.modulation_direction = self.modulation_direction / mod_norm

        # Check that directions are not parallel (within tolerance)
        dot_product = np.dot(self.propagation_direction, self.modulation_direction)
        if np.isclose(np.abs(dot_product), 1.0, atol=1e-8):
            raise ValueError("Propagation and modulation directions must not be parallel")

        # Make modulation direction orthogonal to propagation direction by removing its projection
        self.modulation_direction = self.modulation_direction - (
            dot_product * self.propagation_direction
        )

        mod_norm = np.linalg.norm(self.modulation_direction)
        if mod_norm < 1e-10:
            raise ValueError("Failed to orthogonalize modulation direction")
        self.modulation_direction = self.modulation_direction / mod_norm
        
        logger.debug(f"Normalized directions: prop={self.propagation_direction}, "
                    f"mod={self.modulation_direction}")
        
    def _calculate_beam_physics(self):
        """Calculate relativistic velocity and wave parameters."""
        # Calculate relativistic electron velocity
        # v = c * sqrt(1 - 1/(1 + eV/(mc^2))^2)
        gamma_factor = 1.0 + (self.voltage * ELECTRON_CHARGE) / (ELECTRON_MASS * SPEED_OF_LIGHT**2)
        self.velocity = SPEED_OF_LIGHT * np.sqrt(1.0 - 1.0/(gamma_factor**2))
        
        # Calculate modulation parameters
        self.omega = 2.0 * np.pi * self.modulation_frequency
        self.wavelength = self.velocity / self.modulation_frequency
        self.k = 2.0 * np.pi / self.wavelength
        
        # Calculate element spacing
        self.element_spacing = self.length / self.num_elements_param
        
        logger.debug(f"Beam physics: velocity={self.velocity:.3e}m/s, "
                    f"wavelength={self.wavelength:.3e}m, k={self.k:.3e}m^-1")
        
    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate line element geometry for the oscillating beam at given time.
        
        Args:
            time: Simulation time in seconds
            
        Returns:
            Tuple of (A, B, I) arrays where:
            - A: Start points, shape (num_elements, 3)
            - B: End points, shape (num_elements, 3)
            - I: Current values, shape (num_elements,)
        """
        num_elements = self.num_elements_param
        
        # Initialize arrays
        A = np.zeros((num_elements, 3), dtype=float)
        B = np.zeros((num_elements, 3), dtype=float)
        I = np.full(num_elements, self.current, dtype=float)
        
        # Generate beam segments
        for i in range(num_elements):
            # Position along beam
            z_start = i * self.element_spacing
            z_end = (i + 1) * self.element_spacing
            
            # Modulation displacement at start and end
            phase_start = self.omega * time + self.k * z_start
            phase_end = self.omega * time + self.k * z_end
            
            deflection_start = self.modulation_amplitude * np.sin(phase_start)
            deflection_end = self.modulation_amplitude * np.sin(phase_end)
            
            # Calculate 3D positions
            A[i] = (self.start_position + 
                   z_start * self.propagation_direction + 
                   deflection_start * self.modulation_direction)
                   
            B[i] = (self.start_position + 
                   z_end * self.propagation_direction + 
                   deflection_end * self.modulation_direction)
                   
        logger.debug(f"Generated {num_elements} oscillating beam elements at time {time}s")
        
        return A, B, I
        
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """
        Get spatial bounds of the oscillating beam.
        
        Returns:
            Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
        """
        # Start and end positions of undeflected beam
        start_pos = self.start_position
        end_pos = self.start_position + self.length * self.propagation_direction
        
        # Maximum deflection in modulation direction
        max_deflection = self.modulation_amplitude
        
        # Corner points considering maximum deflection
        corner_points = [
            start_pos + max_deflection * self.modulation_direction,
            start_pos - max_deflection * self.modulation_direction,
            end_pos + max_deflection * self.modulation_direction,
            end_pos - max_deflection * self.modulation_direction
        ]
        
        corners = np.array(corner_points)
        
        return {
            'x': (corners[:, 0].min(), corners[:, 0].max()),
            'y': (corners[:, 1].min(), corners[:, 1].max()),
            'z': (corners[:, 2].min(), corners[:, 2].max())
        }
        
    def is_time_dependent(self) -> bool:
        """
        Check if this geometry varies with time.
        
        Returns:
            True - oscillating beams are always time-dependent
        """
        return True
        
    def get_beam_velocity(self) -> float:
        """Get the relativistic beam velocity in m/s."""
        return self.velocity
        
    def get_modulation_wavelength(self) -> float:
        """Get the modulation wavelength in meters."""
        return self.wavelength
        
    def get_beta(self) -> float:
        """Get the relativistic beta factor (v/c)."""
        return self.velocity / SPEED_OF_LIGHT
        
    def get_gamma(self) -> float:
        """Get the relativistic gamma factor."""
        beta = self.get_beta()
        return 1.0 / np.sqrt(1.0 - beta**2)
        
    def get_kinetic_energy(self) -> float:
        """Get the kinetic energy of electrons in Joules."""
        return self.voltage * ELECTRON_CHARGE
        
    def get_kinetic_energy_eV(self) -> float:
        """Get the kinetic energy of electrons in eV."""
        return self.voltage
        
    def set_voltage(self, voltage: float):
        """
        Set the acceleration voltage and recalculate beam physics.
        
        Args:
            voltage: New acceleration voltage in Volts
        """
        if voltage <= 0:
            raise ValueError("Voltage must be positive")
            
        self.voltage = float(voltage)
        self.parameters['voltage'] = self.voltage
        
        # Recalculate derived parameters
        self._calculate_beam_physics()
        
        logger.debug(f"Updated beam voltage to {self.voltage}V, "
                    f"new velocity: {self.velocity:.3e}m/s")
        
    def set_modulation_parameters(self, frequency: float, amplitude: float):
        """
        Set modulation frequency and amplitude.
        
        Args:
            frequency: Modulation frequency in Hz
            amplitude: Modulation amplitude in meters
        """
        if frequency <= 0:
            raise ValueError("Modulation frequency must be positive")
            
        self.modulation_frequency = float(frequency)
        self.modulation_amplitude = float(amplitude)
        
        self.parameters['modulation_frequency'] = self.modulation_frequency
        self.parameters['modulation_amplitude'] = self.modulation_amplitude
        
        # Recalculate wave parameters
        self._calculate_beam_physics()
        
        logger.debug(f"Updated modulation: f={self.modulation_frequency}Hz, "
                    f"A={self.modulation_amplitude}m")
        
    def set_current(self, current: float):
        """
        Set the beam current.
        
        Args:
            current: New beam current in Amperes
        """
        self.current = float(current)
        self.parameters['current'] = self.current
        
        logger.debug(f"Updated beam current to {self.current}A")
        
    def get_instantaneous_frequency(self, position: float, time: float = 0.0) -> float:
        """
        Get the instantaneous modulation frequency at a given position and time.
        
        Args:
            position: Position along beam (0 to length)
            time: Time in seconds
            
        Returns:
            Instantaneous frequency in Hz
        """
        # For sinusoidal modulation, frequency is constant
        return self.modulation_frequency
        
    def get_phase(self, position: float, time: float = 0.0) -> float:
        """
        Get the modulation phase at a given position and time.
        
        Args:
            position: Position along beam (0 to length)
            time: Time in seconds
            
        Returns:
            Phase in radians
        """
        return self.omega * time + self.k * position
        
    def __str__(self) -> str:
        """String representation of oscillating beam."""
        return (f"OscillatingBeam('{self.name}', voltage={self.voltage}V, "
                f"current={self.current}A, f_mod={self.modulation_frequency}Hz, "
                f"length={self.length}m, {self.num_elements_param} elements)")
