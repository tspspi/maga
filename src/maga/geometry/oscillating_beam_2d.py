"""
Two-axis oscillating electron beam geometry for MAGA library.

Extends the base oscillating beam model by adding a second modulation axis
orthogonal to both the propagation and primary modulation directions. The two
orthogonal modulations share the same frequency and wavenumber, with the second
axis supporting an independent amplitude and phase shift. This enables helical
and circular beam trajectories for polarization studies.
"""

import logging
from typing import Tuple, Dict, Optional

import numpy as np

from .oscillating_beam import OscillatingBeam

logger = logging.getLogger(__name__)


class OscillatingBeam2D(OscillatingBeam):
    """
    Oscillating electron beam with orthogonal dual-axis modulation.

    The primary modulation axis is inherited from OscillatingBeam. A secondary
    axis is constructed as the cross product of the propagation and modulation
    directions, ensuring a right-handed orthonormal basis for the beam motion.
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
                 modulation_amplitude_2ndaxis: float = 0.0,
                 phase_difference_2ndaxis: float = 0.0,
                 name: str = "oscillating_beam_2d"):
        super().__init__(
            voltage=voltage,
            current=current,
            modulation_frequency=modulation_frequency,
            modulation_amplitude=modulation_amplitude,
            start_position=start_position,
            propagation_direction=propagation_direction,
            modulation_direction=modulation_direction,
            length=length,
            num_elements=num_elements,
            name=name
        )

        self.modulation_amplitude_2ndaxis = float(modulation_amplitude_2ndaxis)
        self.phase_difference_2ndaxis = float(phase_difference_2ndaxis)
        self.parameters['modulation_amplitude_2ndaxis'] = self.modulation_amplitude_2ndaxis
        self.parameters['phase_difference_2ndaxis'] = self.phase_difference_2ndaxis

        self.secondary_modulation_direction = self._compute_secondary_direction()
        logger.debug("Computed secondary modulation direction: %s", self.secondary_modulation_direction)

    def _compute_secondary_direction(self) -> np.ndarray:
        """Derive the orthogonal secondary modulation axis."""
        secondary = np.cross(self.propagation_direction, self.modulation_direction)
        norm = np.linalg.norm(secondary)
        if norm < 1e-10:
            raise ValueError("Failed to determine secondary modulation direction")
        return secondary / norm

    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_elements = self.num_elements_param

        A = np.zeros((num_elements, 3), dtype=float)
        B = np.zeros((num_elements, 3), dtype=float)
        I = np.full(num_elements, self.current, dtype=float)

        phase_offset = self.phase_difference_2ndaxis
        secondary_dir = self.secondary_modulation_direction

        for i in range(num_elements):
            z_start = i * self.element_spacing
            z_end = (i + 1) * self.element_spacing

            phase_start = self.omega * time + self.k * z_start
            phase_end = self.omega * time + self.k * z_end

            deflection_start_primary = self.modulation_amplitude * np.sin(phase_start)
            deflection_end_primary = self.modulation_amplitude * np.sin(phase_end)

            deflection_start_secondary = self.modulation_amplitude_2ndaxis * np.sin(phase_start + phase_offset)
            deflection_end_secondary = self.modulation_amplitude_2ndaxis * np.sin(phase_end + phase_offset)

            A[i] = (
                self.start_position +
                z_start * self.propagation_direction +
                deflection_start_primary * self.modulation_direction +
                deflection_start_secondary * secondary_dir
            )

            B[i] = (
                self.start_position +
                z_end * self.propagation_direction +
                deflection_end_primary * self.modulation_direction +
                deflection_end_secondary * secondary_dir
            )

        logger.debug(
            "Generated %d dual-axis oscillating beam elements at time %ss",
            num_elements,
            time
        )

        return A, B, I

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        start_pos = self.start_position
        end_pos = self.start_position + self.length * self.propagation_direction

        deflection_vectors = []
        for sign_primary in (-1.0, 1.0):
            for sign_secondary in (-1.0, 1.0):
                deflection_vectors.append(
                    sign_primary * self.modulation_amplitude * self.modulation_direction +
                    sign_secondary * self.modulation_amplitude_2ndaxis * self.secondary_modulation_direction
                )

        corner_points = []
        for base_point in (start_pos, end_pos):
            for deflection in deflection_vectors:
                corner_points.append(base_point + deflection)

        corners = np.asarray(corner_points)
        return {
            'x': (corners[:, 0].min(), corners[:, 0].max()),
            'y': (corners[:, 1].min(), corners[:, 1].max()),
            'z': (corners[:, 2].min(), corners[:, 2].max())
        }

    def set_modulation_parameters(self,
                                  frequency: float,
                                  amplitude: float,
                                  amplitude_2ndaxis: Optional[float] = None,
                                  phase_difference_2ndaxis: Optional[float] = None):
        super().set_modulation_parameters(frequency, amplitude)

        if amplitude_2ndaxis is not None:
            self.modulation_amplitude_2ndaxis = float(amplitude_2ndaxis)
            self.parameters['modulation_amplitude_2ndaxis'] = self.modulation_amplitude_2ndaxis

        if phase_difference_2ndaxis is not None:
            self.phase_difference_2ndaxis = float(phase_difference_2ndaxis)
            self.parameters['phase_difference_2ndaxis'] = self.phase_difference_2ndaxis

        logger.debug(
            "Updated dual-axis modulation: f=%sHz, A1=%sm, A2=%sm, phase_shift=%srad",
            self.modulation_frequency,
            self.modulation_amplitude,
            self.modulation_amplitude_2ndaxis,
            self.phase_difference_2ndaxis
        )

    def __str__(self) -> str:
        return (
            f"OscillatingBeam2D('{self.name}', voltage={self.voltage}V, "
            f"current={self.current}A, f_mod={self.modulation_frequency}Hz, "
            f"length={self.length}m, A1={self.modulation_amplitude}m, "
            f"A2={self.modulation_amplitude_2ndaxis}m, phase={self.phase_difference_2ndaxis}rad, "
            f"{self.num_elements_param} elements)"
        )
