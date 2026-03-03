"""
Chopped electron beam geometry generator for MAGA library.

This module implements the ChoppedBeam class which models a straight electron
beam that is periodically enabled and disabled ("chopped"). The beam is
represented as a collection of line elements during the "on" portions of the
modulation cycle with gaps where the beam is off. The chopping pattern is
mapped from the temporal modulation to spatial segments using the
relativistically corrected beam velocity.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Tuple, List

import numpy as np

from .base import BaseGeometry, GeometryParameters

logger = logging.getLogger(__name__)

# Physical constants (duplicated here to avoid circular imports)
ELECTRON_CHARGE = 1.60217663e-19  # Coulombs
ELECTRON_MASS = 9.1093837015e-31  # kg
SPEED_OF_LIGHT = 299792458.0      # m/s


class ChoppedBeam(BaseGeometry):
    """Straight electron beam with periodic on/off modulation."""

    def __init__(
        self,
        voltage: float,
        current: float,
        modulation_frequency: float,
        start_position: Tuple[float, float, float],
        propagation_direction: Tuple[float, float, float],
        length: float,
        duty_cycle: float,
        name: str = "chopped_beam",
    ) -> None:
        parameters = GeometryParameters(
            voltage=voltage,
            current=current,
            modulation_frequency=modulation_frequency,
            start_position=start_position,
            propagation_direction=propagation_direction,
            length=length,
            duty_cycle=duty_cycle,
        )

        super().__init__(name, parameters)

        if voltage <= 0:
            raise ValueError("Voltage must be positive")
        if modulation_frequency <= 0:
            raise ValueError("Modulation frequency must be positive")
        if length <= 0:
            raise ValueError("Beam length must be positive")
        if not (0.0 <= duty_cycle <= 1.0):
            raise ValueError("Duty cycle must be within [0, 1]")

        self.voltage = float(voltage)
        self.current = float(current)
        self.modulation_frequency = float(modulation_frequency)
        self.length = float(length)
        self.duty_cycle = float(duty_cycle)

        self.start_position = np.asarray(start_position, dtype=float)
        self.propagation_direction = np.asarray(propagation_direction, dtype=float)

        self._normalize_direction()
        self._calculate_beam_physics()
        self._update_chop_parameters()

        logger.debug(
            "Created chopped beam: voltage=%s V, current=%s A, f_mod=%s Hz, length=%s m, duty=%s",
            self.voltage,
            self.current,
            self.modulation_frequency,
            self.length,
            self.duty_cycle,
        )

    def _normalize_direction(self) -> None:
        norm = np.linalg.norm(self.propagation_direction)
        if norm < 1e-10:
            raise ValueError("Propagation direction cannot be zero vector")
        self.propagation_direction = self.propagation_direction / norm

    def _calculate_beam_physics(self) -> None:
        gamma_factor = 1.0 + (
            self.voltage * ELECTRON_CHARGE
        ) / (ELECTRON_MASS * SPEED_OF_LIGHT ** 2)
        self.velocity = SPEED_OF_LIGHT * np.sqrt(1.0 - 1.0 / (gamma_factor ** 2))
        self.wavelength = self.velocity / self.modulation_frequency
        self.period = 1.0 / self.modulation_frequency
        self.parameters['velocity'] = self.velocity
        self.parameters['wavelength'] = self.wavelength
        self.parameters['period'] = self.period
        logger.debug(
            "Beam physics: velocity=%s m/s, wavelength=%s m",
            self.velocity,
            self.wavelength,
        )

    def _update_chop_parameters(self) -> None:
        """Update chopping-derived temporal and spatial metrics."""
        self.on_duration = self.duty_cycle * self.period
        self.off_duration = max(0.0, self.period - self.on_duration)
        self.on_length = self.velocity * self.on_duration
        self.parameters['on_duration'] = self.on_duration
        self.parameters['off_duration'] = self.off_duration
        self.parameters['on_length'] = self.on_length

    def _invalidate_cache(self) -> None:
        """Reset cached geometry so it will regenerate on next request."""
        self._geometry_A = None
        self._geometry_B = None
        self._geometry_I = None

    def _compute_active_segments(self, time: float) -> List[Tuple[float, float]]:
        if self.duty_cycle <= 0.0 or self.on_duration <= 0.0:
            return []

        period = self.period
        velocity = self.velocity
        if period <= 0.0 or velocity <= 0.0:
            return []

        horizon = (self.length / velocity) + self.on_duration
        n_current = math.floor(time / period)
        n_min = math.floor((time - horizon) / period)

        segments: List[Tuple[float, float]] = []
        for n in range(n_min, n_current + 1):
            t_cycle = n * period
            dt = time - t_cycle
            if dt <= 0.0:
                continue

            segment_end = velocity * dt
            if dt > self.on_duration:
                segment_start = velocity * (dt - self.on_duration)
            else:
                segment_start = 0.0

            start_clipped = max(0.0, segment_start)
            end_clipped = min(self.length, segment_end)

            if end_clipped <= start_clipped:
                continue

            segments.append((start_clipped, end_clipped))

        if not segments:
            return []

        segments.sort(key=lambda item: item[0])
        merged: List[Tuple[float, float]] = []
        current_start, current_end = segments[0]
        for start, end in segments[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        merged.append((current_start, current_end))

        return merged

    def generate_geometry(self, time: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        segments = self._compute_active_segments(time)
        num_segments = len(segments)

        if num_segments == 0:
            empty_points = np.zeros((0, 3), dtype=float)
            empty_current = np.zeros((0,), dtype=float)
            return empty_points, empty_points.copy(), empty_current

        A = np.zeros((num_segments, 3), dtype=float)
        B = np.zeros((num_segments, 3), dtype=float)
        I = np.full(num_segments, self.current, dtype=float)

        for idx, (start, end) in enumerate(segments):
            A[idx] = self.start_position + start * self.propagation_direction
            B[idx] = self.start_position + end * self.propagation_direction

        logger.debug("Generated %s chopped beam elements", num_segments)
        return A, B, I

    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        start_pos = self.start_position
        end_pos = self.start_position + self.length * self.propagation_direction
        points = np.vstack((start_pos, end_pos))
        return {
            'x': (float(points[:, 0].min()), float(points[:, 0].max())),
            'y': (float(points[:, 1].min()), float(points[:, 1].max())),
            'z': (float(points[:, 2].min()), float(points[:, 2].max())),
        }

    def get_modulation_wavelength(self) -> float:
        return self.wavelength

    def get_beta(self) -> float:
        return self.velocity / SPEED_OF_LIGHT

    def get_gamma(self) -> float:
        beta = self.get_beta()
        return 1.0 / np.sqrt(1.0 - beta ** 2)

    def get_kinetic_energy(self) -> float:
        return self.voltage * ELECTRON_CHARGE

    def get_kinetic_energy_eV(self) -> float:
        return self.voltage

    def get_instantaneous_frequency(self, position: float, time: float = 0.0) -> float:
        return self.modulation_frequency

    def is_time_dependent(self) -> bool:
        return True

    def get_beam_velocity(self) -> float:
        return self.velocity

    def set_voltage(self, voltage: float) -> None:
        if voltage <= 0:
            raise ValueError("Voltage must be positive")
        self.voltage = float(voltage)
        self.parameters['voltage'] = self.voltage
        self._calculate_beam_physics()
        self._update_chop_parameters()
        self._invalidate_cache()

    def set_current(self, current: float) -> None:
        self.current = float(current)
        self.parameters['current'] = self.current
        self._invalidate_cache()

    def set_modulation_frequency(self, frequency: float) -> None:
        if frequency <= 0:
            raise ValueError("Modulation frequency must be positive")
        self.modulation_frequency = float(frequency)
        self.parameters['modulation_frequency'] = self.modulation_frequency
        self._calculate_beam_physics()
        self._update_chop_parameters()
        self._invalidate_cache()

    def set_duty_cycle(self, duty_cycle: float) -> None:
        if not (0.0 <= duty_cycle <= 1.0):
            raise ValueError("Duty cycle must be within [0, 1]")
        self.duty_cycle = float(duty_cycle)
        self.parameters['duty_cycle'] = self.duty_cycle
        self._update_chop_parameters()
        self._invalidate_cache()

    def __str__(self) -> str:
        return (
            f"ChoppedBeam('{self.name}', voltage={self.voltage}V, current={self.current}A, "
            f"f_mod={self.modulation_frequency}Hz, length={self.length}m, duty={self.duty_cycle})"
        )
