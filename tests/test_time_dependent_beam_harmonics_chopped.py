
"""Harmonic analysis regression test for the chopped beam."""

import os
import sys
import unittest
from typing import List

import numpy as np

# Allow running tests directly from repository root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from maga import MagneticFieldCalculator, RectangularGrid, ChoppedBeam
    MAGA_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    MAGA_AVAILABLE = False
    IMPORT_ERROR = str(exc)


# Driving modulation frequency for FFT annotations (Hz)
DRIVING_FREQUENCY = 200e6

# Fixed x coordinate for the sampling line in the (x, y, 0) plane (meters)
LINE_SAMPLE_X = 0.0005


class TestTimeDependentBeamHarmonics(unittest.TestCase):
    """Sample a spatial line and analyse the harmonic content over time."""

    def setUp(self):
        if not MAGA_AVAILABLE:
            self.skipTest(f"MAGA library not available: {IMPORT_ERROR}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"matplotlib not available: {exc}")

        self.plt = plt

        try:
            self.calculator = MagneticFieldCalculator()
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"MagneticFieldCalculator unavailable: {exc}")

    def test_time_dependent_beam_line_harmonics(self):
        output_dir = os.path.join(
            os.path.dirname(__file__),
            'artifacts',
            'chopped_beam_harmonics',
        )
        os.makedirs(output_dir, exist_ok=True)

        modulation_frequency = DRIVING_FREQUENCY
        modulation_period = 1.0 / modulation_frequency

        beam = ChoppedBeam(
            voltage=10000.0,
            current=10e-6,
            modulation_frequency=modulation_frequency,
            start_position=(0.0, 0.0, 0.1),
            propagation_direction=(0.0, 0.0, -1.0),
            length=0.2,
            duty_cycle=0.5,
        )

        grid = RectangularGrid(
            x_range=(-0.01, 0.010),
            y_range=(-0.01, 0.010),
            z_range=(-0.001, 0.001),
            nx=51,
            ny=501,
            nz=3,
            name='chopped_beam_grid',
        )

        grid.generate_coordinates()
        nx, ny, nz = grid.grid_shape

        x = np.linspace(grid.x_range[0], grid.x_range[1], nx)
        y = np.linspace(grid.y_range[0], grid.y_range[1], ny)
        z = np.linspace(grid.z_range[0], grid.z_range[1], nz)
        coord_x, coord_y, coord_z = np.meshgrid(x, y, z, indexing='ij')

        line_x = LINE_SAMPLE_X
        self.assertTrue(
            grid.x_range[0] <= line_x <= grid.x_range[1],
            'LINE_SAMPLE_X must lie within the grid bounds.',
        )
        line_iy = np.arange(ny)
        line_ix = int(np.argmin(np.abs(x - line_x)))
        line_iz = int(np.argmin(np.abs(z - 0.0)))

        timestamps = np.linspace(0.0, modulation_period * 10, num=4 * 60, endpoint=False)
        line_bx: List[np.ndarray] = []
        line_by: List[np.ndarray] = []

        for time_point in timestamps:
            geometry_A, geometry_B, geometry_I = beam.get_geometry(time=time_point)
            result = self.calculator.calculate_magnetic_field(
                geometry_A,
                geometry_B,
                geometry_I,
                grid,
                simulation_time=time_point,
            )

            magnetic_field = result.magnetic_field
            grid_coords = result.grid_coordinates

            if magnetic_field.ndim == 2:
                self.assertEqual(magnetic_field.shape, (grid.num_points, 3))
                flat_field = magnetic_field
            else:
                self.assertEqual(magnetic_field.shape, (nx, ny, nz, 3))
                flat_field = magnetic_field.reshape(-1, 3)

            field_vectors = np.reshape(flat_field, (nx, ny, nz, 3), order='F')

            self.assertEqual(grid_coords.ndim, 2)
            self.assertEqual(grid_coords.shape, (grid.num_points, 3))

            coords_from_result = grid_coords.reshape(nx, ny, nz, 3)
            self.assertTrue(np.allclose(coords_from_result[..., 0], coord_x))
            self.assertTrue(np.allclose(coords_from_result[..., 1], coord_y))
            self.assertTrue(np.allclose(coords_from_result[..., 2], coord_z))

            line_field = field_vectors[line_ix, line_iy, line_iz, :]
            line_bx.append(line_field[:, 0])
            line_by.append(line_field[:, 1])

        line_bx_arr = np.asarray(line_bx)
        line_by_arr = np.asarray(line_by)

        sample_spacing = float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
        files_created: List[str] = []

        fundamental_path = os.path.join(output_dir, 'chopped_beam_line_fundamental.png')
        second_harmonic_path = os.path.join(output_dir, 'chopped_beam_line_second_harmonic.png')
        third_harmonic_path = os.path.join(output_dir, 'chopped_beam_line_third_harmonic.png')

        if line_bx_arr.shape[0] > 1 and sample_spacing > 0.0:
            freqs = np.fft.rfftfreq(line_bx_arr.shape[0], d=sample_spacing)
            freq_mhz = freqs * 1e-6

            centered_bx = line_bx_arr - line_bx_arr.mean(axis=0, keepdims=True)
            centered_by = line_by_arr - line_by_arr.mean(axis=0, keepdims=True)

            spectrum_bx = np.abs(np.fft.rfft(centered_bx, axis=0)) / line_bx_arr.shape[0]
            spectrum_by = np.abs(np.fft.rfft(centered_by, axis=0)) / line_bx_arr.shape[0]

            def extract_component(target_freq: float) -> tuple[np.ndarray, np.ndarray]:
                if freqs.size and target_freq <= freqs[-1]:
                    idx = int(np.argmin(np.abs(freqs - target_freq)))
                    return spectrum_bx[idx, :], spectrum_by[idx, :]
                return (np.zeros_like(y), np.zeros_like(y))

            fundamental_bx, fundamental_by = extract_component(DRIVING_FREQUENCY)
            second_bx, second_by = extract_component(2.0 * DRIVING_FREQUENCY)
            third_bx, third_by = extract_component(3.0 * DRIVING_FREQUENCY)

            line_axis_mm = y * 1e3
            line_x_mm = x[line_ix] * 1e3

            fig_fundamental = self.plt.figure(figsize=(9, 5))
            ax_fundamental = fig_fundamental.add_subplot(111)
            ax_fundamental.plot(line_axis_mm, fundamental_bx, color='tab:blue', label='Bx amplitude')
            ax_fundamental.plot(line_axis_mm, fundamental_by, color='tab:orange', label='By amplitude')
            ax_fundamental.set_xlabel('Y [mm]')
            ax_fundamental.set_ylabel('Amplitude [T]')
            ax_fundamental.set_title(
                f'Line spectrum at x = {line_x_mm:.2f} mm (fundamental {DRIVING_FREQUENCY * 1e-6:.1f} MHz)'
            )
            ax_fundamental.grid(True, alpha=0.3)
            ax_fundamental.legend(loc='upper right')
            fig_fundamental.tight_layout()
            fig_fundamental.savefig(fundamental_path, dpi=240)
            self.plt.close(fig_fundamental)
            files_created.append(fundamental_path)

            fig_second = self.plt.figure(figsize=(9, 5))
            ax_second = fig_second.add_subplot(111)
            ax_second.plot(line_axis_mm, second_bx, color='tab:blue', label='Bx amplitude')
            ax_second.plot(line_axis_mm, second_by, color='tab:orange', label='By amplitude')
            ax_second.set_xlabel('Y [mm]')
            ax_second.set_ylabel('Amplitude [T]')
            ax_second.set_title(
                f'Line spectrum at x = {line_x_mm:.2f} mm (second harmonic {2.0 * DRIVING_FREQUENCY * 1e-6:.1f} MHz)'
            )
            ax_second.grid(True, alpha=0.3)
            ax_second.legend(loc='upper right')
            fig_second.tight_layout()
            fig_second.savefig(second_harmonic_path, dpi=240)
            self.plt.close(fig_second)
            files_created.append(second_harmonic_path)

            fig_third = self.plt.figure(figsize=(9, 5))
            ax_third = fig_third.add_subplot(111)
            ax_third.plot(line_axis_mm, third_bx, color='tab:blue', label='Bx amplitude')
            ax_third.plot(line_axis_mm, third_by, color='tab:orange', label='By amplitude')
            ax_third.set_xlabel('Y [mm]')
            ax_third.set_ylabel('Amplitude [T]')
            ax_third.set_title(
                f'Line spectrum at x = {line_x_mm:.2f} mm (third harmonic {3.0 * DRIVING_FREQUENCY * 1e-6:.1f} MHz)'
            )
            ax_third.grid(True, alpha=0.3)
            ax_third.legend(loc='upper right')
            fig_third.tight_layout()
            fig_third.savefig(third_harmonic_path, dpi=240)
            self.plt.close(fig_third)
            files_created.append(third_harmonic_path)
        else:
            for destination, message in [
                (fundamental_path, 'Insufficient samples for fundamental spectrum'),
                (second_harmonic_path, 'Insufficient samples for second harmonic spectrum'),
                (third_harmonic_path, 'Insufficient samples for third harmonic spectrum'),
            ]:
                fig_placeholder = self.plt.figure(figsize=(9, 5))
                ax_placeholder = fig_placeholder.add_subplot(111)
                ax_placeholder.text(0.5, 0.5, message, ha='center', va='center', transform=ax_placeholder.transAxes)
                ax_placeholder.axis('off')
                fig_placeholder.tight_layout()
                fig_placeholder.savefig(destination, dpi=240)
                self.plt.close(fig_placeholder)
                files_created.append(destination)

        for path_created in files_created:
            self.assertTrue(os.path.exists(path_created), f'Expected artifact missing: {path_created}')

        self.assertGreater(len(files_created), 0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
