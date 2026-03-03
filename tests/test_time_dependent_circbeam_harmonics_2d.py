"""2D harmonic analysis regression test for the circularly modulated beam."""

import os
import sys
import unittest
from typing import List

import numpy as np

# Allow running tests directly from repository root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from maga import MagneticFieldCalculator, RectangularGrid, OscillatingBeam2D
    MAGA_AVAILABLE = True
except ImportError as exc:  # pragma: no cover
    MAGA_AVAILABLE = False
    IMPORT_ERROR = str(exc)


# Driving modulation frequency shared across visualisation tests (Hz)
DRIVING_FREQUENCY = 200e6

# Z plane to sample (meters)
PLANE_SAMPLE_Z = 0.0


class TestTimeDependentCircBeamHarmonics2D(unittest.TestCase):
    """Capture a full plane over time and analyse harmonic content spatially."""

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

    def test_time_dependent_circbeam_slice_harmonics(self):
        output_dir = os.path.join(
            os.path.dirname(__file__),
            'artifacts',
            'oscillating_circbeam_harmonics_2d',
        )
        os.makedirs(output_dir, exist_ok=True)

        modulation_frequency = DRIVING_FREQUENCY
        modulation_period = 1.0 / modulation_frequency

        modulation_amplitude = 0.001
        beam = OscillatingBeam2D(
            voltage=10000.0,
            current=10e-6,
            modulation_frequency=modulation_frequency,
            modulation_amplitude=modulation_amplitude,
            start_position=(0.0, 0.0, 0.1),
            propagation_direction=(0.0, 0.0, -1.0),
            modulation_direction=(0.0, 1.0, 0.0),
            length=0.2,
            num_elements=5000,
            modulation_amplitude_2ndaxis=modulation_amplitude,
            phase_difference_2ndaxis=np.pi / 2.0,
        )

        grid = RectangularGrid(
            x_range=(-0.0025, 0.0025),
            y_range=(-0.0025, 0.0025),
            z_range=(-0.001, 0.001),
            nx=201,
            ny=201,
            nz=3,
            name='oscillating_circbeam_grid',
        )

        grid.generate_coordinates()
        nx, ny, nz = grid.grid_shape

        x = np.linspace(grid.x_range[0], grid.x_range[1], nx)
        y = np.linspace(grid.y_range[0], grid.y_range[1], ny)
        z = np.linspace(grid.z_range[0], grid.z_range[1], nz)
        coord_x, coord_y, coord_z = np.meshgrid(x, y, z, indexing='ij')

        self.assertTrue(
            grid.z_range[0] <= PLANE_SAMPLE_Z <= grid.z_range[1],
            'PLANE_SAMPLE_Z must lie within the grid bounds.',
        )
        plane_iz = int(np.argmin(np.abs(z - PLANE_SAMPLE_Z)))
        center_ix = int(np.argmin(np.abs(x - 0.0)))

        timestamps = np.linspace(0.0, modulation_period * 10, num=4 * 120, endpoint=False)
        num_times = timestamps.size

        slice_bx = np.empty((num_times, nx, ny), dtype=float)
        slice_by = np.empty((num_times, nx, ny), dtype=float)

        for frame_idx, time_point in enumerate(timestamps):
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

            slice_bx[frame_idx] = field_vectors[:, :, plane_iz, 0]
            slice_by[frame_idx] = field_vectors[:, :, plane_iz, 1]
            slice_bx[frame_idx, center_ix, :] = 0.0
            slice_by[frame_idx, center_ix, :] = 0.0

        sample_spacing = float(np.mean(np.diff(timestamps))) if num_times > 1 else 0.0
        files_created: List[str] = []

        npz_path = os.path.join(output_dir, 'oscillating_circbeam_slice_fields.npz')
        np.savez_compressed(
            npz_path,
            timestamps=timestamps,
            x=x,
            y=y,
            slice_bx=slice_bx,
            slice_by=slice_by,
            plane_z=z[plane_iz],
        )
        files_created.append(npz_path)

        fundamental_path = os.path.join(output_dir, 'oscillating_circbeam_slice_fundamental.png')
        second_harmonic_path = os.path.join(output_dir, 'oscillating_circbeam_slice_second_harmonic.png')

        if num_times > 1 and sample_spacing > 0.0:
            freqs = np.fft.rfftfreq(num_times, d=sample_spacing)

            centered_bx = slice_bx - slice_bx.mean(axis=0, keepdims=True)
            centered_by = slice_by - slice_by.mean(axis=0, keepdims=True)

            spectrum_bx = np.abs(np.fft.rfft(centered_bx, axis=0)) / num_times
            spectrum_by = np.abs(np.fft.rfft(centered_by, axis=0)) / num_times

            def extract_component(target_freq: float) -> tuple[np.ndarray, np.ndarray]:
                if freqs.size and target_freq <= freqs[-1]:
                    idx = int(np.argmin(np.abs(freqs - target_freq)))
                    return spectrum_bx[idx, :, :], spectrum_by[idx, :, :]
                return (np.zeros((nx, ny)), np.zeros((nx, ny)))

            fundamental_bx, fundamental_by = extract_component(DRIVING_FREQUENCY)
            second_bx, second_by = extract_component(2.0 * DRIVING_FREQUENCY)

            x_mm = x * 1e3
            y_mm = y * 1e3
            extent = (x_mm[0], x_mm[-1], y_mm[0], y_mm[-1])

            fig_fundamental = self.plt.figure(figsize=(10, 5))
            ax_fundamental = fig_fundamental.add_subplot(121)
            ax_fundamental.set_title('Fundamental Bx amplitude')
            im_bx = ax_fundamental.imshow(
                fundamental_bx.T,
                origin='lower',
                extent=extent,
                cmap='inferno',
            )
            fig_fundamental.colorbar(im_bx, ax=ax_fundamental, fraction=0.046, pad=0.04, label='|Bx| [T]')
            ax_fundamental.set_xlabel('X [mm]')
            ax_fundamental.set_ylabel('Y [mm]')

            ax_fundamental_by = fig_fundamental.add_subplot(122)
            ax_fundamental_by.set_title('Fundamental By amplitude')
            im_by = ax_fundamental_by.imshow(
                fundamental_by.T,
                origin='lower',
                extent=extent,
                cmap='inferno',
            )
            fig_fundamental.colorbar(im_by, ax=ax_fundamental_by, fraction=0.046, pad=0.04, label='|By| [T]')
            ax_fundamental_by.set_xlabel('X [mm]')
            ax_fundamental_by.set_ylabel('Y [mm]')
            fig_fundamental.tight_layout()
            fig_fundamental.savefig(fundamental_path, dpi=240)
            self.plt.close(fig_fundamental)
            files_created.append(fundamental_path)

            fig_second = self.plt.figure(figsize=(10, 5))
            ax_second = fig_second.add_subplot(121)
            ax_second.set_title('Second harmonic Bx amplitude')
            im_second_bx = ax_second.imshow(
                second_bx.T,
                origin='lower',
                extent=extent,
                cmap='inferno',
            )
            fig_second.colorbar(im_second_bx, ax=ax_second, fraction=0.046, pad=0.04, label='|Bx| [T]')
            ax_second.set_xlabel('X [mm]')
            ax_second.set_ylabel('Y [mm]')

            ax_second_by = fig_second.add_subplot(122)
            ax_second_by.set_title('Second harmonic By amplitude')
            im_second_by = ax_second_by.imshow(
                second_by.T,
                origin='lower',
                extent=extent,
                cmap='inferno',
            )
            fig_second.colorbar(im_second_by, ax=ax_second_by, fraction=0.046, pad=0.04, label='|By| [T]')
            ax_second_by.set_xlabel('X [mm]')
            ax_second_by.set_ylabel('Y [mm]')
            fig_second.tight_layout()
            fig_second.savefig(second_harmonic_path, dpi=240)
            self.plt.close(fig_second)
            files_created.append(second_harmonic_path)

        self.assertTrue(files_created, 'Expected at least one artefact to be created for inspection.')
        for created in files_created:
            self.assertTrue(os.path.exists(created), f"Artefact missing: {created}")


if __name__ == '__main__':
    unittest.main()
