"""Visualization regression tests for the circularly modulated oscillating beam."""

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


# Configurable trace point in the (x, y, 0) plane (meters)
TRACE_POINT = (0.0005, 0.0, 0.0)

# Driving modulation frequency for FFT annotations (Hz)
DRIVING_FREQUENCY = 200e6

COLORBAR_MAX_TESLA = 5e-8


class TestTimeDependentCircBeamVisualization(unittest.TestCase):
    """Render a short frame sequence of the circularly polarized beam."""

    def setUp(self):
        if not MAGA_AVAILABLE:
            self.skipTest(f"MAGA library not available: {IMPORT_ERROR}")

        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt, cm, colors
        except ImportError as exc:  # pragma: no cover
            self.skipTest(f"matplotlib not available: {exc}")

        self.plt = plt
        self.cm = cm
        self.colors = colors

        try:
            self.calculator = MagneticFieldCalculator()
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"MagneticFieldCalculator unavailable: {exc}")

    def test_time_dependent_circbeam_frames(self):
        output_dir = os.path.join(os.path.dirname(__file__), "artifacts", "oscillating_circbeam_frames")
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
            x_range=(-0.01, 0.010),
            y_range=(-0.01, 0.010),
            z_range=(-0.001, 0.001),
            nx=101,
            ny=101,
            nz=11,
            name="oscillating_circbeam_grid",
        )

        grid.generate_coordinates()
        nx, ny, nz = grid.grid_shape

        x = np.linspace(grid.x_range[0], grid.x_range[1], nx)
        y = np.linspace(grid.y_range[0], grid.y_range[1], ny)
        z = np.linspace(grid.z_range[0], grid.z_range[1], nz)
        coord_x, coord_y, coord_z = np.meshgrid(x, y, z, indexing="ij")

        trace_point = np.asarray(TRACE_POINT, dtype=float)
        self.assertEqual(trace_point.shape, (3,), "TRACE_POINT must contain three coordinates.")
        self.assertTrue(np.isclose(trace_point[2], 0.0), "TRACE_POINT must lie on the z = 0 plane.")
        self.assertTrue(grid.x_range[0] <= trace_point[0] <= grid.x_range[1], "TRACE_POINT x-coordinate outside grid bounds.")
        self.assertTrue(grid.y_range[0] <= trace_point[1] <= grid.y_range[1], "TRACE_POINT y-coordinate outside grid bounds.")

        trace_ix = int(np.argmin(np.abs(x - trace_point[0])))
        trace_iy = int(np.argmin(np.abs(y - trace_point[1])))
        trace_iz = int(np.argmin(np.abs(z - trace_point[2])))
        grid_trace_point = np.array([x[trace_ix], y[trace_iy], z[trace_iz]])
        trace_point_mm = grid_trace_point * 1e3

        timestamps = np.linspace(0.0, modulation_period * 10, num=4 * 60, endpoint=False)
        frame_paths: List[str] = []

        trace_times: List[float] = []
        trace_components_x: List[float] = []
        trace_components_y: List[float] = []

        simulation_duration = modulation_period * 10.0
        time_axis_max_ns = simulation_duration * 1e9

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

            field_vectors = np.reshape(flat_field, (nx, ny, nz, 3), order="F")

            self.assertEqual(grid_coords.ndim, 2)
            self.assertEqual(grid_coords.shape, (grid.num_points, 3))

            coords_from_result = grid_coords.reshape(nx, ny, nz, 3)
            self.assertTrue(np.allclose(coords_from_result[..., 0], coord_x))
            self.assertTrue(np.allclose(coords_from_result[..., 1], coord_y))
            self.assertTrue(np.allclose(coords_from_result[..., 2], coord_z))

            trace_vector = field_vectors[trace_ix, trace_iy, trace_iz, :]
            trace_times.append(time_point)
            trace_components_x.append(trace_vector[0])
            trace_components_y.append(trace_vector[1])

            fig = self.plt.figure(figsize=(12, 7))
            gridspec = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0])
            ax3d = fig.add_subplot(gridspec[0, 0], projection="3d")
            trace_ax = fig.add_subplot(gridspec[0, 1])

            beam_path = np.vstack((geometry_A[0], geometry_B))
            distinct_indices = [0]
            for idx in range(1, beam_path.shape[0]):
                if not np.allclose(beam_path[idx], beam_path[distinct_indices[-1]]):
                    distinct_indices.append(idx)
            beam_path = beam_path[distinct_indices]
            beam_path_mm = beam_path * 1e3

            zmin, zmax = -2.0, 2.0
            mask = (beam_path_mm[:, 2] >= zmin) & (beam_path_mm[:, 2] <= zmax)
            beam_path_mm = beam_path_mm[mask]

            (beam_line,) = ax3d.plot(
                beam_path_mm[:, 0],
                beam_path_mm[:, 1],
                beam_path_mm[:, 2],
                color="green",
                linewidth=1.0,
                label="Beam path",
            )
            beam_line.set_antialiased(False)

            step_x = max(1, nx // 5)
            step_y = max(1, ny // 7)
            sampled_field = field_vectors[::step_x, ::step_y, :, :]

            sampled_x = coord_x[::step_x, ::step_y, :] * 1e3
            sampled_y = coord_y[::step_x, ::step_y, :] * 1e3
            sampled_z = coord_z[::step_x, ::step_y, :] * 1e3

            X = sampled_x.ravel()
            Y = sampled_y.ravel()
            Z = sampled_z.ravel()
            U = sampled_field[..., 0].ravel()
            V = sampled_field[..., 1].ravel()
            W = sampled_field[..., 2].ravel()

            sampled_strength = np.sqrt(U ** 2 + V ** 2 + W ** 2)

            central_layer = nz // 2
            plane_field = field_vectors[:, :, central_layer, :]
            plane_strength = np.linalg.norm(plane_field, axis=-1)

            quiver_norm = self.colors.Normalize(vmin=0.0, vmax=COLORBAR_MAX_TESLA)
            plane_norm = self.colors.Normalize(vmin=0.0, vmax=COLORBAR_MAX_TESLA)

            color_map = self.cm.get_cmap('viridis')
            colors_quiver = color_map(quiver_norm(np.clip(sampled_strength, 0.0, COLORBAR_MAX_TESLA)))

            ax3d.quiver(
                X, Y, Z,
                U, V, W,
                length=0.5,
                normalize=False,
                colors=colors_quiver,
                linewidths=0.5,
            )
            ax3d.set_xlabel('X [mm]')
            ax3d.set_ylabel('Y [mm]')
            ax3d.set_zlabel('Z [mm]')
            ax3d.set_title(f'Circular beam field vectors at t = {time_point * 1e9:.2f} ns')
            ax3d.set_box_aspect((1, 1, 1))

            plane_levels = np.linspace(0.0, COLORBAR_MAX_TESLA, num=21)
            plane_im = ax3d.contourf(
                coord_x[:, :, central_layer] * 1e3,
                coord_y[:, :, central_layer] * 1e3,
                np.clip(plane_strength, 0.0, COLORBAR_MAX_TESLA),
                levels=plane_levels,
                zdir='z',
                offset=coord_z[0, 0, central_layer] * 1e3,
                cmap='magma',
                norm=plane_norm,
                alpha=0.6,
            )
            scalar_mappable = self.cm.ScalarMappable(norm=plane_norm, cmap='magma')
            scalar_mappable.set_array([])
            colorbar = fig.colorbar(
                scalar_mappable,
                ax=ax3d,
                fraction=0.046,
                pad=0.04,
                label='|B| [T]',
            )
            colorbar.set_ticks(np.linspace(0.0, COLORBAR_MAX_TESLA, 5))

            trace_ax.plot(np.array(trace_times) * 1e9, trace_components_x, color='tab:blue', label='Bx')
            trace_ax.plot(np.array(trace_times) * 1e9, trace_components_y, color='tab:orange', label='By')
            trace_ax.set_xlim(0.0, time_axis_max_ns)
            trace_ax.set_xlabel('Time [ns]')
            trace_ax.set_ylabel('Field [T]')
            trace_ax.set_title(
                f'Magnetic field at ({trace_point_mm[0]:.2f}, {trace_point_mm[1]:.2f}, {trace_point_mm[2]:.2f}) mm'
            )
            trace_ax.grid(True, alpha=0.3)
            trace_ax.legend(loc='upper right')

            fig.tight_layout()
            frame_path = os.path.join(output_dir, f'circular_beam_frame_{frame_idx:04d}.png')
            fig.savefig(frame_path, dpi=180)
            self.plt.close(fig)
            frame_paths.append(frame_path)

        self.assertTrue(frame_paths, 'Expected frame artefacts for inspection.')
        for created in frame_paths:
            self.assertTrue(os.path.exists(created), f"Artefact missing: {created}")

        trace_array_x = np.asarray(trace_components_x)
        trace_array_y = np.asarray(trace_components_y)
        sample_spacing = float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
        if len(timestamps) > 1 and sample_spacing > 0.0:
            freqs = np.fft.rfftfreq(len(timestamps), d=sample_spacing)
            centered_x = trace_array_x - trace_array_x.mean()
            centered_y = trace_array_y - trace_array_y.mean()
            spectrum_x = np.abs(np.fft.rfft(centered_x)) / len(timestamps)
            spectrum_y = np.abs(np.fft.rfft(centered_y)) / len(timestamps)

            amplitude_path = os.path.join(output_dir, 'circular_beam_trace_spectrum.png')
            fig_amp = self.plt.figure(figsize=(8, 4))
            ax_amp = fig_amp.add_subplot(111)
            ax_amp.plot(freqs * 1e-6, spectrum_x, label='|Bx|')
            ax_amp.plot(freqs * 1e-6, spectrum_y, label='|By|')
            ax_amp.set_xlabel('Frequency [MHz]')
            ax_amp.set_ylabel('Spectrum amplitude [T]')
            ax_amp.set_title('Trace spectrum for circular beam')
            ax_amp.set_xlim(0.0, DRIVING_FREQUENCY * 3.0 * 1e-6)
            ax_amp.grid(True, alpha=0.3)
            ax_amp.legend(loc='upper right')
            fig_amp.tight_layout()
            fig_amp.savefig(amplitude_path, dpi=240)
            self.plt.close(fig_amp)
            self.assertTrue(os.path.exists(amplitude_path), 'Expected trace spectrum artifact.')


if __name__ == '__main__':
    unittest.main()
