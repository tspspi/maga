"""Visualization regression tests for the time-dependent chopped beam."""

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


# Configurable trace point in the (x, y, 0) plane (meters)
TRACE_POINT = (0.0005, 0.002, 0.0)

# Driving modulation frequency for FFT annotations (Hz)
DRIVING_FREQUENCY = 200e6


class TestTimeDependentBeamVisualization(unittest.TestCase):
    """Render a short frame sequence of the modulated electron beam."""

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

    def test_time_dependent_beam_frames(self):
        output_dir = os.path.join(os.path.dirname(__file__), "artifacts", "chopped_beam_frames")
        os.makedirs(output_dir, exist_ok=True)

        modulation_frequency = DRIVING_FREQUENCY
        modulation_period = 1.0 / modulation_frequency

        beam = ChoppedBeam(
            voltage=10000.0,
            current=10e-6, # 0.02,
            modulation_frequency=modulation_frequency,
            start_position=(0.0, 0.0, 0.1),
            propagation_direction=(0.0, 0.0, -1.0),
            length=0.2,
            duty_cycle=0.5,
        )

        grid = RectangularGrid(
            #x_range=(-0.002, 0.002),
            x_range=(-0.01, 0.010),
            y_range=(-0.01, 0.010),
            z_range=(-0.001, 0.001),
            nx=101,
            ny=101,
            nz=11,
            name="chopped_beam_grid",
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
        grid_trace_point = np.array([x[trace_ix], y[trace_iy], z[trace_iz]])  # Snap to nearest grid sample
        trace_point_mm = grid_trace_point * 1e3

        timestamps = np.linspace(0.0, modulation_period*10, num=4*60, endpoint=False)
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

            # Build a continuous centre-line path without overlapping duplicates
            beam_path = np.vstack((geometry_A[0], geometry_B))
            # Remove duplicate consecutive points while preserving order
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
                #solid_capstyle="round",
                #solid_joinstyle="round",
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

            strength_values = []
            if sampled_strength.size > 0:
                strength_values.append(sampled_strength)
            if plane_strength.size > 0:
                strength_values.append(plane_strength.ravel())

            normalizer = None
            if strength_values:
                combined_strength = np.concatenate(strength_values)
                finite_strength = combined_strength[np.isfinite(combined_strength)]
                if finite_strength.size:
                    lower = np.percentile(finite_strength, 5)
                    upper = np.percentile(finite_strength, 99)
                    if lower == upper:
                        lower = finite_strength.min()
                        upper = finite_strength.max()
                    if upper > lower:
                        normalizer = self.colors.Normalize(vmin=lower, vmax=upper)
                        sampled_strength = np.clip(sampled_strength, lower, upper)
                        plane_strength = np.clip(plane_strength, lower, upper)
                    elif upper > 0:
                        normalizer = self.colors.Normalize(vmin=0.0, vmax=upper)

            if normalizer is not None:
                arrow_colors = self.cm.inferno(normalizer(sampled_strength))
                plane_facecolors = self.cm.inferno(normalizer(plane_strength))
            else:
                arrow_colors = "#d62728"
                plane_facecolors = np.empty(plane_strength.shape + (4,))
                plane_facecolors[...] = self.colors.to_rgba("#440154")

            plane_x = coord_x[:, :, central_layer] * 1e3
            plane_y = coord_y[:, :, central_layer] * 1e3
            plane_z = coord_z[:, :, central_layer] * 1e3

            ax3d.plot_surface(
                plane_x,
                plane_y,
                plane_z,
                facecolors=plane_facecolors,
                rstride=1,
                cstride=1,
                antialiased=False,
                shade=False,
                alpha=0.7,
            )

            ax3d.quiver(
                X,
                Y,
                Z,
                U,
                V,
                W,
                length=1.0,
                normalize=True,
                colors=arrow_colors,
                alpha=0.25,
                cmap="inferno",
            )

            if normalizer is not None:
                scalar_map = self.cm.ScalarMappable(norm=normalizer, cmap=self.cm.inferno)
                scalar_map.set_array([])
                colorbar = fig.colorbar(
                    scalar_map,
                    ax=ax3d,
                    shrink=0.65,
                    pad=0.1,
                )
                colorbar.set_label("|B| [T]")

            ax3d.set_title(f"Chopped beam at t = {time_point * 1e9:.2f} ns")
            ax3d.set_xlabel("X [mm]")
            ax3d.set_ylabel("Y [mm]")
            ax3d.set_zlabel("Z [mm]")
            ax3d.legend(loc="upper right")

            ax3d.set_xlim(-10.5, 10.5)
            ax3d.set_ylim(-10.5, 10.5)
            ax3d.set_zlim(-1.5, 1.5)

            trace_times_ns = np.asarray(trace_times) * 1e9
            trace_ax.plot(trace_times_ns, trace_components_x, color="tab:blue", label="B_x")
            trace_ax.plot(trace_times_ns, trace_components_y, color="tab:orange", label="B_y")
            trace_ax.set_xlim(0.0, time_axis_max_ns)
            trace_ax.set_xlabel("Time [ns]")
            trace_ax.set_ylabel("B-field [T]")
            trace_ax.set_title(
                f"Field trace at ({trace_point_mm[0]:.2f}, {trace_point_mm[1]:.2f}, {trace_point_mm[2]:.2f}) mm"
            )
            trace_ax.grid(True, alpha=0.3)
            trace_ax.legend(loc="upper right")

            frame_path = os.path.join(output_dir, f"chopped_beam_frame_{frame_idx:04d}.png")
            fig.tight_layout()
            fig.savefig(frame_path, dpi=220)
            self.plt.close(fig)

            frame_paths.append(frame_path)

        spectrum_path = os.path.join(output_dir, "chopped_beam_frequency_spectrum.png")
        if len(trace_times) > 1:
            trace_times_arr = np.asarray(trace_times)
            trace_x_arr = np.asarray(trace_components_x)
            trace_y_arr = np.asarray(trace_components_y)
            sample_spacing = float(np.mean(np.diff(trace_times_arr)))

            if sample_spacing > 0.0:
                freqs = np.fft.rfftfreq(trace_times_arr.size, d=sample_spacing).astype(float, copy=False)
                spectrum_x = np.abs(np.fft.rfft(trace_x_arr - trace_x_arr.mean())) / trace_times_arr.size
                spectrum_y = np.abs(np.fft.rfft(trace_y_arr - trace_y_arr.mean())) / trace_times_arr.size

                fig_spectrum = self.plt.figure(figsize=(8, 5))
                ax_spec = fig_spectrum.add_subplot(111)
                freq_mhz = freqs * 1e-6
                ax_spec.plot(freq_mhz, spectrum_x, color="tab:blue", label="Bx spectrum")
                ax_spec.plot(freq_mhz, spectrum_y, color="tab:orange", label="By spectrum")

                harmonics = [
                    ("1st harmonic", DRIVING_FREQUENCY, "tab:red"),
                    ("2nd harmonic", 2.0 * DRIVING_FREQUENCY, "tab:purple"),
                    ("3rd harmonic", 3.0 * DRIVING_FREQUENCY, "tab:green"),
                ]
                max_freq_plot = freq_mhz[-1] if freq_mhz.size else DRIVING_FREQUENCY * 1e-6
                for label, harmonic_freq, color in harmonics:
                    if freqs.size and harmonic_freq <= freqs[-1]:
                        harmonic_mhz = harmonic_freq * 1e-6
                        ax_spec.axvline(
                            harmonic_mhz,
                            color=color,
                            linestyle="--",
                            linewidth=1.0,
                            alpha=0.7,
                            label=f"{label} ({harmonic_freq * 1e-6:.1f} MHz)",
                        )
                        idx = int(np.argmin(np.abs(freqs - harmonic_freq)))
                        peak = max(spectrum_x[idx], spectrum_y[idx])
                        ax_spec.plot(harmonic_mhz, peak, marker="o", color=color)

                ax_spec.set_xlabel("Frequency [MHz]")
                ax_spec.set_ylabel("|B| spectrum [T]")
                max_harmonic_mhz = max(harmonic_freq for _, harmonic_freq, _ in harmonics) * 1e-6
                desired_max = max_harmonic_mhz * 1.05
                x_max = min(max_freq_plot, desired_max) if freq_mhz.size else desired_max
                ax_spec.set_xlim(0.0, x_max)
                ax_spec.set_title("Trace frequency spectrum")
                ax_spec.grid(True, alpha=0.3)
                ax_spec.legend(loc="upper right")

                fig_spectrum.tight_layout()
                fig_spectrum.savefig(spectrum_path, dpi=240)
                self.plt.close(fig_spectrum)
            else:
                fig_spectrum = self.plt.figure(figsize=(8, 5))
                ax_spec = fig_spectrum.add_subplot(111)
                ax_spec.text(
                    0.5,
                    0.5,
                    "Invalid sampling interval for spectrum",
                    ha="center",
                    va="center",
                    transform=ax_spec.transAxes,
                )
                ax_spec.axis("off")
                fig_spectrum.tight_layout()
                fig_spectrum.savefig(spectrum_path, dpi=240)
                self.plt.close(fig_spectrum)
        else:
            fig_spectrum = self.plt.figure(figsize=(8, 5))
            ax_spec = fig_spectrum.add_subplot(111)
            ax_spec.text(
                0.5,
                0.5,
                "Insufficient samples for spectrum",
                ha="center",
                va="center",
                transform=ax_spec.transAxes,
            )
            ax_spec.axis("off")
            fig_spectrum.tight_layout()
            fig_spectrum.savefig(spectrum_path, dpi=240)
            self.plt.close(fig_spectrum)

        frame_paths.append(spectrum_path)

        for path in frame_paths:
            self.assertTrue(os.path.exists(path), f"Expected frame missing: {path}")

        self.assertGreater(len(frame_paths), 0)



if __name__ == "__main__":  # pragma: no cover
    unittest.main()
