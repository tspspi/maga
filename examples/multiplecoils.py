"""
    Multiple Coils Torus Configuration Demonstration

    This script demonstrates the MAGA library by computing magnetic fields for
    16 circular coils arranged in a torus configuration. The coils are positioned
    with their centers on a circle with radius 1.5 times the coil radius, and
    their normals oriented tangentially to form a toroidal field configuration.

    The script shows how to:

    - Create multiple circular coils in a torus arrangement
    - Combine multiple geometries for field calculation
    - Perform the same analysis and visualization as in maga_demonstration.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the src directory to Python path
# TODO: Remove before uploading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import the library. Since we are a test we
# handle exceptions here
try:
    from maga import MagneticFieldCalculator, DeviceManager, RectangularGrid
    from maga.geometry import CircularCoil
    print("✓ Successfully imported MAGA library components")
except ImportError as e:
    print(f"✗ Failed to import MAGA components: {e}")
    print("Make sure you have PyOpenCL installed and the MAGA library is properly set up")
    sys.exit(1)


class TorusCoilConfiguration:
    """
        Class to create and manage circular coils
        arranged in a torus configuration.
    """
    
    def __init__(self,
                 coil_radius: float = 0.75,
                 torus_radius: float = None,
                 current: float = 2.0,
                 num_coils: int = 16,
                 num_elements_per_coil: int = 150,
                 name: str = "torus_coils"):
        """
            Initialize torus coil configuration.
            
            Args:
                coil_radius:            Radius of each individual coil
                torus_radius:           Radius of the torus (circle on which coil centers are placed)
                                        If None, uses 1.5 * coil_radius
                current:                Current through each coil
                num_coils:              Number of coils around the torus (default 16)
                num_elements_per_coil:  Number of elements per coil for discretization
                name:                   Name for this configuration
        """
        self.coil_radius = float(coil_radius)
        self.torus_radius = float(torus_radius) if torus_radius is not None else 1.5 * self.coil_radius
        self.current = float(current)
        self.num_coils = int(num_coils)
        self.num_elements_per_coil = int(num_elements_per_coil)
        self.name = name
        
        # Create the coils after initializing our empty geometry list
        self.coils = []
        self._create_torus_coils()
        
        print(f"Created torus configuration: {self.num_coils} coils, "
              f"coil radius={self.coil_radius}, torus radius={self.torus_radius}")
    
    def _create_torus_coils(self):
        """Create coils arranged in torus configuration."""
        
        for i in range(self.num_coils):
            # Angle for this coil around the torus
            phi = 2 * np.pi * i / self.num_coils
            
            # Center position on the torus circle
            center_x = self.torus_radius * np.cos(phi)
            center_y = self.torus_radius * np.sin(phi)
            center_z = 0.0
            center = (center_x, center_y, center_z)
            
            # Normal vector tangent to the torus circle (perpendicular to radial direction)
            # This creates the toroidal field configuration
            normal_x = -np.sin(phi)  # Tangent to the circle
            normal_y = np.cos(phi)
            normal_z = 0.0
            normal = (normal_x, normal_y, normal_z)
            
            # Create the coil based on CircularCoil geometry (and assign systematic names)
            coil = CircularCoil(
                center=center,
                radius=self.coil_radius,
                current=self.current,
                num_elements=self.num_elements_per_coil,
                normal_vector=normal,
                name=f"{self.name}_coil_{i:02d}"
            )
            
            self.coils.append(coil)
    
    def get_geometry(self):
        """
            Get combined geometry for all coils. We assume here that the geometry
            is contains in CPU buffers, not GPU buffers that this moment. Then we
            simply vertically stack the transformation matrices and currents.
            
            Returns:
                Tuple of (A, B, I) arrays combining all coils
        """
        A_all = []
        B_all = []
        I_all = []
        
        for coil in self.coils:
            A, B, I = coil.get_geometry()
            A_all.append(A)
            B_all.append(B)
            I_all.append(I)
        
        # Combine all arrays
        A_combined = np.vstack(A_all)
        B_combined = np.vstack(B_all)
        I_combined = np.hstack(I_all)
        
        return A_combined, B_combined, I_combined
    
    def get_bounds(self):
        """Get spatial bounds of the entire torus configuration."""
        max_extent = self.torus_radius + self.coil_radius
        return {
            'x': (-max_extent, max_extent),
            'y': (-max_extent, max_extent),
            'z': (-self.coil_radius, self.coil_radius)
        }
    
    def get_total_current(self):
        """Get total current in the system."""
        return self.num_coils * self.current
    
    def get_total_elements(self):
        """Get total number of current elements."""
        return self.num_coils * self.num_elements_per_coil
    
    def __str__(self):
        return (f"TorusCoilConfiguration('{self.name}', {self.num_coils} coils, "
                f"coil_radius={self.coil_radius}, torus_radius={self.torus_radius}, "
                f"current={self.current}A each, total_elements={self.get_total_elements()})")


def demonstrate_torus_coils():
    """
        Magnetic field calculation for torus coil configuration.
    """
    print("\n=== Torus Coil Configuration Demonstration ===")
    
    # Create torus coil configuration using our composite helper class
    torus_coils = TorusCoilConfiguration(
        coil_radius=0.75,
        torus_radius=1.125,
        current=2.0,
        num_coils=16,
        num_elements_per_coil=100,
        name="demo_torus"
    )
    
    print(f"Created torus configuration: {torus_coils}")
    print(f"Total current elements: {torus_coils.get_total_elements()}")
    print(f"Total current: {torus_coils.get_total_current():.1f}A")
    
    # Get bounds and create calculation grid
    bounds = torus_coils.get_bounds()
    print(f"Configuration bounds: x={bounds['x']}, y={bounds['y']}, z={bounds['z']}")
    
    # Create grid that encompasses the torus with some margin
    margin = 0.5
    grid = RectangularGrid(
        x_range=(bounds['x'][0] - margin, bounds['x'][1] + margin),
        y_range=(bounds['y'][0] - margin, bounds['y'][1] + margin),
        z_range=(bounds['z'][0] - margin, bounds['z'][1] + margin),
        nx=32, ny=32, nz=32,
        name="torus_field_grid"
    )
    
    print(f"Created calculation grid: {grid.num_points} points")
    
    # Set up calculator and compute field
    try:
        calculator = MagneticFieldCalculator()
        print(f"Using device: {calculator.get_device_info()}")
        
        # Get geometry and calculate field
        A, B, I = torus_coils.get_geometry()
        print(f"Total geometry elements: {len(A)} current segments")
        
        result = calculator.calculate_magnetic_field(A, B, I, grid)
        
        print(f"Calculation completed in {result.calculation_time:.3f} seconds")
        print(f"Field range: {result.min_field:.2e} to {result.max_field:.2e} T")
        
        # Calculate field statistics
        field_std = np.std(result.field_magnitude)
        field_mean = np.mean(result.field_magnitude)
        print(f"Field statistics: mean={field_mean:.2e} T, std={field_std:.2e} T")
        
        return result, torus_coils, grid
        
    except Exception as e:
        print(f"Calculation failed: {e}")
        return None, None, None


def create_field_visualization(result, title, grid):
    """Create field visualization"""
    if result is None:
        return
        
    try:
        import matplotlib.pyplot as plt
        
        # Get the magnetic field result - should now be 4D array (nx, ny, nz, 3)
        field = result.magnetic_field
        
        # If field is flat, skip visualization
        if field.ndim != 4:
            print(f"Skipping visualization for {title} - field array has wrong shape: {field.shape}")
            return
            
        nx, ny, nz = field.shape[:3]
        
        # Get grid coordinates for axis labels
        if hasattr(grid, 'x_range') and hasattr(grid, 'y_range') and hasattr(grid, 'z_range'):
            x = np.linspace(grid.x_range[0], grid.x_range[1], nx)
            y = np.linspace(grid.y_range[0], grid.y_range[1], ny)
            z = np.linspace(grid.z_range[0], grid.z_range[1], nz)
        else:
            x = np.arange(nx)
            y = np.arange(ny)
            z = np.arange(nz)
        
        # Create 3x3 subplot visualization like the notebook
        fig = plt.figure(figsize=(15, 12))
        
        # Row 1: Front slices (z=0)
        ax1 = fig.add_subplot(331)
        color = np.log(np.hypot(field[0,:,:,0], field[0,:,:,1]) + 1e-12)
        try:
            ax1.streamplot(x, y, field[0,:,:,0], field[0,:,:,1], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            # Fallback if streamplot fails
            ax1.contourf(x, y, np.hypot(field[0,:,:,0], field[0,:,:,1]), cmap='inferno')
        ax1.set_title(f'{title} - XY (front)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        ax2 = fig.add_subplot(332)
        color = np.log(np.hypot(field[:,0,:,0], field[:,0,:,2]) + 1e-12)
        try:
            ax2.streamplot(x, z, field[:,0,:,0], field[:,0,:,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax2.contourf(x, z, np.hypot(field[:,0,:,0], field[:,0,:,2]), cmap='inferno')
        ax2.set_title(f'{title} - XZ (front)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        
        ax3 = fig.add_subplot(333)
        color = np.log(np.hypot(field[:,:,0,1], field[:,:,0,2]) + 1e-12)
        try:
            ax3.streamplot(y, z, field[:,:,0,1], field[:,:,0,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax3.contourf(y, z, np.hypot(field[:,:,0,1], field[:,:,0,2]), cmap='inferno')
        ax3.set_title(f'{title} - YZ (front)')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        
        # Row 2: Middle slices
        mid_z, mid_y, mid_x = nz//2, ny//2, nx//2
        
        ax4 = fig.add_subplot(334)
        color = np.log(np.hypot(field[mid_z,:,:,0], field[mid_z,:,:,1]) + 1e-12)
        try:
            ax4.streamplot(x, y, field[mid_z,:,:,0], field[mid_z,:,:,1], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax4.contourf(x, y, np.hypot(field[mid_z,:,:,0], field[mid_z,:,:,1]), cmap='inferno')
        ax4.set_title(f'{title} - XY (middle)')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        
        ax5 = fig.add_subplot(335)
        color = np.log(np.hypot(field[:,mid_y,:,0], field[:,mid_y,:,2]) + 1e-12)
        try:
            ax5.streamplot(x, z, field[:,mid_y,:,0], field[:,mid_y,:,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax5.contourf(x, z, np.hypot(field[:,mid_y,:,0], field[:,mid_y,:,2]), cmap='inferno')
        ax5.set_title(f'{title} - XZ (middle)')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Z')
        
        ax6 = fig.add_subplot(336)
        color = np.log(np.hypot(field[:,:,mid_x,1], field[:,:,mid_x,2]) + 1e-12)
        try:
            ax6.streamplot(y, z, field[:,:,mid_x,1], field[:,:,mid_x,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax6.contourf(y, z, np.hypot(field[:,:,mid_x,1], field[:,:,mid_x,2]), cmap='inferno')
        ax6.set_title(f'{title} - YZ (middle)')
        ax6.set_xlabel('Y')
        ax6.set_ylabel('Z')
        
        # Row 3: Back slices
        ax7 = fig.add_subplot(337)
        color = np.log(np.hypot(field[nz-1,:,:,0], field[nz-1,:,:,1]) + 1e-12)
        try:
            ax7.streamplot(x, y, field[nz-1,:,:,0], field[nz-1,:,:,1], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax7.contourf(x, y, np.hypot(field[nz-1,:,:,0], field[nz-1,:,:,1]), cmap='inferno')
        ax7.set_title(f'{title} - XY (back)')
        ax7.set_xlabel('X')
        ax7.set_ylabel('Y')
        
        ax8 = fig.add_subplot(338)
        color = np.log(np.hypot(field[:,ny-1,:,0], field[:,ny-1,:,2]) + 1e-12)
        try:
            ax8.streamplot(x, z, field[:,ny-1,:,0], field[:,ny-1,:,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax8.contourf(x, z, np.hypot(field[:,ny-1,:,0], field[:,ny-1,:,2]), cmap='inferno')
        ax8.set_title(f'{title} - XZ (back)')
        ax8.set_xlabel('X')
        ax8.set_ylabel('Z')
        
        ax9 = fig.add_subplot(339)
        color = np.log(np.hypot(field[:,:,nx-1,1], field[:,:,nx-1,2]) + 1e-12)
        try:
            ax9.streamplot(y, z, field[:,:,nx-1,1], field[:,:,nx-1,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax9.contourf(y, z, np.hypot(field[:,:,nx-1,1], field[:,:,nx-1,2]), cmap='inferno')
        ax9.set_title(f'{title} - YZ (back)')
        ax9.set_xlabel('Y')
        ax9.set_ylabel('Z')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed for {title}: {e}")


def create_3d_field_visualization(result, title, grid, geometry_data=None):
    """Create 3D quiver plot visualization"""
    if result is None:
        return
        
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d import Axes3D
        
        # Get the magnetic field result - should be 4D array (nx, ny, nz, 3)
        field = result.magnetic_field
        
        # If field is flat, skip visualization
        if field.ndim != 4:
            print(f"Skipping 3D visualization for {title} - field array has wrong shape: {field.shape}")
            return
            
        nx, ny, nz = field.shape[:3]
        
        # Get grid coordinates
        if hasattr(grid, 'x_range') and hasattr(grid, 'y_range') and hasattr(grid, 'z_range'):
            x = np.linspace(grid.x_range[0], grid.x_range[1], nx)
            y = np.linspace(grid.y_range[0], grid.y_range[1], ny)
            z = np.linspace(grid.z_range[0], grid.z_range[1], nz)
        else:
            x = np.arange(nx)
            y = np.arange(ny)
            z = np.arange(nz)

        # For clarity, subsample the grid (otherwise there will be too many arrows)
        step = max(1, min(nx, ny, nz) // 8)  # Adaptive step size
        Z_mesh, Y_mesh, X_mesh = np.meshgrid(z, y, x, indexing='ij')
        X_sub = X_mesh[::step, ::step, ::step]
        Y_sub = Y_mesh[::step, ::step, ::step]
        Z_sub = Z_mesh[::step, ::step, ::step]
        Bx_sub = field[::step, ::step, ::step, 0]
        By_sub = field[::step, ::step, ::step, 1]
        Bz_sub = field[::step, ::step, ::step, 2]

        # Calculate field strength for coloring
        field_strength = np.sqrt(Bx_sub**2 + By_sub**2 + Bz_sub**2)
        
        # Create a 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D vector field using quiver
        # Note: adjust length and normalize for better visualization
        q = ax.quiver(X_sub, Y_sub, Z_sub, Bx_sub, By_sub, Bz_sub,
                     length=0.3, normalize=True, cmap=cm.viridis, alpha=0.7)

        # Color the arrows by field strength
        if field_strength.size > 0 and field_strength.max() > field_strength.min():
            norm = plt.Normalize(vmin=field_strength.min(), vmax=field_strength.max())
            colors = cm.viridis(norm(field_strength.flatten()))
            q.set_facecolor(colors)

        # Plot geometry objects (current-carrying conductors) if provided
        if geometry_data is not None:
            A, B, I = geometry_data
            if A.size > 0 and B.size > 0:
                # Ensure A and B are 2D arrays
                if A.ndim == 1:
                    A = A.reshape(1, -1)
                if B.ndim == 1:
                    B = B.reshape(1, -1)
                    
                # Sample a subset of segments for clearer visualization
                num_segments = len(A)
                sample_step = max(1, num_segments // 200)  # Show at most 200 segments
                
                for i in range(0, num_segments, sample_step):
                    if A.shape[1] >= 3 and B.shape[1] >= 3:  # Ensure we have x,y,z coordinates
                        xs = [A[i, 0], B[i, 0]]
                        ys = [A[i, 1], B[i, 1]]
                        zs = [A[i, 2], B[i, 2]]
                        # Use thicker lines for higher currents
                        linewidth = max(1.0, min(3.0, abs(I[i]) * 1.5)) if I.size > i else 2.0
                        ax.plot(xs, ys, zs, color='red', linewidth=linewidth, alpha=0.8)

        ax.set_title(f'{title} - 3D Magnetic Field Vectors')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Add colorbar for field strength
        if field_strength.size > 0 and field_strength.max() > field_strength.min():
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cm.viridis), ax=ax, shrink=0.8)
            cbar.set_label('Field Strength (T)')

        # Set equal aspect ratio for better visualization
        max_range = max(abs(X_sub.max() - X_sub.min()),
                       abs(Y_sub.max() - Y_sub.min()),
                       abs(Z_sub.max() - Z_sub.min())) / 2.0
        mid_x = (X_sub.max() + X_sub.min()) * 0.5
        mid_y = (Y_sub.max() + Y_sub.min()) * 0.5
        mid_z = (Z_sub.max() + Z_sub.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"3D visualization failed for {title}: {e}")


def create_torus_geometry_visualization(torus_coils):
    """Create a visualization of the torus coil geometry itself."""
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each coil
        for i, coil in enumerate(torus_coils.coils):
            # Get coil geometry
            A, B, I = coil.get_geometry()
            
            # Color based on coil index
            color = plt.cm.tab20(i / len(torus_coils.coils))
            
            # Plot the coil segments
            for j in range(len(A)):
                xs = [A[j, 0], B[j, 0]]
                ys = [A[j, 1], B[j, 1]]
                zs = [A[j, 2], B[j, 2]]
                ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=0.8)
            
            # Mark the coil center
            center = coil.center
            ax.scatter(center[0], center[1], center[2], color=color, s=50, alpha=0.9)
        
        # Set equal aspect ratio
        bounds = torus_coils.get_bounds()
        max_range = max(bounds['x'][1] - bounds['x'][0],
                       bounds['y'][1] - bounds['y'][0],
                       bounds['z'][1] - bounds['z'][0]) / 2.0
        mid_x = (bounds['x'][0] + bounds['x'][1]) * 0.5
        mid_y = (bounds['y'][0] + bounds['y'][1]) * 0.5
        mid_z = (bounds['z'][0] + bounds['z'][1]) * 0.5
        
        ax.set_xlim(mid_x - max_range * 1.1, mid_x + max_range * 1.1)
        ax.set_ylim(mid_y - max_range * 1.1, mid_y + max_range * 1.1)
        ax.set_zlim(mid_z - max_range * 1.1, mid_z + max_range * 1.1)
        
        ax.set_title('Torus Coil Configuration Geometry')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Geometry visualization failed: {e}")


def main():
    print("Torus Coil Configuration Demonstration")
    print("=" * 38)
    
    # Check for matplotlib
    try:
        import matplotlib.pyplot as plt
        show_plots = True
    except ImportError:
        print("Matplotlib not available - skipping visualizations")
        show_plots = False
    
    # Run demonstration
    result, torus_coils, grid = demonstrate_torus_coils()
    
    # Summary
    print("\n=== Demonstration Summary ===")
    if result is not None:
        print(f"Calculation time: {result.calculation_time:.3f} seconds")
        print(f"Field range: {result.min_field:.2e} to {result.max_field:.2e} T")
        print(f"Total current elements: {torus_coils.get_total_elements()}")
        print(f"Total current: {torus_coils.get_total_current():.1f}A")
    else:
        print("✗ Calculation failed")
        return
    
    # Create visualizations if possible
    if show_plots and result is not None:
        try:
            print("\n=== Creating Visualizations ===")
            
            # 1. Show the coil geometry itself
            print("Creating torus geometry visualization...")
            create_torus_geometry_visualization(torus_coils)
            
            # 2. Create field visualization
            print("Creating magnetic field visualization...")
            create_field_visualization(result, "Torus Coil Configuration", grid)
            
            # 3. Create 3D field visualization
            print("Creating 3D magnetic field visualization...")
            # Use smaller grid for 3D visualization
            grid_3d = RectangularGrid(
                x_range=grid.x_range, y_range=grid.y_range, z_range=grid.z_range,
                nx=16, ny=16, nz=16, name="3d_vis_grid"
            )
            
            # Recalculate with smaller grid for 3D visualization
            calculator = MagneticFieldCalculator()
            A, B, I = torus_coils.get_geometry()
            result_3d = calculator.calculate_magnetic_field(A, B, I, grid_3d)
            
            geometry_data = torus_coils.get_geometry()
            create_3d_field_visualization(result_3d, "Torus Coil Configuration", grid_3d, geometry_data)
            
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Field analysis
    if result is not None:
        print("\n=== Field Analysis ===")
        field_mag = result.field_magnitude
        
        print(f"Field statistics:")
        print(f"  Mean field: {np.mean(field_mag):.2e} T")
        print(f"  Std deviation: {np.std(field_mag):.2e} T")
        print(f"  Max field: {np.max(field_mag):.2e} T")
        print(f"  Min field: {np.min(field_mag):.2e} T")
        
        # Find field at center
        if hasattr(grid, 'nx') and hasattr(grid, 'ny') and hasattr(grid, 'nz'):
            center_idx = (grid.nx//2, grid.ny//2, grid.nz//2)
            if result.magnetic_field.ndim == 4:
                center_field = result.magnetic_field[center_idx[0], center_idx[1], center_idx[2], :]
                center_magnitude = np.linalg.norm(center_field)
                print(f"  Field at center: {center_magnitude:.2e} T")
                print(f"  Center field vector: [{center_field[0]:.2e}, {center_field[1]:.2e}, {center_field[2]:.2e}] T")


if __name__ == "__main__":
    main()
