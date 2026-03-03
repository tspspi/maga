"""
    MAGA Library Example

    This script demonstrates some of the key features of
    the MAGA (Magnetic Analysis with GPU Acceleration) library
    by computing magnetic fields for various geometries using
    GPU-accelerated calculations.

    Examples shown:
        - Single circular coil
        - Helmholtz coil pair
        - Anti-Helmholtz coil pair
        - Rectangular coil
        
    The script shows how to:
        - Create different geometry types
        - Set up calculation grids
        - Perform GPU-accelerated field calculations
        - Fetch and process results
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add the src directory to Python path
# TODO: Remove before release
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the library.
# Since we are an example we catch import errors.
try:
    from maga import MagneticFieldCalculator, DeviceManager, RectangularGrid
    from maga.geometry import (CircularCoil, RectangularCoil, 
                               HelmholtzCoils, AntiHelmholtzCoils)
    print("✓ Successfully imported MAGA library components")
except ImportError as e:
    print(f"✗ Failed to import MAGA components: {e}")
    print("Make sure you have PyOpenCL installed and the MAGA library is properly set up")
    sys.exit(1)


def demonstrate_single_coil():
    """
        Magnetic field calculation for a single circular coil.

        The coil is centered at (0,0,0) and resides in the XY plane.
    """

    print("\n=== Single Circular Coil Demonstration ===")
    
    # Create a circular coil
    coil = CircularCoil(
        center=(0.0, 0.0, 0.0),
        radius=1.0,
        current=10.0,
        num_elements=50,
        name="demo_coil"
    )
    
    print(f"Created coil: {coil}")
    print(f"Coil area: {coil.get_area():.3f} m²")
    print(f"Magnetic dipole moment: {coil.get_magnetic_dipole_moment()}")
    
    # Create calculation grid. The grid extens over 6.4m in all direction
    # and has 32 x 32 x 32 points
    grid = RectangularGrid(
        x_range=(-3.2, 3.2), y_range=(-3.2, 3.2), z_range=(-3.2, 3.2),
        nx=32, ny=32, nz=32,
        name="coil_field_grid"
    )
    
    print(f"Created calculation grid: {grid.num_points} points")
    
    # Set up calculator and compute field
    try:
        calculator = MagneticFieldCalculator()
        print(f"Using device: {calculator.get_device_info()}")
        
        # Get geometry and calculate field
        A, B, I = coil.get_geometry()
        result = calculator.calculate_magnetic_field(A, B, I, grid)
        
        print(f"Calculation completed in {result.calculation_time:.3f} seconds")
        print(f"Field range: {result.min_field:.2e} to {result.max_field:.2e} T")
        
        return result
        
    except Exception as e:
        print(f"Calculation failed: {e}")
        return None


def demonstrate_helmholtz_coils():
    """
        A Helmholtz coil pair.

        Here we use a geomtry generator HelmholtzCoils that builds on
        the circular coils. It should demonstrate how to build composite
        geometry generators
    """
    print("\n=== Helmholtz Coils Demonstration ===")
    
    # Create Helmholtz coil pair
    helmholtz = HelmholtzCoils(
        center=(0.0, 0.0, 0.0),
        radius=2.0,
        current=5.0,
        num_elements_per_coil=40,
        name="helmholtz_pair"
    )
    
    print(f"Created Helmholtz coils: {helmholtz}")
    print(f"Optimal separation: {helmholtz.get_optimal_separation():.3f} m")
    
    uniformity_region = helmholtz.get_field_uniformity_region()
    print(f"Uniformity region radius: ±{uniformity_region['radius_x']:.3f} m")
    
    # Create grid well outside the 2.0m radius coils to avoid singularities
    grid = RectangularGrid(
        x_range=(-6.0, 6.0), y_range=(-6.0, 6.0), z_range=(-6.0, 6.0),
        nx=64, ny=64, nz=64,
        name="helmholtz_grid"
    )
    
    try:
        calculator = MagneticFieldCalculator()
        A, B, I = helmholtz.get_geometry()
        result = calculator.calculate_magnetic_field(A, B, I, grid)
        
        # TODO: Check Field Uniformity only in a region in the center ...
        print(f"Calculation completed in {result.calculation_time:.3f} seconds")
        print(f"Field uniformity: {result.min_field:.2e} to {result.max_field:.2e} T")
        
        # Calculate field uniformity
        field_std = np.std(result.field_magnitude)
        field_mean = np.mean(result.field_magnitude)
        uniformity_percent = (field_std / field_mean) * 100
        print(f"Field uniformity: {uniformity_percent:.2f}% variation")
        
        return result
        
    except Exception as e:
        print(f"Helmholtz calculation failed: {e}")
        return None


def demonstrate_anti_helmholtz_coils():
    """
        Anti-Helmholtz coils

        This is a configuratoin that is often used to generate gradient
        fields (like in magneto optical traps)
    """
    print("\n=== Anti-Helmholtz Coils Demonstration ===")
    
    # Create Anti-Helmholtz coil pair
    # Again we utilize a composite geometry generator

    anti_helmholtz = AntiHelmholtzCoils(
        center=(0.0, 0.0, 0.0),
        radius=0.3,
        current=8.0,
        separation=0.4,
        num_elements_per_coil=30,
        name="anti_helmholtz_pair"
    )
    
    print(f"Created Anti-Helmholtz coils: {anti_helmholtz}")
    gradient = anti_helmholtz.get_gradient_strength()
    print(f"Estimated gradient strength: {gradient:.2e} T/m")
    
    # Example trap frequencies for Rubidium atoms

    mass_rb = 1.45e-25  # kg (Rb-87)
    mu_b = 9.274e-24    # Bohr magneton in J/T
    trap_freqs = anti_helmholtz.get_trap_frequencies(mass_rb, mu_b)
    print(f"Trap frequencies (Rb atoms): radial={trap_freqs['radial_frequency']:.0f} Hz")
    
    grid = RectangularGrid(
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-1.0, 1.0),
        nx=32, ny=32, nz=32,
        name="gradient_grid"
    )
    
    try:
        calculator = MagneticFieldCalculator()
        A, B, I = anti_helmholtz.get_geometry()
        result = calculator.calculate_magnetic_field(A, B, I, grid)
        
        print(f"Calculation completed in {result.calculation_time:.3f} seconds")
        print(f"Field range: {result.min_field:.2e} to {result.max_field:.2e} T")
        
        return result
        
    except Exception as e:
        print(f"Anti-Helmholtz calculation failed: {e}")
        return None


def demonstrate_rectangular_coil():
    """
        Rectangular coil geometry
    """

    print("\n=== Rectangular Coil Demonstration ===")
    
    # Create a rectangular coil
    rect_coil = RectangularCoil(
        center=(0.0, 0.0, 0.0),
        width=1.5,
        height=0.8,
        current=3.0,
        num_elements=80,
        name="rectangular_coil"
    )
    
    print(f"Created rectangular coil: {rect_coil}")
    print(f"Aspect ratio: {rect_coil.get_aspect_ratio():.2f}")
    print(f"Enclosed area: {rect_coil.get_area():.3f} m²")
    print(f"Perimeter: {rect_coil.get_perimeter():.3f} m")
    
    # Create calculation grid that avoids the 1.5x0.8 rectangular coil
    grid = RectangularGrid(
        x_range=(-3.0, 3.0), y_range=(-2.0, 2.0), z_range=(-2.0, 2.0),
        nx=32, ny=32, nz=32,
        name="rect_coil_grid"
    )
    
    try:
        calculator = MagneticFieldCalculator()
        A, B, I = rect_coil.get_geometry()
        result = calculator.calculate_magnetic_field(A, B, I, grid)
        
        print(f"Calculation completed in {result.calculation_time:.3f} seconds")
        print(f"Field range: {result.min_field:.2e} to {result.max_field:.2e} T")
        
        return result
        
    except Exception as e:
        print(f"Rectangular coil calculation failed: {e}")
        return None


def create_field_visualization(result, title, grid):
    """
        Create field visualization

        Here we plot slices through the 3D space and show field lines.
    """
    if result is None:
        return
        
    try:
        import matplotlib.pyplot as plt
        
        # Get the magnetic field result - should now be 4D array (nx, ny, nz, 3)
        # The last 3 components are the magnetic field in x,y,z direction.
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
        
        ax2 = fig.add_subplot(332)
        color = np.log(np.hypot(field[:,0,:,0], field[:,0,:,2]) + 1e-12)
        try:
            ax2.streamplot(x, z, field[:,0,:,0], field[:,0,:,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax2.contourf(x, z, np.hypot(field[:,0,:,0], field[:,0,:,2]), cmap='inferno')
        ax2.set_title(f'{title} - XZ (front)')
        
        ax3 = fig.add_subplot(333)
        color = np.log(np.hypot(field[:,:,0,1], field[:,:,0,2]) + 1e-12)
        try:
            ax3.streamplot(y, z, field[:,:,0,1], field[:,:,0,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax3.contourf(y, z, np.hypot(field[:,:,0,1], field[:,:,0,2]), cmap='inferno')
        ax3.set_title(f'{title} - YZ (front)')
        
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
        
        ax5 = fig.add_subplot(335)
        color = np.log(np.hypot(field[:,mid_y,:,0], field[:,mid_y,:,2]) + 1e-12)
        try:
            ax5.streamplot(x, z, field[:,mid_y,:,0], field[:,mid_y,:,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax5.contourf(x, z, np.hypot(field[:,mid_y,:,0], field[:,mid_y,:,2]), cmap='inferno')
        ax5.set_title(f'{title} - XZ (middle)')
        
        ax6 = fig.add_subplot(336)
        color = np.log(np.hypot(field[:,:,mid_x,1], field[:,:,mid_x,2]) + 1e-12)
        try:
            ax6.streamplot(y, z, field[:,:,mid_x,1], field[:,:,mid_x,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax6.contourf(y, z, np.hypot(field[:,:,mid_x,1], field[:,:,mid_x,2]), cmap='inferno')
        ax6.set_title(f'{title} - YZ (middle)')
        
        # Row 3: Back slices
        ax7 = fig.add_subplot(337)
        color = np.log(np.hypot(field[nz-1,:,:,0], field[nz-1,:,:,1]) + 1e-12)
        try:
            ax7.streamplot(x, y, field[nz-1,:,:,0], field[nz-1,:,:,1], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax7.contourf(x, y, np.hypot(field[nz-1,:,:,0], field[nz-1,:,:,1]), cmap='inferno')
        ax7.set_title(f'{title} - XY (back)')
        
        ax8 = fig.add_subplot(338)
        color = np.log(np.hypot(field[:,ny-1,:,0], field[:,ny-1,:,2]) + 1e-12)
        try:
            ax8.streamplot(x, z, field[:,ny-1,:,0], field[:,ny-1,:,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax8.contourf(x, z, np.hypot(field[:,ny-1,:,0], field[:,ny-1,:,2]), cmap='inferno')
        ax8.set_title(f'{title} - XZ (back)')
        
        ax9 = fig.add_subplot(339)
        color = np.log(np.hypot(field[:,:,nx-1,1], field[:,:,nx-1,2]) + 1e-12)
        try:
            ax9.streamplot(y, z, field[:,:,nx-1,1], field[:,:,nx-1,2], color=color,
                          linewidth=0.5, cmap='inferno', density=2, arrowstyle='->', arrowsize=1)
        except:
            ax9.contourf(y, z, np.hypot(field[:,:,nx-1,1], field[:,:,nx-1,2]), cmap='inferno')
        ax9.set_title(f'{title} - YZ (back)')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Visualization failed for {title}: {e}")


def create_3d_field_visualization(result, title, grid, geometry_data=None):
    """
        Create 3D quiver plot visualization
    """
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
                    
                for i in range(len(A)):
                    if A.shape[1] >= 3 and B.shape[1] >= 3:  # Ensure we have x,y,z coordinates
                        xs = [A[i, 0], B[i, 0]]
                        ys = [A[i, 1], B[i, 1]]
                        zs = [A[i, 2], B[i, 2]]
                        # Use thicker lines for higher currents
                        linewidth = max(1.0, min(4.0, abs(I[i]) * 2.0)) if I.size > i else 2.0
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


def main():
    print("Library Demonstration")
    print("=" * 21)
    
    # Import matplotlib - if installed
    try:
        import matplotlib.pyplot as plt
        show_plots = True
    except ImportError:
        print("Matplotlib not available - skipping visualizations")
        show_plots = False
    
    # Run demonstrations
    results = {}
    
    results['single_coil'] = demonstrate_single_coil()
    results['helmholtz'] = demonstrate_helmholtz_coils()
    results['anti_helmholtz'] = demonstrate_anti_helmholtz_coils()
    results['rectangular'] = demonstrate_rectangular_coil()
        
    # Summary
    print("\n=== Demonstration Summary ===")
    successful = sum(1 for r in results.values() if r is not None)
    total = len([r for r in results.values() if not isinstance(r, list)])
        
    print(f"Successfully completed {successful}/{total} calculations")
    
    # Create visualizations if possible
    if show_plots:
        try:
            # Get the grids for visualization
            if results['single_coil'] is not None:
                single_grid = RectangularGrid(
                    x_range=(-3.2, 3.2), y_range=(-3.2, 3.2), z_range=(-3.2, 3.2),
                    nx=32, ny=32, nz=32, name="vis_grid"
                )
                create_field_visualization(results['single_coil'], "Single Circular Coil", single_grid)
                
            if results['helmholtz'] is not None:
                helm_grid = RectangularGrid(
                    x_range=(-6.0, 6.0), y_range=(-6.0, 6.0), z_range=(-6.0, 6.0),
                    nx=64, ny=64, nz=64, name="vis_grid"
                )
                create_field_visualization(results['helmholtz'], "Helmholtz Coils", helm_grid)
                
            if results['anti_helmholtz'] is not None:
                anti_helm_grid = RectangularGrid(
                    x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-1.0, 1.0),
                    nx=32, ny=32, nz=32, name="vis_grid"
                )
                create_field_visualization(results['anti_helmholtz'], "Anti-Helmholtz Coils", anti_helm_grid)
                
            if results['rectangular'] is not None:
                rect_grid = RectangularGrid(
                    x_range=(-3.0, 3.0), y_range=(-2.0, 2.0), z_range=(-2.0, 2.0),
                    nx=32, ny=32, nz=32, name="vis_grid"
                )
                create_field_visualization(results['rectangular'], "Rectangular Coil", rect_grid)
                
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # Create 3D quiver visualizations with geometry
    if show_plots:
        try:
            print("\n=== Creating 3D Field Visualizations ===")
            
            if results['single_coil'] is not None:
                print("Creating 3D visualization for Single Circular Coil...")
                coil = CircularCoil(center=(0.0, 0.0, 0.0), radius=1.0, current=10.0, num_elements=50, name="demo_coil")
                single_grid_3d = RectangularGrid(x_range=(-3.2, 3.2), y_range=(-3.2, 3.2), z_range=(-3.2, 3.2), nx=16, ny=16, nz=16, name="3d_vis_grid")
                geometry_data = coil.get_geometry()
                create_3d_field_visualization(results['single_coil'], "Single Circular Coil", single_grid_3d, geometry_data)
            
            if results['helmholtz'] is not None:
                print("Creating 3D visualization for Helmholtz Coils...")
                helmholtz = HelmholtzCoils(center=(0.0, 0.0, 0.0), radius=2.0, current=5.0, num_elements_per_coil=40, name="helmholtz_pair")
                helm_grid_3d = RectangularGrid(x_range=(-4.0, 4.0), y_range=(-4.0, 4.0), z_range=(-4.0, 4.0), nx=12, ny=12, nz=12, name="3d_vis_grid")
                geometry_data = helmholtz.get_geometry()
                create_3d_field_visualization(results['helmholtz'], "Helmholtz Coils", helm_grid_3d, geometry_data)
            
            if results['anti_helmholtz'] is not None:
                print("Creating 3D visualization for Anti-Helmholtz Coils...")
                anti_helmholtz = AntiHelmholtzCoils(center=(0.0, 0.0, 0.0), radius=0.3, current=8.0, separation=0.4, num_elements_per_coil=30, name="anti_helmholtz_pair")
                anti_helm_grid_3d = RectangularGrid(x_range=(-0.8, 0.8), y_range=(-0.8, 0.8), z_range=(-0.8, 0.8), nx=12, ny=12, nz=12, name="3d_vis_grid")
                geometry_data = anti_helmholtz.get_geometry()
                create_3d_field_visualization(results['anti_helmholtz'], "Anti-Helmholtz Coils", anti_helm_grid_3d, geometry_data)
            
            if results['rectangular'] is not None:
                print("Creating 3D visualization for Rectangular Coil...")
                rect_coil = RectangularCoil(center=(0.0, 0.0, 0.0), width=1.5, height=0.8, current=3.0, num_elements=80, name="rectangular_coil")
                rect_grid_3d = RectangularGrid(x_range=(-2.0, 2.0), y_range=(-1.5, 1.5), z_range=(-1.5, 1.5), nx=12, ny=12, nz=12, name="3d_vis_grid")
                geometry_data = rect_coil.get_geometry()
                create_3d_field_visualization(results['rectangular'], "Rectangular Coil", rect_grid_3d, geometry_data)                
        except Exception as e:
            print(f"3D Visualization failed: {e}")
    
    # Summary
    calc_times = []
    for key, result in results.items():
        if result is not None:
            calc_times.append(result.calculation_time)
    
    if calc_times:
        print(f"\nPerformance Summary:")
        print(f"Average calculation time: {np.mean(calc_times):.3f} seconds")
        print(f"Total computation time: {sum(calc_times):.3f} seconds")
        print(f"Fastest calculation: {min(calc_times):.3f} seconds")
        print(f"Slowest calculation: {max(calc_times):.3f} seconds")

if __name__ == "__main__":
    main()
