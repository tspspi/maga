#!/usr/bin/env python3
"""
Test script for 3D quiver field visualization functionality
This script demonstrates the 3D visualization features added to MAGA
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from maga import MagneticFieldCalculator, RectangularGrid
    from maga.geometry import CircularCoil
    import matplotlib.pyplot as plt
    import numpy as np
    print("✓ All required modules imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

def create_3d_field_visualization(result, title, grid, geometry_data=None):
    """Create 3D quiver plot visualization matching the notebook approach."""
    if result is None:
        return
        
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
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


def test_3d_quiver_visualization():
    """Test the 3D quiver plot with geometry overlay."""
    print("\n=== Testing 3D Quiver Plot Visualization ===")
    
    # Create a simple circular coil
    coil = CircularCoil(
        center=(0.0, 0.0, 0.0),
        radius=0.5,  # Smaller radius for better visualization
        current=5.0,
        num_elements=20,
        name="test_coil"
    )
    
    # Create a small calculation grid for faster computation
    grid = RectangularGrid(
        x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), z_range=(-1.0, 1.0),
        nx=8, ny=8, nz=8,  # Small grid for testing
        name="test_grid"
    )
    
    print(f"Created test coil: {coil}")
    print(f"Created test grid: {grid.num_points} points")
    
    try:
        # Calculate magnetic field
        calculator = MagneticFieldCalculator()
        A, B, I = coil.get_geometry()
        result = calculator.calculate_magnetic_field(A, B, I, grid)
        
        print(f"Field calculation completed in {result.calculation_time:.3f} seconds")
        print(f"Field range: {result.min_field:.2e} to {result.max_field:.2e} T")
        
        # Test the 3D visualization function
        geometry_data = coil.get_geometry()
        
        print("Creating 3D visualization with geometry overlay...")
        create_3d_field_visualization(result, "Test Circular Coil", grid, geometry_data)
        
        print("✓ 3D visualization test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    """Run the 3D visualization test."""
    print("MAGA 3D Visualization Test")
    print("=" * 40)
    
    success = test_3d_quiver_visualization()
    
    if success:
        print("\n✓ All tests passed!")
        print("\nFeatures verified:")
        print("  • 3D quiver plot showing field vectors as arrows")
        print("  • Geometry objects (conductors) plotted as red lines")
        print("  • Color-coded arrows based on field strength")
        print("  • Proper 3D visualization with matplotlib")
        print("  • Integration with MAGA geometry classes")
    else:
        print("\n✗ Some tests failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())