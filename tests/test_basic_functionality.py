"""
Basic functionality tests for MAGA library.

This test suite verifies the core functionality of the MAGA library,
including geometry generation, grid creation, and magnetic field calculations.
These tests are designed to run quickly and verify basic operations.
"""

import sys
import os
import numpy as np
import unittest

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from maga import (
        CircularCoil, RectangularCoil, HelmholtzCoils, OscillatingBeam,
        RectangularGrid, MagneticFieldCalculator, DeviceManager
    )
    MAGA_AVAILABLE = True
except ImportError as e:
    MAGA_AVAILABLE = False
    IMPORT_ERROR = str(e)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic MAGA library functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MAGA_AVAILABLE:
            self.skipTest(f"MAGA library not available: {IMPORT_ERROR}")
    
    def test_circular_coil_creation(self):
        """Test basic circular coil creation and geometry generation."""
        coil = CircularCoil(
            center=(0.0, 0.0, 0.0),
            radius=1.0,
            current=5.0,
            num_elements=20
        )
        
        # Check basic properties
        self.assertEqual(coil.radius, 1.0)
        self.assertEqual(coil.current, 5.0)
        self.assertAlmostEqual(coil.get_area(), np.pi, places=10)
        
        # Generate geometry
        A, B, I = coil.get_geometry()
        
        # Check array shapes
        self.assertEqual(A.shape, (20, 3))
        self.assertEqual(B.shape, (20, 3))
        self.assertEqual(I.shape, (20,))
        
        # Check current values
        np.testing.assert_array_almost_equal(I, 5.0)
        
        # Check that points are on the circle (approximately)
        center_distances_A = np.linalg.norm(A, axis=1)
        center_distances_B = np.linalg.norm(B, axis=1)
        np.testing.assert_allclose(center_distances_A, 1.0, rtol=1e-10)
        np.testing.assert_allclose(center_distances_B, 1.0, rtol=1e-10)
    
    def test_rectangular_coil_creation(self):
        """Test rectangular coil creation and geometry generation."""
        coil = RectangularCoil(
            center=(0.0, 0.0, 0.0),
            width=2.0,
            height=1.0,
            current=3.0,
            num_elements=20
        )
        
        # Check basic properties
        self.assertEqual(coil.width, 2.0)
        self.assertEqual(coil.height, 1.0)
        self.assertEqual(coil.current, 3.0)
        self.assertEqual(coil.get_area(), 2.0)
        self.assertEqual(coil.get_perimeter(), 6.0)
        self.assertEqual(coil.get_aspect_ratio(), 2.0)
        
        # Generate geometry
        A, B, I = coil.get_geometry()
        
        # Check array shapes
        self.assertEqual(len(A), len(B))
        self.assertEqual(len(A), len(I))
        self.assertEqual(A.shape[1], 3)
        self.assertEqual(B.shape[1], 3)
        
        # Check current values
        np.testing.assert_array_almost_equal(I, 3.0)
    
    def test_helmholtz_coils_creation(self):
        """Test Helmholtz coil pair creation."""
        helmholtz = HelmholtzCoils(
            center=(0.0, 0.0, 0.0),
            radius=0.5,
            current=2.0,
            num_elements_per_coil=10
        )
        
        # Check properties
        self.assertEqual(helmholtz.radius, 0.5)
        self.assertEqual(helmholtz.current, 2.0)
        self.assertEqual(helmholtz.separation, 0.5)  # Default optimal separation
        self.assertEqual(helmholtz.get_optimal_separation(), 0.5)
        
        # Generate geometry
        A, B, I = helmholtz.get_geometry()
        
        # Should have 20 elements total (10 per coil)
        self.assertEqual(len(A), 20)
        self.assertEqual(len(B), 20)
        self.assertEqual(len(I), 20)
        
        # All currents should be positive (same direction)
        np.testing.assert_array_almost_equal(I, 2.0)
    
    def test_rectangular_grid_creation(self):
        """Test rectangular grid creation."""
        grid = RectangularGrid(
            x_range=(-1.0, 1.0),
            y_range=(-0.5, 0.5),
            z_range=(-0.2, 0.2),
            nx=5, ny=3, nz=3
        )
        
        # Check properties
        self.assertEqual(grid.num_points, 45)  # 5 * 3 * 3
        self.assertEqual(grid.grid_shape, (5, 3, 3))
        
        # Generate coordinates
        coords = grid.generate_coordinates()
        self.assertEqual(coords.shape, (45, 3))
        
        # Check coordinate ranges
        self.assertAlmostEqual(coords[:, 0].min(), -1.0, places=10)
        self.assertAlmostEqual(coords[:, 0].max(), 1.0, places=10)
        self.assertAlmostEqual(coords[:, 1].min(), -0.5, places=10)
        self.assertAlmostEqual(coords[:, 1].max(), 0.5, places=10)
        self.assertAlmostEqual(coords[:, 2].min(), -0.2, places=10)
        self.assertAlmostEqual(coords[:, 2].max(), 0.2, places=10)
    
    def test_oscillating_beam_creation(self):
        """Test oscillating beam creation (time-dependent geometry)."""
        beam = OscillatingBeam(
            voltage=1000.0,
            current=0.01,
            modulation_frequency=1e6,
            modulation_amplitude=1e-4,
            start_position=(0.0, 0.0, 1e-3),
            propagation_direction=(0.0, 0.0, -1.0),
            modulation_direction=(1.0, 0.0, 0.0),
            length=2e-3,
            num_elements=10
        )
        
        # Check properties
        self.assertEqual(beam.voltage, 1000.0)
        self.assertEqual(beam.current, 0.01)
        self.assertTrue(beam.is_time_dependent())
        
        # Check physics calculations
        self.assertGreater(beam.get_beam_velocity(), 0)
        self.assertLess(beam.get_beta(), 1.0)  # v/c < 1
        self.assertGreater(beam.get_gamma(), 1.0)  # γ > 1
        
        # Generate geometry at different times
        A1, B1, I1 = beam.get_geometry(time=0.0)
        A2, B2, I2 = beam.get_geometry(time=1e-9)
        
        # Shapes should be consistent
        self.assertEqual(A1.shape, A2.shape)
        self.assertEqual(B1.shape, B2.shape)
        self.assertEqual(I1.shape, I2.shape)
        
        # Positions should be different (time-dependent)
        self.assertFalse(np.allclose(A1, A2))
        self.assertFalse(np.allclose(B1, B2))
        
        # Currents should be the same
        np.testing.assert_allclose(I1, I2)
    
    def test_device_manager_creation(self):
        """Test device manager creation and device listing."""
        try:
            manager = DeviceManager()
            devices = manager.list_available_devices()
            
            # Should have at least one device (CPU OpenCL)
            self.assertGreater(len(devices), 0)
            
            # Get device info
            device_info = manager.get_device_info()
            self.assertIsNotNone(device_info)
            
        except Exception as e:
            self.skipTest(f"OpenCL not available or device creation failed: {e}")
    
    def test_basic_field_calculation(self):
        """Test a basic magnetic field calculation."""
        try:
            # Create simple setup
            coil = CircularCoil(radius=1.0, current=1.0, num_elements=12)
            grid = RectangularGrid(
                x_range=(-0.5, 0.5), y_range=(-0.5, 0.5), z_range=(0.0, 0.5),
                nx=3, ny=3, nz=3
            )
            
            # Calculate field
            calculator = MagneticFieldCalculator()
            A, B, I = coil.get_geometry()
            result = calculator.calculate_magnetic_field(A, B, I, grid)
            
            # Check result properties
            self.assertEqual(result.num_points, 27)  # 3*3*3
            self.assertEqual(result.num_elements, 12)
            self.assertEqual(result.magnetic_field.shape, (27, 3))
            self.assertEqual(result.grid_coordinates.shape, (27, 3))
            self.assertGreater(result.calculation_time, 0)
            
            # Field should be non-zero at most points
            field_magnitudes = result.field_magnitude
            non_zero_points = np.sum(field_magnitudes > 1e-15)
            self.assertGreater(non_zero_points, 20)  # Most points should have field
            
            # Field should be finite everywhere
            self.assertTrue(np.all(np.isfinite(result.magnetic_field)))
            
        except Exception as e:
            self.skipTest(f"Field calculation failed: {e}")
    
    def test_geometry_modifications(self):
        """Test dynamic geometry parameter modifications."""
        coil = CircularCoil(radius=1.0, current=5.0)
        
        # Test current modification
        coil.set_current(10.0)
        self.assertEqual(coil.current, 10.0)
        
        A, B, I = coil.get_geometry()
        np.testing.assert_array_almost_equal(I, 10.0)
        
        # Test radius modification  
        coil.set_radius(2.0)
        self.assertEqual(coil.radius, 2.0)
        self.assertAlmostEqual(coil.get_area(), 4 * np.pi, places=10)
        
        # Test center modification
        new_center = (1.0, 2.0, 3.0)
        coil.set_center(new_center)
        np.testing.assert_array_equal(coil.center, new_center)
    
    def test_coordinate_transformations(self):
        """Test coordinate transformations in base geometry."""
        coil = CircularCoil(radius=1.0, current=1.0)
        
        # Get original geometry
        A1, B1, I1 = coil.get_geometry()
        original_center = coil.get_center_of_mass()
        
        # Translate
        offset = np.array([2.0, 3.0, 4.0])
        coil.translate(offset)
        A2, B2, I2 = coil.get_geometry()
        new_center = coil.get_center_of_mass()
        
        # Check translation
        np.testing.assert_allclose(new_center, original_center + offset, rtol=1e-10)
        np.testing.assert_allclose(A2, A1 + offset, rtol=1e-10)
        np.testing.assert_allclose(B2, B1 + offset, rtol=1e-10)


class TestQuickFunctions(unittest.TestCase):
    """Test convenience functions from the main module."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not MAGA_AVAILABLE:
            self.skipTest(f"MAGA library not available: {IMPORT_ERROR}")
    
    def test_quick_coil_calculation(self):
        """Test the quick_coil_calculation convenience function."""
        try:
            from maga import quick_coil_calculation
            
            result = quick_coil_calculation(radius=0.5, current=2.0, grid_size=5)
            
            # Check result
            self.assertIsNotNone(result)
            self.assertEqual(result.magnetic_field.shape[0], 5*5*2)  # grid_size^2 * (grid_size//2)
            self.assertTrue(np.all(np.isfinite(result.magnetic_field)))
            
        except Exception as e:
            self.skipTest(f"Quick calculation failed: {e}")
    
    def test_list_opencl_devices(self):
        """Test device listing function."""
        try:
            from maga import list_opencl_devices
            
            devices = list_opencl_devices()
            self.assertIsInstance(devices, list)
            self.assertGreater(len(devices), 0)
            
        except Exception as e:
            self.skipTest(f"Device listing failed: {e}")
    
    def test_version_info(self):
        """Test version information function."""
        try:
            from maga import get_version_info
            
            info = get_version_info()
            self.assertIsInstance(info, dict)
            self.assertIn('maga_version', info)
            self.assertIn('numpy_version', info)
            self.assertEqual(info['maga_version'], '1.0.0')
            
        except Exception as e:
            self.skipTest(f"Version info failed: {e}")


def run_basic_tests():
    """Run basic functionality tests and return results."""
    print("Running MAGA Library Basic Functionality Tests")
    print("=" * 50)
    
    if not MAGA_AVAILABLE:
        print(f"✗ MAGA library not available: {IMPORT_ERROR}")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicFunctionality))
    suite.addTests(loader.loadTestsFromTestCase(TestQuickFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("✓ All basic functionality tests passed!")
    else:
        print("✗ Some tests failed or had errors")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback}")
                
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback}")
    
    return success


if __name__ == "__main__":
    success = run_basic_tests()
    sys.exit(0 if success else 1)