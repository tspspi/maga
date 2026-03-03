"""
Magnetic field calculator for MAGA library.

This module provides the main calculation engine that combines geometry data,
grid configurations, and OpenCL kernels to compute magnetic fields using
the GPU-accelerated Biot-Savart implementation.

Key features:
- 3-stage kernel pipeline for efficient GPU calculation
- Memory management with automatic batching
- Support for time-dependent simulations
- Error handling and validation
- Performance monitoring
"""

import numpy as np
import pyopencl as cl
from typing import Optional, Union, Dict, Any, Tuple
import time
import logging

from .kernels import KernelRegistry
from .device_manager import DeviceManager
from .grid import GridConfiguration

logger = logging.getLogger(__name__)


class CalculationResult:
    """Container for magnetic field calculation results."""
    
    def __init__(self, 
                 magnetic_field: np.ndarray,
                 grid_coordinates: np.ndarray,
                 calculation_time: float,
                 num_elements: int,
                 simulation_time: Optional[float] = None):
        """
        Initialize calculation result.
        
        Args:
            magnetic_field: Array of shape (N, 3) with [Bx, By, Bz] components
            grid_coordinates: Array of shape (N, 3) with [x, y, z] coordinates
            calculation_time: Total calculation time in seconds
            num_elements: Number of line elements in geometry
            simulation_time: Simulation time for time-dependent calculations
        """
        self.magnetic_field = magnetic_field
        self.grid_coordinates = grid_coordinates
        self.calculation_time = calculation_time
        self.num_elements = num_elements
        self.simulation_time = simulation_time
        
    @property
    def num_points(self) -> int:
        """Number of calculation points."""
        return self.magnetic_field.shape[0]
        
    @property
    def field_magnitude(self) -> np.ndarray:
        """Magnitude of magnetic field at each point."""
        return np.linalg.norm(self.magnetic_field, axis=1)
        
    @property
    def max_field(self) -> float:
        """Maximum field magnitude."""
        return self.field_magnitude.max()
        
    @property
    def min_field(self) -> float:
        """Minimum field magnitude."""
        return self.field_magnitude.min()
        
    def __str__(self) -> str:
        return (f"CalculationResult({self.num_points} points, "
                f"{self.num_elements} elements, "
                f"t={self.calculation_time:.3f}s)")


class MagneticFieldCalculator:
    """
    Main magnetic field calculation engine for MAGA library.
    
    Orchestrates the GPU-accelerated Biot-Savart calculation by managing
    kernels, device resources, and the computation pipeline.
    """
    
    def __init__(self, 
                 device_manager: Optional[DeviceManager] = None,
                 kernel_registry: Optional[KernelRegistry] = None):
        """
        Initialize magnetic field calculator.
        
        Args:
            device_manager: OpenCL device manager (created if None)
            kernel_registry: Kernel registry (created if None)
        """
        # Initialize components with proper dependency injection
        # DeviceManager must be created first as KernelRegistry depends on it
        if device_manager is not None:
            self.device_manager = device_manager
            # If device_manager is provided but kernel_registry is not, create one with dependency
            self.kernel_registry = kernel_registry or KernelRegistry(self.device_manager)
        else:
            # Create DeviceManager first, then KernelRegistry with dependency
            self.device_manager = DeviceManager()
            self.kernel_registry = kernel_registry or KernelRegistry(self.device_manager)
        
        # Register default kernels if kernel_registry was created (not provided)
        if kernel_registry is None:
            from .kernels import register_default_kernels
            register_default_kernels(self.kernel_registry)
        
        # Compiled kernels (cached after first use)
        self._compiled_kernels = None
        self._kernel_program = None
        
        # Calculation statistics
        self.last_calculation_time = None
        self.total_calculations = 0
        
        logger.info(f"Initialized MagneticFieldCalculator with {self.device_manager}")
        
    def _compile_kernels(self, macros: Optional[Dict[str, Any]] = None):
        """
        Compile OpenCL kernels if not already compiled.
        
        Args:
            macros: Preprocessor macros for kernel compilation
        """
        if self._compiled_kernels is not None:
            return  # Already compiled
            
        try:
            # Compile the main magnetic field program with optional macros
            macro_args = [(name, value) for name, value in (macros or {}).items()]
            self.kernel_registry.compile_kernel("magnetic_field_program", [
                "utility_functions",
                "initialize_target",
                "process_point_pairs",
                "calculate_magnetic_field"
            ], args=macro_args)
            
            # Cache individual kernels using the KernelRegistry API
            self._compiled_kernels = {
                'initialize_target_buffer': self.kernel_registry.get_kernel("magnetic_field_program", "initialize_target_buffer"),
                'process_point_pairs_optimized': self.kernel_registry.get_kernel("magnetic_field_program", "process_point_pairs_optimized"),
                'calculate_magnetic_field': self.kernel_registry.get_kernel("magnetic_field_program", "calculate_magnetic_field")
            }
            
            logger.info("OpenCL kernels compiled successfully")
            
        except Exception as e:
            logger.error(f"Kernel compilation failed: {e}")
            raise RuntimeError(f"Failed to compile OpenCL kernels: {e}")
            
    def calculate_magnetic_field(self,
                                geometry_A: np.ndarray,
                                geometry_B: np.ndarray, 
                                geometry_I: np.ndarray,
                                grid: GridConfiguration,
                                simulation_time: float = 0.0,
                                batch_size: Optional[int] = None) -> CalculationResult:
        """
        Calculate magnetic field for given geometry and grid.
        
        Args:
            geometry_A: Line element start points, shape (N, 3)
            geometry_B: Line element end points, shape (N, 3)  
            geometry_I: Current values, shape (N,)
            grid: Grid configuration defining calculation points
            simulation_time: Time parameter for time-dependent calculations
            batch_size: Maximum points per batch (None for automatic)
            
        Returns:
            CalculationResult with magnetic field data
        """
        start_time = time.time()
        
        # Validate inputs and get processed arrays back
        geometry_A, geometry_B, geometry_I = self._validate_geometry_inputs(geometry_A, geometry_B, geometry_I)
        
        # Generate grid coordinates
        if grid.coordinates is None:
            grid.generate_coordinates()
        coordinates = grid.coordinates
        
        # Store grid for diagnostic purposes
        self._last_grid = grid
        
        # Check memory requirements and determine batching
        num_elements = len(geometry_I)
        num_points = len(coordinates)
        
        if batch_size is None:
            batch_size = self._determine_batch_size(num_elements, num_points)
            
        logger.info(f"Starting calculation: {num_elements} elements, {num_points} points, "
                   f"batch_size={batch_size}, t={simulation_time}")
        
        # Perform calculation (with batching if needed)
        if batch_size >= num_points:
            # Single batch calculation
            magnetic_field = self._calculate_single_batch(
                geometry_A, geometry_B, geometry_I, coordinates, simulation_time
            )
        else:
            # Multi-batch calculation
            magnetic_field = self._calculate_multi_batch(
                geometry_A, geometry_B, geometry_I, coordinates, simulation_time, batch_size
            )
            
        # Create result
        calculation_time = time.time() - start_time
        self.last_calculation_time = calculation_time
        self.total_calculations += 1
        
        result = CalculationResult(
            magnetic_field=magnetic_field,
            grid_coordinates=coordinates.copy(),
            calculation_time=calculation_time,
            num_elements=num_elements,
            simulation_time=simulation_time
        )
        
        logger.info(f"Calculation completed in {calculation_time:.3f}s, "
                   f"max field: {result.max_field:.2e} T")
        
        return result
        
    def _validate_geometry_inputs(self, geometry_A: np.ndarray, geometry_B: np.ndarray, geometry_I: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate geometry input arrays and return processed arrays."""
        # Convert to numpy arrays and ensure proper shape
        geometry_A = np.asarray(geometry_A, dtype=np.float64)
        geometry_B = np.asarray(geometry_B, dtype=np.float64)
        geometry_I = np.asarray(geometry_I, dtype=np.float64)
        
        # Handle scalar current input - expand to match number of line elements
        if geometry_I.ndim == 0:
            # Single scalar current value - need to determine how many elements we have
            num_elements = len(geometry_A) if geometry_A.ndim > 0 else 1
            geometry_I = np.full(num_elements, geometry_I)  # Replicate scalar for each element
        
        # Geometry arrays cannot be scalars
        if geometry_A.ndim == 0:
            raise ValueError("geometry_A cannot be a scalar")
        if geometry_B.ndim == 0:
            raise ValueError("geometry_B cannot be a scalar")
        
        # Ensure geometry_A and geometry_B are 2D
        if geometry_A.ndim == 1:
            if len(geometry_A) == 3:
                geometry_A = geometry_A.reshape(1, 3)  # Single point
            else:
                raise ValueError("geometry_A must have shape (N, 3)")
        
        if geometry_B.ndim == 1:
            if len(geometry_B) == 3:
                geometry_B = geometry_B.reshape(1, 3)  # Single point
            else:
                raise ValueError("geometry_B must have shape (N, 3)")
        
        # Check shapes
        if geometry_A.ndim != 2 or geometry_A.shape[1] != 3:
            raise ValueError(f"geometry_A must have shape (N, 3), got {geometry_A.shape}")
        if geometry_B.ndim != 2 or geometry_B.shape[1] != 3:
            raise ValueError(f"geometry_B must have shape (N, 3), got {geometry_B.shape}")
        if geometry_I.ndim != 1:
            raise ValueError(f"geometry_I must have shape (N,), got {geometry_I.shape}")
            
        # Check consistent sizes
        if len(geometry_A) != len(geometry_B) or len(geometry_A) != len(geometry_I):
            raise ValueError(f"Geometry arrays must have consistent sizes: A={len(geometry_A)}, B={len(geometry_B)}, I={len(geometry_I)}")
            
        # Check for invalid values
        if not np.all(np.isfinite(geometry_A)) or not np.all(np.isfinite(geometry_B)):
            raise ValueError("Geometry coordinates contain invalid values")
        if not np.all(np.isfinite(geometry_I)):
            raise ValueError("Current values contain invalid values")
            
        return geometry_A, geometry_B, geometry_I
            
    def _determine_batch_size(self, num_elements: int, num_points: int) -> int:
        """
        Determine optimal batch size based on memory constraints.
        
        Args:
            num_elements: Number of line elements
            num_points: Number of grid points
            
        Returns:
            Optimal batch size for points
        """
        # Check if single batch fits in memory
        if self.device_manager.check_memory_availability(num_elements, num_points):
            return num_points
            
        # Calculate maximum batch size that fits in memory
        device_memory = self.device_manager.device.global_mem_size
        usable_memory = device_memory * 0.8  # 80% safety margin
        
        # Estimate per-point memory usage (excluding geometry)
        bytes_per_point = 3 * 8  # Target field (3 doubles)
        
        # Add per-element memory (shared across all points in batch)
        geometry_bytes = num_elements * (3 * 3 + 1) * 8  # A, B, I (3x3 + 1 doubles each)
        transform_bytes = num_elements * (16 + 16 + 2) * 8  # M, R_inv, LI
        
        # Available memory for points
        available_for_points = usable_memory - geometry_bytes - transform_bytes
        
        # Maximum points that fit
        max_points = max(1, int(available_for_points / bytes_per_point))
        
        # Use smaller of max_points and total points
        batch_size = min(max_points, num_points)
        
        logger.info(f"Determined batch size: {batch_size} points "
                   f"(memory limit: {device_memory / 1024**2:.0f} MB)")
        
        return batch_size
        
    def _calculate_single_batch(self,
                               geometry_A: np.ndarray,
                               geometry_B: np.ndarray,
                               geometry_I: np.ndarray,
                               coordinates: np.ndarray,
                               simulation_time: float) -> np.ndarray:
        """
        Calculate magnetic field for single batch that fits in GPU memory.
        
        Returns:
            Magnetic field array of shape (num_points, 3)
        """
        # Ensure kernels are compiled
        self._compile_kernels({'SIMULATION_TIME': simulation_time})
        
        num_elements = len(geometry_I)
        num_points = len(coordinates)
        
        # Create device buffers
        buffers = self._create_calculation_buffers(
            geometry_A, geometry_B, geometry_I, coordinates
        )
        
        try:
            # Execute kernel pipeline
            self._execute_kernel_pipeline(buffers, num_elements, num_points, simulation_time)
            
            # Copy result back to host
            # Check if we have proper grid shape info for reshaping
            grid_info = getattr(self, '_last_grid', None)
            if hasattr(grid_info, 'grid_shape'):
                # Return as 4D array (nx, ny, nz, 3) like the notebook
                shape = grid_info.grid_shape
                result_field = np.empty(shape + (3,), dtype=np.float64)
                cl.enqueue_copy(self.device_manager.queue, result_field, buffers['target_field']).wait()
            else:
                # Fallback to flat array for non-rectangular grids
                result_field = np.empty((num_points, 3), dtype=np.float64)
                cl.enqueue_copy(self.device_manager.queue, result_field, buffers['target_field']).wait()
            
            return result_field
            
        finally:
            # Release buffers
            for buffer in buffers.values():
                buffer.release()
                
    def _calculate_multi_batch(self,
                              geometry_A: np.ndarray,
                              geometry_B: np.ndarray, 
                              geometry_I: np.ndarray,
                              coordinates: np.ndarray,
                              simulation_time: float,
                              batch_size: int) -> np.ndarray:
        """
        Calculate magnetic field using multiple batches.
        
        Returns:
            Magnetic field array of shape (num_points, 3)
        """
        num_points = len(coordinates)
        num_batches = (num_points + batch_size - 1) // batch_size
        
        # Initialize result array - check if we need 4D shape
        grid_info = getattr(self, '_last_grid', None)
        if hasattr(grid_info, 'grid_shape'):
            shape = grid_info.grid_shape
            total_field = np.zeros(shape + (3,), dtype=np.float64)
        else:
            total_field = np.zeros((num_points, 3), dtype=np.float64)
        
        logger.info(f"Processing {num_batches} batches of up to {batch_size} points each")
        
        # Process each batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_points)
            batch_coords = coordinates[start_idx:end_idx]
            
            logger.debug(f"Processing batch {batch_idx + 1}/{num_batches}: "
                        f"points {start_idx}:{end_idx}")
            
            # Calculate field for this batch
            batch_field = self._calculate_single_batch(
                geometry_A, geometry_B, geometry_I, batch_coords, simulation_time
            )
            
            # Store result (handle both flat and 4D arrays)
            if hasattr(grid_info, 'grid_shape'):
                # For 4D arrays, we need to reshape the batch result properly
                shape = grid_info.grid_shape
                batch_points = end_idx - start_idx
                # Calculate which slice of the 4D array this batch represents
                flat_total = total_field.reshape(-1, 3)
                flat_total[start_idx:end_idx] = batch_field.reshape(-1, 3)
            else:
                total_field[start_idx:end_idx] = batch_field
            
        return total_field
        
    def _create_calculation_buffers(self,
                                  geometry_A: np.ndarray,
                                  geometry_B: np.ndarray,
                                  geometry_I: np.ndarray,
                                  coordinates: np.ndarray) -> Dict[str, cl.Buffer]:
        """
        Create OpenCL buffers for calculation.
        
        Returns:
            Dictionary of OpenCL buffers
        """
        num_elements = len(geometry_I)
        num_points = len(coordinates)
        
        # Create buffers with proper flags
        buffers = {
            # Input geometry (READ_ONLY)
            'geometry_A': self.device_manager.create_buffer(
                hostptr=geometry_A, flags=cl.mem_flags.READ_ONLY
            ),
            'geometry_B': self.device_manager.create_buffer(
                hostptr=geometry_B, flags=cl.mem_flags.READ_ONLY
            ),
            'geometry_I': self.device_manager.create_buffer(
                hostptr=geometry_I, flags=cl.mem_flags.READ_ONLY
            ),
            
            # Target coordinates (READ_ONLY)
            'target_coords': self.device_manager.create_buffer(
                hostptr=coordinates, flags=cl.mem_flags.READ_ONLY
            ),
            
            # Transformation data (READ_WRITE for kernel chaining)
            'transform_M': self.device_manager.create_buffer(
                size=num_elements * 16 * 8, flags=cl.mem_flags.READ_WRITE
            ),
            'transform_R_inv': self.device_manager.create_buffer(
                size=num_elements * 16 * 8, flags=cl.mem_flags.READ_WRITE
            ),
            'transform_LI': self.device_manager.create_buffer(
                size=num_elements * 2 * 8, flags=cl.mem_flags.READ_WRITE
            ),
            
            # Target field (READ_WRITE for accumulation)
            'target_field': self.device_manager.create_buffer(
                size=num_points * 3 * 8, flags=cl.mem_flags.READ_WRITE
            )
        }
        
        return buffers
        
    def _execute_kernel_pipeline(self,
                                buffers: Dict[str, cl.Buffer],
                                num_elements: int,
                                num_points: int,
                                simulation_time: float):
        """
        Execute the 3-stage kernel pipeline.
        
        Args:
            buffers: OpenCL buffers for calculation
            num_elements: Number of line elements
            num_points: Number of grid points
            simulation_time: Simulation time parameter
        """
        queue = self.device_manager.queue
        kernels = self._compiled_kernels
        
        # Extract proper grid parameters if available
        grid_info = getattr(self, '_last_grid', None)
        if hasattr(grid_info, 'grid_shape') and hasattr(grid_info, 'spacing'):
            # Use actual grid parameters from RectangularGrid
            origin = (grid_info.x_range[0], grid_info.y_range[0], grid_info.z_range[0])
            spacing = grid_info.spacing
            shape = grid_info.grid_shape  # (nx, ny, nz)
            global_work_size = shape  # Use proper 3D dimensions
            logger.info(f"Using proper grid parameters: origin={origin}, spacing={spacing}, shape={shape}")
        else:
            # Fallback to linearized approach for non-rectangular grids
            origin = (0.0, 0.0, 0.0)
            spacing = (1.0, 1.0, 1.0)
            shape = (num_points, 1, 1)
            global_work_size = (num_points, 1, 1)
            logger.warning("No grid metadata available, using fallback parameters")
        
        # Stage 1: Initialize target buffer
        logger.debug("Executing initialize_target_buffer kernel")
        kernels['initialize_target_buffer'](
            queue, global_work_size, None,
            buffers['target_field'],
            np.int32(shape[0]),
            np.int32(shape[1]),
            np.int32(shape[2])
        ).wait()
        
        # Stage 2: Process point pairs (calculate transformation matrices)
        logger.debug("Executing process_point_pairs_optimized kernel")
        kernels['process_point_pairs_optimized'](
            queue, (num_elements,), None,
            buffers['geometry_A'],
            buffers['geometry_B'],
            buffers['geometry_I'],
            buffers['transform_M'],
            buffers['transform_R_inv'],
            buffers['transform_LI'],
            np.int32(num_elements)
        ).wait()
        
        # Stage 3: Calculate magnetic field
        logger.debug("Executing calculate_magnetic_field kernel")
        kernels['calculate_magnetic_field'](
            queue, global_work_size, None,
            buffers['transform_M'],
            buffers['transform_R_inv'],
            buffers['transform_LI'],
            np.int32(num_elements),
            buffers['target_field'],
            # Use proper grid origin
            np.float64(origin[0]), np.float64(origin[1]), np.float64(origin[2]),
            # Use proper grid spacing
            np.float64(spacing[0]), np.float64(spacing[1]), np.float64(spacing[2]),
            # Use proper grid dimensions
            np.int32(shape[0]), np.int32(shape[1]), np.int32(shape[2])
        ).wait()
        logger.debug("calculate_magnetic_field completed with proper grid parameters")
        
    def get_device_info(self) -> str:
        """Get information about the calculation device."""
        return str(self.device_manager.get_device_info())
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get calculation statistics."""
        return {
            'total_calculations': self.total_calculations,
            'last_calculation_time': self.last_calculation_time,
            'device_info': str(self.device_manager.get_device_info()),
            'total_allocated_memory': self.device_manager.total_allocated_bytes
        }
        
    def release_resources(self):
        """Release OpenCL resources."""
        self.device_manager.release_resources()
        self._compiled_kernels = None
        self._kernel_program = None
        
    def __del__(self):
        """Cleanup on destruction."""
        self.release_resources()