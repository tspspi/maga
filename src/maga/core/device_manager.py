"""
OpenCL device management for MAGA library.

This module handles:
1. Platform and device discovery
2. Context and command queue creation
3. Memory buffer management with proper flags
4. Error handling and graceful fallbacks

Key features:
- Automatic device selection with manual override
- CPU fallback support for broad compatibility
- Smart buffer allocation with size calculation
- Memory usage tracking
"""

import pyopencl as cl
import numpy as np
from typing import List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class DeviceInfo:
    """Information about an available OpenCL device."""
    
    def __init__(self, platform_idx: int, device_idx: int, platform: cl.Platform, device: cl.Device):
        self.platform_idx = platform_idx
        self.device_idx = device_idx
        self.platform = platform
        self.device = device
        
    @property
    def name(self) -> str:
        """Device name."""
        return self.device.name
        
    @property
    def platform_name(self) -> str:
        """Platform name."""
        return self.platform.name
        
    @property
    def device_type(self) -> str:
        """Device type (CPU, GPU, etc.)."""
        return cl.device_type.to_string(self.device.type)
        
    @property
    def global_memory_mb(self) -> int:
        """Global memory size in MB."""
        return self.device.global_mem_size // (1024 * 1024)
        
    @property
    def max_work_group_size(self) -> int:
        """Maximum work group size."""
        return self.device.max_work_group_size
        
    def __str__(self) -> str:
        return f"{self.platform_name} / {self.name} ({self.device_type}, {self.global_memory_mb}MB)"


class DeviceManager:
    """
    Manages OpenCL devices, contexts, and memory for MAGA calculations.
    
    Handles device discovery, context creation, and buffer management
    with proper memory flags for the MAGA kernel pipeline.
    """
    
    def __init__(self, platform_id: Optional[int] = None, device_id: Optional[int] = None):
        """
        Initialize device manager with optional device selection.
        
        Args:
            platform_id: Platform index (None for auto-selection)
            device_id: Device index within platform (None for auto-selection)
        """
        self.platform_id = platform_id
        self.device_id = device_id
        
        # OpenCL objects
        self.platform = None
        self.device = None
        self.context = None
        self.queue = None
        
        # Memory tracking
        self.allocated_buffers = []
        self.total_allocated_bytes = 0
        
        # Kernel registry (will be set by field calculator)
        self.kernel_registry = None
        
        # Initialize the device
        self._initialize_device()
        
    def list_available_devices(self) -> List[DeviceInfo]:
        """
        List all available OpenCL devices across all platforms.
        
        Returns:
            List of DeviceInfo objects
        """
        devices = []
        try:
            platforms = cl.get_platforms()
            for platform_idx, platform in enumerate(platforms):
                platform_devices = platform.get_devices()
                for device_idx, device in enumerate(platform_devices):
                    devices.append(DeviceInfo(platform_idx, device_idx, platform, device))
        except Exception as e:
            logger.warning(f"Error listing devices: {e}")
            
        return devices
        
    def _initialize_device(self):
        """Initialize OpenCL platform, device, context, and queue."""
        try:
            available_devices = self.list_available_devices()
            if not available_devices:
                raise RuntimeError("No OpenCL devices found")
                
            # Log available devices
            logger.info("Available OpenCL devices:")
            for i, device_info in enumerate(available_devices):
                logger.info(f"  {i}: {device_info}")
                
            # Select device
            if self.platform_id is not None and self.device_id is not None:
                # Manual device selection
                selected_device = self._find_device_by_indices(available_devices)
                if selected_device is None:
                    raise ValueError(f"Device not found: platform {self.platform_id}, device {self.device_id}")
            else:
                # Automatic device selection (prefer GPU, then CPU)
                selected_device = self._auto_select_device(available_devices)
                
            logger.info(f"Selected device: {selected_device}")
            
            # Store device information
            self.platform = selected_device.platform
            self.device = selected_device.device
            self.platform_id = selected_device.platform_idx
            self.device_id = selected_device.device_idx
            
            # Create context and command queue
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, self.device)
            
            logger.info("OpenCL context and queue created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenCL device: {e}")
            raise RuntimeError(f"OpenCL initialization failed: {e}")
            
    def _find_device_by_indices(self, available_devices: List[DeviceInfo]) -> Optional[DeviceInfo]:
        """Find device by platform and device indices."""
        for device_info in available_devices:
            if device_info.platform_idx == self.platform_id and device_info.device_idx == self.device_id:
                return device_info
        return None
        
    def _auto_select_device(self, available_devices: List[DeviceInfo]) -> DeviceInfo:
        """
        Automatically select the best available device.
        
        Priority: GPU with most memory > CPU with most memory > any device
        """
        if not available_devices:
            raise RuntimeError("No devices available for selection")
            
        # Separate GPUs and CPUs
        gpus = [d for d in available_devices if 'GPU' in d.device_type]
        cpus = [d for d in available_devices if 'CPU' in d.device_type]
        
        # Prefer GPU with most memory
        if gpus:
            return max(gpus, key=lambda d: d.global_memory_mb)
            
        # Fall back to CPU with most memory
        if cpus:
            return max(cpus, key=lambda d: d.global_memory_mb)
            
        # Last resort: any device
        return available_devices[0]
        
    def create_buffer(self, size: Optional[int] = None, flags: cl.mem_flags = cl.mem_flags.READ_WRITE,
                     hostptr: Optional[np.ndarray] = None, dtype: type = np.float64) -> cl.Buffer:
        """
        Create OpenCL buffer with automatic size calculation and proper flags.
        
        Args:
            size: Buffer size in bytes (calculated from hostptr if None)
            flags: OpenCL memory flags (default: READ_WRITE for kernel chaining)
            hostptr: Numpy array to copy to device (enables COPY_HOST_PTR)
            dtype: Data type for size calculation
            
        Returns:
            OpenCL buffer object
            
        Note:
            Most MAGA buffers need READ_WRITE flags because:
            - Transformation kernel WRITES matrices
            - Field calculation kernel READS matrices
            - Target buffer accumulates results
        """
        if hostptr is not None:
            # Use numpy array for initialization
            if size is None:
                size = hostptr.nbytes
            flags |= cl.mem_flags.COPY_HOST_PTR
            buffer = cl.Buffer(self.context, flags, hostbuf=hostptr)
        else:
            # Create empty buffer
            if size is None:
                raise ValueError("Either size or hostptr must be provided")
            buffer = cl.Buffer(self.context, flags, size=size)
            
        # Track memory usage
        self.allocated_buffers.append(buffer)
        self.total_allocated_bytes += size
        
        logger.debug(f"Created buffer: {size} bytes, total allocated: {self.total_allocated_bytes}")
        
        return buffer
        
    def copy_to_device(self, host_array: np.ndarray, 
                      buffer: Optional[cl.Buffer] = None) -> cl.Buffer:
        """
        Copy numpy array to device buffer.
        
        Args:
            host_array: Source numpy array
            buffer: Target buffer (created if None)
            
        Returns:
            Device buffer containing the data
        """
        if buffer is None:
            buffer = self.create_buffer(hostptr=host_array)
        else:
            cl.enqueue_copy(self.queue, buffer, host_array)
            
        return buffer
        
    def copy_from_device(self, device_buffer: cl.Buffer, 
                        host_array: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Copy data from device buffer to host.
        
        Args:
            device_buffer: Source device buffer
            host_array: Target numpy array (created if None)
            
        Returns:
            Numpy array with copied data
        """
        if host_array is None:
            # Create appropriately sized array
            buffer_size = device_buffer.size
            host_array = np.empty(buffer_size // 8, dtype=np.float64)  # Assume double precision
            
        event = cl.enqueue_copy(self.queue, host_array, device_buffer)
        event.wait()
        
        return host_array
        
    def get_device_info(self) -> DeviceInfo:
        """Get information about the current device."""
        return DeviceInfo(self.platform_id, self.device_id, self.platform, self.device)
        
    def estimate_memory_requirements(self, num_elements: int, grid_points: int) -> dict:
        """
        Estimate memory requirements for a calculation.
        
        Args:
            num_elements: Number of line elements
            grid_points: Number of grid points
            
        Returns:
            Dictionary with memory estimates in bytes
        """
        double_size = 8  # 8 bytes per double
        
        requirements = {
            # Input geometry (can be READ_ONLY)
            'geometry_A': num_elements * 3 * double_size,
            'geometry_B': num_elements * 3 * double_size,
            'geometry_I': num_elements * double_size,
            
            # Transformation data (READ_WRITE)
            'transform_M': num_elements * 16 * double_size,
            'transform_R_inv': num_elements * 16 * double_size,
            'transform_LI': num_elements * 2 * double_size,
            
            # Target field (READ_WRITE)
            'target_field': grid_points * 3 * double_size,
        }
        
        requirements['total'] = sum(requirements.values())
        requirements['total_mb'] = requirements['total'] / (1024 * 1024)
        
        return requirements
        
    def check_memory_availability(self, num_elements: int, grid_points: int) -> bool:
        """
        Check if device has sufficient memory for calculation.
        
        Args:
            num_elements: Number of line elements
            grid_points: Number of grid points
            
        Returns:
            True if sufficient memory available
        """
        requirements = self.estimate_memory_requirements(num_elements, grid_points)
        available_bytes = self.device.global_mem_size
        
        # Use 80% of available memory as safety margin
        usable_bytes = available_bytes * 0.8
        
        return requirements['total'] <= usable_bytes
        
    def release_resources(self):
        """Release OpenCL resources and clear memory tracking."""
        try:
            # Release buffers
            for buffer in self.allocated_buffers:
                try:
                    buffer.release()
                except:
                    pass  # Buffer may already be released
                    
            # Clear tracking
            self.allocated_buffers.clear()
            self.total_allocated_bytes = 0
            
            # Release OpenCL objects
            if self.queue:
                self.queue.finish()  # Wait for pending operations
                
            logger.info("OpenCL resources released")
            
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
            
    def __del__(self):
        """Cleanup resources on destruction."""
        self.release_resources()
        
    def __str__(self) -> str:
        """String representation of device manager."""
        if self.device:
            device_info = self.get_device_info()
            return f"DeviceManager({device_info})"
        return "DeviceManager(not initialized)"