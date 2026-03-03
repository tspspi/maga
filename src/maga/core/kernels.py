"""
OpenCL kernel management and compilation for MAGA library.

This module contains:
1. KernelRegistry - manages source registration and compilation
2. Extracted OpenCL kernels from the original Jupyter notebook
3. Supporting utility functions for kernel operations

All kernels maintain double precision for numerical accuracy.
"""

import pyopencl as cl
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class KernelRegistry:
    """
    Manages OpenCL kernel source code and compilation.
    
    Supports source concatenation and preprocessor macro injection
    for flexible kernel compilation.
    """
    
    def __init__(self, device_manager):
        """
        Initialize kernel registry with device manager.
        
        Args:
            device_manager: DeviceManager instance for compilation context
        """
        self.device_manager = device_manager
        self.sources = {}  # name -> source code
        self.compiled_programs = {}  # compiled_name -> cl.Program
        self.kernels = {}  # (compiled_name, kernel_name) -> cl.Kernel
        
    def register_source(self, source_name: str, source: str):
        """
        Register OpenCL source code by name.
        
        Args:
            source_name: Unique identifier for the source
            source: OpenCL source code string
        """
        self.sources[source_name] = source
        
    def compile_kernel(self, compiled_name: str, sources: List[str], 
                      args: Optional[List[Tuple[str, Any]]] = None):
        """
        Compile source(s) with optional preprocessor macros.
        
        Args:
            compiled_name: Name for the compiled program
            sources: List of source names to concatenate
            args: Optional list of (name, value) tuples for #define macros
        """
        if compiled_name in self.compiled_programs:
            return  # Already compiled
            
        # Concatenate sources
        full_source = ""
        
        # Add preprocessor macros
        if args:
            for name, value in args:
                full_source += f"#define {name} {value}\n"
            full_source += "\n"
            
        # Add source code
        for source_name in sources:
            if source_name not in self.sources:
                raise ValueError(f"Source '{source_name}' not registered")
            full_source += self.sources[source_name]
            full_source += "\n\n"
            
        # Compile program
        try:
            program = cl.Program(self.device_manager.context, full_source).build()
            self.compiled_programs[compiled_name] = program
        except cl.RuntimeError as e:
            raise RuntimeError(f"Kernel compilation failed for '{compiled_name}': {e}")
            
    def get_kernel(self, compiled_name: str, kernel_name: str) -> cl.Kernel:
        """
        Retrieve compiled kernel.
        
        Args:
            compiled_name: Name of compiled program
            kernel_name: Name of specific kernel function
            
        Returns:
            Compiled OpenCL kernel
        """
        key = (compiled_name, kernel_name)
        if key not in self.kernels:
            if compiled_name not in self.compiled_programs:
                raise ValueError(f"Program '{compiled_name}' not compiled")
            program = self.compiled_programs[compiled_name]
            self.kernels[key] = getattr(program, kernel_name)
        return self.kernels[key]


# =============================================================================
# OpenCL Kernel Source Code (Extracted from Jupyter Notebook)
# =============================================================================

# Inline utility functions for matrix operations
UTILITY_FUNCTIONS = '''//CL//

/**
 * Compute axis-angle rotation matrix using Rodrigues' formula.
 * 
 * This is the corrected version from the notebook (lines 585-751).
 * Creates a 4x4 homogeneous transformation matrix for rotation around
 * an arbitrary axis by the specified angle.
 * 
 * Mathematical Formula:
 * R = I + sin(θ)·[k]× + (1-cos(θ))·[k]×²
 * where [k]× is the skew-symmetric matrix of the normalized axis vector
 * 
 * @param axis Input axis as array of 3 double values (will be normalized)
 * @param angle Rotation angle in radians
 * @param result Output 4x4 matrix as flat array (16 elements, row-major)
 */
inline void axis_angle_matrix(
    const double* axis, // Input axis as an array of 3 double values
    const double angle, // Input angle in radians
    double* result      // Output 4x4 matrix as a flat array (16 elements)
) {
    const double epsilon = 1e-8;

    // Check if axis is effectively zero
    if ((fabs(axis[0]) < epsilon) && (fabs(axis[1]) < epsilon) && (fabs(axis[2]) < epsilon)) {
        // Identity matrix for zero axis
        result[0] = 1.0; result[1] = 0.0; result[2] = 0.0; result[3] = 0.0;
        result[4] = 0.0; result[5] = 1.0; result[6] = 0.0; result[7] = 0.0;
        result[8] = 0.0; result[9] = 0.0; result[10] = 1.0; result[11] = 0.0;
        result[12] = 0.0; result[13] = 0.0; result[14] = 0.0; result[15] = 1.0;
        return;
    }

    // Normalize the axis
    double length = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
    double kx = axis[0] / length;
    double ky = axis[1] / length;
    double kz = axis[2] / length;
    
    // Compute trigonometric values
    double c = cos(angle);        
    double s = sin(angle);
    double one_minus_c = 1.0 - c; // Precompute (1 - cos(angle))
    
    // Compute rotation matrix using Rodrigues' formula
    result[0] = c + one_minus_c * kx * kx;
    result[1] = -s * kz + one_minus_c * kx * ky;
    result[2] = s * ky + one_minus_c * kx * kz;
    result[3] = 0.0;
    
    result[4] = s * kz + one_minus_c * kx * ky;
    result[5] = c + one_minus_c * ky * ky;
    result[6] = -s * kx + one_minus_c * ky * kz;
    result[7] = 0.0;
    
    result[8] = -s * ky + one_minus_c * kx * kz;
    result[9] = s * kx + one_minus_c * ky * kz;
    result[10] = c + one_minus_c * kz * kz;
    result[11] = 0.0;
    
    result[12] = 0.0;
    result[13] = 0.0;
    result[14] = 0.0;
    result[15] = 1.0;
}

/**
 * Create translation matrix for homogeneous coordinates.
 * 
 * Mathematical Formula:
 * T = [I  t]  where I is 3x3 identity, t is translation vector
 *     [0  1]
 * 
 * @param vec Input translation vector (3 doubles: x, y, z)
 * @param result Output 4x4 matrix as flat array (16 elements, row-major)
 */
inline void translation_matrix(
    const double* vec, // Input translation vector (3 doubles)
    double* result      // Output 4x4 matrix as a flat array (16 elements)
) {
    // Initialize as identity matrix with translation components
    result[0] = 1.0; result[1] = 0.0; result[2] = 0.0; result[3] = vec[0];
    result[4] = 0.0; result[5] = 1.0; result[6] = 0.0; result[7] = vec[1];
    result[8] = 0.0; result[9] = 0.0; result[10] = 1.0; result[11] = vec[2];
    result[12] = 0.0; result[13] = 0.0; result[14] = 0.0; result[15] = 1.0;
}

'''

# Target buffer initialization kernel
INITIALIZE_TARGET_KERNEL = '''//CL//

/**
 * Initialize target magnetic field buffer to zero.
 * 
 * Sets all magnetic field components (Bx, By, Bz) to zero across the entire
 * calculation grid. This kernel uses a 3D thread organization matching the grid.
 * 
 * @param target_buffer Output buffer for magnetic field (3 doubles per grid point)
 * @param grid_size_x Grid size in X dimension
 * @param grid_size_y Grid size in Y dimension  
 * @param grid_size_z Grid size in Z dimension
 */
__kernel void initialize_target_buffer(
    __global double* target_buffer,  // Target buffer to initialize
    const int grid_size_x,          // Grid size in X dimension
    const int grid_size_y,          // Grid size in Y dimension
    const int grid_size_z           // Grid size in Z dimension
) {
    int tid_x = get_global_id(0); // Thread index in X dimension
    int tid_y = get_global_id(1); // Thread index in Y dimension
    int tid_z = get_global_id(2); // Thread index in Z dimension

    // Check bounds
    if (tid_x >= grid_size_x || tid_y >= grid_size_y || tid_z >= grid_size_z) return;

    // Calculate linear index for this grid point
    int target_idx = (tid_z * grid_size_y * grid_size_x + tid_y * grid_size_x + tid_x) * 3;

    // Set the magnetic field components to zero
    target_buffer[target_idx] = 0.0;     // Bx
    target_buffer[target_idx + 1] = 0.0; // By
    target_buffer[target_idx + 2] = 0.0; // Bz
}

'''

# Point pair processing kernel for transformation matrices
PROCESS_POINT_PAIRS_KERNEL = '''//CL//

/**
 * Process line element point pairs to create transformation matrices.
 * 
 * This kernel takes line elements defined by start points (A), end points (B),
 * and currents (I), then computes the transformation matrices needed for the
 * Biot-Savart calculation. Each line element is transformed to a canonical
 * orientation (aligned with z-axis, centered at origin) to enable use of
 * analytical solutions.
 * 
 * Mathematical Process:
 * 1. Compute line element length: L = |B - A|
 * 2. Find midpoint: Mp = (A + B)/2  
 * 3. Calculate direction: d = (B - A)/L
 * 4. Determine rotation to align with z-axis
 * 5. Create transformation matrix M = R * T
 * 6. Store inverse rotation for field transformation
 * 
 * @param input_A Input buffer for start points (3 doubles per element: x,y,z)
 * @param input_B Input buffer for end points (3 doubles per element: x,y,z)
 * @param input_I Input buffer for currents (1 double per element)
 * @param output_M Output transformation matrices (16 doubles per element)
 * @param output_R_inv Output inverse rotation matrices (16 doubles per element)
 * @param output_LI Output length and current pairs (2 doubles per element)
 * @param num_elements Total number of line elements to process
 */
__kernel void process_point_pairs_optimized(
    __global const double* input_A, // Input buffer for A (A_x, A_y, A_z interleaved)
    __global const double* input_B, // Input buffer for B (B_x, B_y, B_z interleaved)
    __global const double* input_I, // Input buffer for currents (I values)
    __global double* output_M,      // Output buffer for transformation matrices M (16 doubles per element)
    __global double* output_R_inv,  // Output buffer for inverse matrices R_inv (16 doubles per element)
    __global double* output_LI,     // Output buffer for lengths L and currents I (2 doubles per element)
    const int num_elements          // Number of elements
) {
    int idx = get_global_id(0); // Parallel thread index
    if (idx >= num_elements) return;

    // Read input data for this element
    const double* A = &input_A[idx * 3];
    const double* B = &input_B[idx * 3];
    double I = input_I[idx];

    // Compute length L = |B - A|
    double dX = B[0] - A[0];
    double dY = B[1] - A[1];
    double dZ = B[2] - A[2];
    double L = sqrt(dX * dX + dY * dY + dZ * dZ);

    // Compute midpoint Mp = (A + B)/2
    double Mp[3] = {
        (A[0] + B[0]) / 2.0,
        (A[1] + B[1]) / 2.0,
        (A[2] + B[2]) / 2.0
    };

    // Compute direction vector d = (B - A)/L
    double d[3] = { dX / L, dY / L, dZ / L };

    // Compute rotation angle and axis to align with z-axis
    // For vector d to align with (0,0,1), rotation axis is d × (0,0,1) = (d_y, -d_x, 0)
    double alpha = acos(d[2]); // Angle between d and z-axis
    double a_rot[3] = { d[1], -d[0], 0.0 }; // Rotation axis

    // Compute rotation matrices R and R_inv
    double R[16], R_inv[16];
    axis_angle_matrix(a_rot, alpha, R);
    axis_angle_matrix(a_rot, -alpha, R_inv);

    // Compute translation matrix T (translate by -Mp to center at origin)
    double T[16];
    double neg_Mp[3] = { -Mp[0], -Mp[1], -Mp[2] };
    translation_matrix(neg_Mp, T);

    // Compute combined transformation matrix M = R * T
    double M[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            M[i * 4 + j] = 0.0;
            for (int k = 0; k < 4; ++k) {
                M[i * 4 + j] += R[i * 4 + k] * T[k * 4 + j];
            }
        }
    }

    // Write results to output buffers
    for (int i = 0; i < 16; ++i) {
        output_M[idx * 16 + i] = M[i];         // Transformation matrix
        output_R_inv[idx * 16 + i] = R_inv[i]; // Inverse rotation matrix
    }

    // Write length and current
    output_LI[idx * 2] = L;      // Element length
    output_LI[idx * 2 + 1] = I;  // Element current
}

'''

# Main Biot-Savart magnetic field calculation kernel
CALCULATE_MAGNETIC_FIELD_KERNEL = '''//CL//

/**
 * Calculate magnetic field using Biot-Savart law with analytical solutions.
 * 
 * This kernel applies the Biot-Savart law to compute magnetic field contributions
 * from all line elements at each grid point. It uses the transformation matrices
 * from process_point_pairs_optimized to work in canonical coordinates where
 * analytical solutions are available.
 * 
 * Mathematical Formula (Biot-Savart Law):
 * dB = (μ₀/4π) * I * (dl × r) / |r|³
 * 
 * For a finite line segment aligned with z-axis from -L/2 to +L/2:
 * Bₓ = (μ₀I/4π) * (y/(x²+y²)) * [t₁ - t₂]
 * Bᵧ = -(μ₀I/4π) * (x/(x²+y²)) * [t₁ - t₂]  
 * Bᵤ = 0
 * 
 * where: t₁ = (z-L/2)/√(x²+y²+(z-L/2)²)
 *        t₂ = (z+L/2)/√(x²+y²+(z+L/2)²)
 * 
 * @param buffer_M Transformation matrices from point pair processing
 * @param buffer_R_inv Inverse rotation matrices for field transformation
 * @param buffer_LI Length and current data for each element
 * @param num_elements Total number of line elements
 * @param target_buffer Output magnetic field buffer (Bx,By,Bz per grid point)
 * @param X_origin Grid origin X coordinate
 * @param Y_origin Grid origin Y coordinate  
 * @param Z_origin Grid origin Z coordinate
 * @param dX Grid spacing in X direction
 * @param dY Grid spacing in Y direction
 * @param dZ Grid spacing in Z direction
 * @param grid_size_x Grid dimensions in X
 * @param grid_size_y Grid dimensions in Y
 * @param grid_size_z Grid dimensions in Z
 */
__kernel void calculate_magnetic_field(
    __global const double* buffer_M,       // Buffer for matrices M (16 doubles per element)
    __global const double* buffer_R_inv,  // Buffer for matrices R_inv (16 doubles per element)
    __global const double* buffer_LI,     // Buffer for lengths L and currents I (2 doubles per element)
    const int num_elements,               // Number of segments

    __global double* target_buffer,       // Target buffer for magnetic field components (3 doubles per grid point)

    const double X_origin, const double Y_origin, const double Z_origin, // Origin of the grid
    const double dX, const double dY, const double dZ,                   // Grid spacing

    const int grid_size_x, const int grid_size_y, const int grid_size_z  // Grid dimensions
) {
    // Get 3D thread indices
    int tid_x = get_global_id(0);
    int tid_y = get_global_id(1);
    int tid_z = get_global_id(2);

    // Check bounds to handle cases where grid size is not a multiple of workgroup size
    if (tid_x >= grid_size_x || tid_y >= grid_size_y || tid_z >= grid_size_z) {
        return;
    }

    // Calculate the grid point position in global coordinate space
    double r_global[3] = {
        X_origin + tid_x * dX,
        Y_origin + tid_y * dY,
        Z_origin + tid_z * dZ
    };

    // Initialize accumulator for magnetic field components
    // Use local memory to minimize global memory access
    double B_global[3] = {0.0, 0.0, 0.0};

    // Iterate over all line segments
    // This allows broadcasting of transformation matrices to all threads
    for (int i = 0; i < num_elements; ++i) {
        // Read transformation matrix M
        const double* M = &buffer_M[i * 16];
       
        // Read inverse rotation matrix R_inv
        const double* R_inv = &buffer_R_inv[i * 16];
        
        // Read length L and current I
        double L = buffer_LI[i * 2];
        double I = buffer_LI[i * 2 + 1];

        // Transform r_global into the local coordinate system of the line element
        double r_local[4] = {0.0, 0.0, 0.0, 1.0};
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                r_local[row] += M[row * 4 + col] * ((col < 3) ? r_global[col] : 1.0);
            }
        }

        // Extract local coordinates
        double x = r_local[0];
        double y = r_local[1];
        double z = r_local[2];

        // Apply Biot-Savart Law in local coordinates
        const double mu_0 = 4.0 * M_PI * 1e-7; // Permeability of free space
        double a2 = x * x + y * y; // Radial distance squared from z-axis
        double B_local[3] = {0.0, 0.0, 0.0};

        if (a2 > 0.0) {
            // Calculate geometric factors for finite line segment
            double t1 = (z - L / 2.0) / sqrt(a2 + (z - L / 2.0) * (z - L / 2.0));
            double t2 = (z + L / 2.0) / sqrt(a2 + (z + L / 2.0) * (z + L / 2.0));
            double factor = (mu_0 * I) / (4.0 * M_PI * a2);

            // Analytical solution for finite line segment
            B_local[0] = factor * y * (t1 - t2);  // Bx component
            B_local[1] = -factor * x * (t1 - t2); // By component  
            B_local[2] = 0.0;                     // Bz (zero in local coordinates)
        } else {
            // Point is on the conductor axis - field calculation would be singular
            // Set field to zero (should be handled by upstream logic)
            B_local[0] = 0.0;
            B_local[1] = 0.0;
            B_local[2] = 0.0;
        }

        // Transform B_local back to global coordinate system using R_inv
        double B_contrib[3] = {0.0, 0.0, 0.0};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                B_contrib[row] += R_inv[row * 4 + col] * B_local[col];
            }
        }

        // Add contribution to global field using superposition principle
        B_global[0] += B_contrib[0];
        B_global[1] += B_contrib[1];
        B_global[2] += B_contrib[2];
    }

    // Write accumulated magnetic field to target buffer
    // Each thread writes to different locations to avoid conflicts
    int target_idx = (tid_z * grid_size_y * grid_size_x + tid_y * grid_size_x + tid_x) * 3;
    target_buffer[target_idx] += B_global[0];     // Bx
    target_buffer[target_idx + 1] += B_global[1]; // By
    target_buffer[target_idx + 2] += B_global[2]; // Bz
}

'''

def register_default_kernels(kernel_registry: KernelRegistry):
    """
    Register all default MAGA kernels with the kernel registry.
    
    Args:
        kernel_registry: KernelRegistry instance to register kernels with
    """
    # Register individual source components
    kernel_registry.register_source("utility_functions", UTILITY_FUNCTIONS)
    kernel_registry.register_source("initialize_target", INITIALIZE_TARGET_KERNEL)
    kernel_registry.register_source("process_point_pairs", PROCESS_POINT_PAIRS_KERNEL)
    kernel_registry.register_source("calculate_magnetic_field", CALCULATE_MAGNETIC_FIELD_KERNEL)
    
    # Compile the main magnetic field calculation program
    kernel_registry.compile_kernel("magnetic_field_program", [
        "utility_functions",
        "initialize_target", 
        "process_point_pairs",
        "calculate_magnetic_field"
    ])