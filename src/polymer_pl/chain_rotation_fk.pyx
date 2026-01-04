"""
Optimized version using Forward Kinematics approach.
Key improvements:
1. Direct chain construction using rotation matrices
2. Local coordinate frame transformations
3. Compute only unit vectors for correlation (no full positions needed)
4. Vectorized operations where possible
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin, sqrt, atan2

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void build_local_rotation_matrix(double theta, double phi, double[:, :] R) noexcept:
    """
    Build local rotation matrix from bond angle theta and dihedral angle phi.
    This transforms from the previous local frame to the current local frame.
    
    Convention: 
    - theta: bond angle (deviation from straight)
    - phi: dihedral/torsion angle
    """
    cdef double cos_theta = cos(theta)
    cdef double sin_theta = sin(theta)
    cdef double cos_phi = cos(phi)
    cdef double sin_phi = sin(phi)
    
    # Rotation matrix: first rotate by phi around z-axis, then by theta around y-axis
    R[0, 0] = cos_theta
    R[0, 1] = -sin_theta * cos_phi
    R[0, 2] = sin_theta * sin_phi
    R[1, 0] = sin_theta
    R[1, 1] = cos_theta * cos_phi
    R[1, 2] = -cos_theta * sin_phi
    R[2, 0] = 0.0
    R[2, 1] = sin_phi
    R[2, 2] = cos_phi

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void matmul_3x3(const double[:, :] A, const double[:, :] B, double[:, :] C) noexcept:
    """Fast 3x3 matrix multiplication: C = A @ B"""
    cdef int i, j, k
    for i in range(3):
        for j in range(3):
            C[i, j] = 0.0
            for k in range(3):
                C[i, j] += A[i, k] * B[k, j]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void matvec_3x3(const double[:, :] A, const double[:] v, double[:] result) noexcept:
    """Fast 3x3 matrix-vector multiplication: result = A @ v"""
    cdef int i, j
    for i in range(3):
        result[i] = 0.0
        for j in range(3):
            result[i] += A[i, j] * v[j]

@cython.boundscheck(False)
@cython.wraparound(False)
def build_chain_unit_vectors_fk(const double[:] bond_lengths,
                                 const double[:] bond_angles_rad,
                                 const double[:] dihedral_angles_rad,
                                 int n_repeat_units):
    """
    Build chain using forward kinematics and return unit vectors.
    This is the core optimization: we directly compute unit vectors without
    storing full positions (unless needed for visualization).
    
    Parameters:
    -----------
    bond_lengths : array of length n_bonds_per_unit
    bond_angles_rad : array of length n_bonds_per_unit (bond angles)
    dihedral_angles_rad : array of length n_total_bonds (pre-generated)
    n_repeat_units : number of repeat units
    
    Returns:
    --------
    unit_vectors : (n_repeat_units, 3) array of unit vectors for each repeat unit
    """
    cdef int n_bonds_per_unit = bond_lengths.shape[0]
    cdef int total_bonds = n_bonds_per_unit * n_repeat_units
    
    # Output: one unit vector per repeat unit
    cdef cnp.ndarray[double, ndim=2] unit_vectors = np.zeros((n_repeat_units, 3), dtype=np.float64)
    cdef double[:, :] uv_view = unit_vectors
    
    # Working memory
    cdef double[:, :] R_global = np.eye(3, dtype=np.float64)  # Global rotation matrix
    cdef double[:, :] R_local = np.empty((3, 3), dtype=np.float64)  # Local rotation
    cdef double[:, :] R_temp = np.empty((3, 3), dtype=np.float64)  # Temp for matmul
    cdef double[:] bond_vec_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # Local bond vector (x-axis)
    cdef double[:] bond_vec_global = np.empty(3, dtype=np.float64)
    cdef double[:] unit_sum = np.zeros(3, dtype=np.float64)  # Sum of bonds in current repeat unit
    
    cdef int bond_idx, unit_idx, local_bond_idx
    cdef double bond_length, bond_angle, dihedral_angle
    cdef double norm
    
    # Initialize: first bond along x-axis
    R_global[0, 0] = 1.0; R_global[0, 1] = 0.0; R_global[0, 2] = 0.0
    R_global[1, 0] = 0.0; R_global[1, 1] = 1.0; R_global[1, 2] = 0.0
    R_global[2, 0] = 0.0; R_global[2, 1] = 0.0; R_global[2, 2] = 1.0
    
    for bond_idx in range(total_bonds):
        unit_idx = bond_idx // n_bonds_per_unit
        local_bond_idx = bond_idx % n_bonds_per_unit
        
        bond_length = bond_lengths[local_bond_idx]
        bond_angle = bond_angles_rad[local_bond_idx]
        dihedral_angle = dihedral_angles_rad[bond_idx]
        
        # Build local rotation matrix
        build_local_rotation_matrix(bond_angle, dihedral_angle, R_local)
        
        # Update global rotation: R_global = R_global @ R_local
        matmul_3x3(R_global, R_local, R_temp)
        R_global[:] = R_temp
        
        # Transform local bond vector to global frame
        bond_vec_local[0] = bond_length
        bond_vec_local[1] = 0.0
        bond_vec_local[2] = 0.0
        matvec_3x3(R_global, bond_vec_local, bond_vec_global)
        
        # Accumulate for this repeat unit
        unit_sum[0] += bond_vec_global[0]
        unit_sum[1] += bond_vec_global[1]
        unit_sum[2] += bond_vec_global[2]
        
        # At the end of each repeat unit, normalize and store
        if local_bond_idx == n_bonds_per_unit - 1:
            norm = sqrt(unit_sum[0]*unit_sum[0] + unit_sum[1]*unit_sum[1] + unit_sum[2]*unit_sum[2])
            uv_view[unit_idx, 0] = unit_sum[0] / norm
            uv_view[unit_idx, 1] = unit_sum[1] / norm
            uv_view[unit_idx, 2] = unit_sum[2] / norm
            
            # Reset for next repeat unit
            unit_sum[0] = 0.0
            unit_sum[1] = 0.0
            unit_sum[2] = 0.0
    
    return unit_vectors

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_correlation_from_unit_vectors(const double[:, :] unit_vectors):
    """
    Compute correlation function <v_0 · v_n> from unit vectors.
    
    Parameters:
    -----------
    unit_vectors : (n_repeat_units, 3) array
    
    Returns:
    --------
    correlations : (n_repeat_units - 1,) array
    """
    cdef int n_units = unit_vectors.shape[0]
    cdef cnp.ndarray[double, ndim=1] corr = np.empty(n_units - 1, dtype=np.float64)
    cdef double[:] corr_view = corr
    
    cdef int i, j
    cdef double dot_prod
    
    # Reference vector is the first unit vector
    for i in range(1, n_units):
        dot_prod = 0.0
        for j in range(3):
            dot_prod += unit_vectors[0, j] * unit_vectors[i, j]
        corr_view[i - 1] = dot_prod
    
    return corr

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_correlation_fk(const double[:] bond_lengths,
                         const double[:] bond_angles_rad,
                         const double[:, :] all_dihedral_angles,
                         int n_repeat_units):
    """
    Batch process multiple samples using forward kinematics.
    
    Parameters:
    -----------
    bond_lengths : (n_bonds_per_unit,)
    bond_angles_rad : (n_bonds_per_unit,)
    all_dihedral_angles : (n_samples, total_bonds) in radians
    n_repeat_units : int
    
    Returns:
    --------
    correlations : (n_samples, n_repeat_units - 1)
    """
    cdef int n_samples = all_dihedral_angles.shape[0]
    cdef int n_corr = n_repeat_units - 1
    
    cdef cnp.ndarray[double, ndim=2] results = np.empty((n_samples, n_corr), dtype=np.float64)
    cdef double[:, :] results_view = results
    
    cdef int i, j
    cdef cnp.ndarray[double, ndim=2] unit_vecs
    cdef cnp.ndarray[double, ndim=1] corr
    
    for i in range(n_samples):
        unit_vecs = build_chain_unit_vectors_fk(bond_lengths, bond_angles_rad, 
                                                all_dihedral_angles[i, :], n_repeat_units)
        corr = compute_correlation_from_unit_vectors(unit_vecs)
        
        for j in range(n_corr):
            results_view[i, j] = corr[j]
    
    return results

# Optional: Full chain construction for visualization
@cython.boundscheck(False)
@cython.wraparound(False)
def build_full_chain_fk(const double[:] bond_lengths,
                        const double[:] bond_angles_rad,
                        const double[:] dihedral_angles_rad,
                        int n_repeat_units):
    """
    Build full chain positions using forward kinematics.
    Only use this when you need actual positions (e.g., for visualization).
    """
    cdef int n_bonds_per_unit = bond_lengths.shape[0]
    cdef int total_bonds = n_bonds_per_unit * n_repeat_units
    
    cdef cnp.ndarray[double, ndim=2] positions = np.zeros((total_bonds + 1, 3), dtype=np.float64)
    cdef double[:, :] pos_view = positions
    
    cdef double[:, :] R_global = np.eye(3, dtype=np.float64)
    cdef double[:, :] R_local = np.empty((3, 3), dtype=np.float64)
    cdef double[:, :] R_temp = np.empty((3, 3), dtype=np.float64)
    cdef double[:] bond_vec_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cdef double[:] bond_vec_global = np.empty(3, dtype=np.float64)
    cdef double[:] current_pos = np.zeros(3, dtype=np.float64)
    
    cdef int bond_idx, local_bond_idx
    cdef double bond_length, bond_angle, dihedral_angle
    
    for bond_idx in range(total_bonds):
        local_bond_idx = bond_idx % n_bonds_per_unit
        
        bond_length = bond_lengths[local_bond_idx]
        bond_angle = bond_angles_rad[local_bond_idx]
        dihedral_angle = dihedral_angles_rad[bond_idx]
        
        build_local_rotation_matrix(bond_angle, dihedral_angle, R_local)
        matmul_3x3(R_global, R_local, R_temp)
        R_global[:] = R_temp
        
        bond_vec_local[0] = bond_length
        matvec_3x3(R_global, bond_vec_local, bond_vec_global)
        
        current_pos[0] += bond_vec_global[0]
        current_pos[1] += bond_vec_global[1]
        current_pos[2] += bond_vec_global[2]
        
        pos_view[bond_idx + 1, 0] = current_pos[0]
        pos_view[bond_idx + 1, 1] = current_pos[1]
        pos_view[bond_idx + 1, 2] = current_pos[2]
    
    return positions
