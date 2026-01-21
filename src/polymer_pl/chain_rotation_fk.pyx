import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin, sqrt, atan2, log, exp
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time
from libc.stdint cimport int64_t

ctypedef cnp.int64_t LONG64 
ctypedef Py_ssize_t INDEX_T 

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

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_end_to_end(const double[:] bond_lengths,
                     const double[:] bond_angles_rad,
                     const double[:, :] all_dihedral_angles,
                     int n_repeat_units):
    
    cdef int n_samples = all_dihedral_angles.shape[0]
    cdef int n_bonds_per_unit = bond_lengths.shape[0]
    cdef int n_corr = n_repeat_units + 1

    cdef cnp.ndarray[double, ndim=2] results = np.empty((n_samples, n_corr), dtype=np.float64)
    cdef double[:, :] results_view = results

    cdef int i, j, idx
    cdef double dx, dy, dz

    cdef double[:, :] vecs
    
    for i in range(n_samples):
        vecs = build_full_chain_fk(
            bond_lengths, 
            bond_angles_rad, 
            all_dihedral_angles[i, :], 
            n_repeat_units
        )
        for j in range(n_corr):
            idx = j * n_bonds_per_unit
            dx = vecs[idx, 0]
            dy = vecs[idx, 1]
            dz = vecs[idx, 2]
            results_view[i, j] = dx*dx + dy*dy + dz*dz
    
    return results


# -----------------------------------------------------------------------------
# MULTI-UNIT / COPOLYMER
# -----------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double sample_from_cdf(const double[:] cdf_x, const double[:] cdf_y, double u) noexcept:
    """
    Sample from inverse CDF using linear interpolation.
    cdf_x: x values (angles in degrees)
    cdf_y: CDF values (0 to 1)
    u: uniform random value [0, 1]
    """
    cdef int n = cdf_y.shape[0]
    cdef int i
    
    # Find the interval where u falls
    for i in range(n - 1):
        if cdf_y[i] <= u <= cdf_y[i + 1]:
            # Linear interpolation
            if cdf_y[i + 1] - cdf_y[i] < 1e-10:
                return cdf_x[i]
            return cdf_x[i] + (cdf_x[i + 1] - cdf_x[i]) * (u - cdf_y[i]) / (cdf_y[i + 1] - cdf_y[i])
    
    # Edge cases
    if u <= cdf_y[0]:
        return cdf_x[0]
    return cdf_x[n - 1]

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void generate_dihedral_angles(const int[:] rotation_types,
                                   const int[:, :] rotation_cdf_indices,
                                   const double[:, :] rotation_cdf_x,
                                   const double[:, :] rotation_cdf_y,
                                   const int[:] ris_types,
                                   const int[:, :] ris_angle_indices,
                                   const double[:, :] ris_angles,
                                   const double[:, :] ris_probs,
                                   double[:] output_angles,
                                   unsigned int seed) noexcept:
    """
    Generate dihedral angles for a chain based on rotation and RIS types.
    
    Logic:
    - If rotation_types[i] == 0, check ris_types[i]
      - If ris_types[i] > 0, sample from RIS
      - If ris_types[i] == 0, set angle to 0 (no rotation, no RIS)
    - If rotation_types[i] > 0, sample from rotation CDF (index is rotation_types[i] - 1)
    
    Parameters:
    -----------
    rotation_types : array of rotation type IDs (0 means check RIS, >0 means use rotation CDF)
    rotation_cdf_indices : (n_rot_types, 2) start and end indices for each rotation type's CDF
    rotation_cdf_x : (max_cdf_size, n_rot_types) CDF x values (angles in degrees)
    rotation_cdf_y : (max_cdf_size, n_rot_types) CDF y values
    ris_types : array of RIS type IDs (0 means no RIS, >0 means use RIS, index is ris_types[i] - 1)
    ris_angle_indices : (n_ris_types, 2) start and end indices for each RIS type
    ris_angles : (max_ris_size, n_ris_types) RIS angles in degrees
    ris_probs : (max_ris_size, n_ris_types) RIS probabilities
    output_angles : output array for angles in radians
    seed : random seed
    """
    cdef int n_bonds = rotation_types.shape[0]
    cdef int i, j, rot_id, ris_id, start, end, n_states
    cdef double u, cumsum, angle_deg
    cdef double pi = 3.14159265358979323846
    
    srand(seed)
    
    for i in range(n_bonds):
        rot_id = rotation_types[i]
        
        # Check if rotation_type is 0
        if rot_id == 0:
            # Check for RIS
            if ris_types[i] > 0:
                # Use RIS (ris_types[i] - 1 is the actual index)
                u = <double>rand() / <double>RAND_MAX
                ris_id = ris_types[i] - 1
                start = ris_angle_indices[ris_id, 0]
                end = ris_angle_indices[ris_id, 1]
                n_states = end - start
                
                # Sample from discrete RIS distribution
                cumsum = 0.0
                angle_deg = ris_angles[start, ris_id]  # default to first angle
                for j in range(n_states):
                    cumsum += ris_probs[start + j, ris_id]
                    if u <= cumsum:
                        angle_deg = ris_angles[start + j, ris_id]
                        break
            else:
                # ris_types[i] == 0: no rotation, no RIS, fixed at 0 degrees
                angle_deg = 0.0
        else:
            # Use rotation CDF (rot_id - 1 is the actual index)
            u = <double>rand() / <double>RAND_MAX
            rot_id = rot_id - 1
            start = rotation_cdf_indices[rot_id, 0]
            end = rotation_cdf_indices[rot_id, 1]
            
            angle_deg = sample_from_cdf(rotation_cdf_x[start:end, rot_id],
                                       rotation_cdf_y[start:end, rot_id],
                                       u)
        
        output_angles[i] = angle_deg * pi / 180.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void build_chain_unit_vectors_fk_multi(const double[:] bond_lengths,
                                            const double[:] bond_angles_rad,
                                            const double[:] dihedral_angles_rad,
                                            const int[:] unit_end_indices,
                                            int n_repeat_units,
                                            double[:, :] unit_vectors) noexcept:
    """
    Build chain using forward kinematics and return unit vectors for variable-length units.
    
    Parameters:
    -----------
    bond_lengths : array of all bond lengths
    bond_angles_rad : array of all bond angles
    dihedral_angles_rad : array of all dihedral angles
    unit_end_indices : (n_repeat_units + 1,) indices marking end of each unit
    n_repeat_units : number of repeat units
    unit_vectors : (n_repeat_units, 3) output array for unit vectors
    """
    cdef int total_bonds = bond_lengths.shape[0]
    
    # Working memory
    cdef double[:, :] R_global = np.eye(3, dtype=np.float64)
    cdef double[:, :] R_local = np.empty((3, 3), dtype=np.float64)
    cdef double[:, :] R_temp = np.empty((3, 3), dtype=np.float64)
    cdef double[:] bond_vec_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    cdef double[:] bond_vec_global = np.empty(3, dtype=np.float64)
    cdef double[:] unit_sum = np.zeros(3, dtype=np.float64)
    
    cdef int bond_idx, unit_idx, unit_start, unit_end
    cdef double bond_length, bond_angle, dihedral_angle
    cdef double norm
    
    # Initialize: first bond along x-axis
    R_global[0, 0] = 1.0; R_global[0, 1] = 0.0; R_global[0, 2] = 0.0
    R_global[1, 0] = 0.0; R_global[1, 1] = 1.0; R_global[1, 2] = 0.0
    R_global[2, 0] = 0.0; R_global[2, 1] = 0.0; R_global[2, 2] = 1.0
    
    unit_idx = 0
    unit_start = unit_end_indices[0]
    unit_end = unit_end_indices[1]
    
    for bond_idx in range(total_bonds):
        bond_length = bond_lengths[bond_idx]
        bond_angle = bond_angles_rad[bond_idx]
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
        
        # Check if we reached the end of current unit
        if bond_idx == unit_end - 1:
            # Normalize and store
            norm = sqrt(unit_sum[0]*unit_sum[0] + unit_sum[1]*unit_sum[1] + unit_sum[2]*unit_sum[2])
            if norm > 1e-10:
                unit_vectors[unit_idx, 0] = unit_sum[0] / norm
                unit_vectors[unit_idx, 1] = unit_sum[1] / norm
                unit_vectors[unit_idx, 2] = unit_sum[2] / norm
            else:
                unit_vectors[unit_idx, 0] = 1.0
                unit_vectors[unit_idx, 1] = 0.0
                unit_vectors[unit_idx, 2] = 0.0
            
            # Reset for next repeat unit
            unit_sum[0] = 0.0
            unit_sum[1] = 0.0
            unit_sum[2] = 0.0
            
            # Move to next unit
            unit_idx += 1
            if unit_idx < n_repeat_units:
                unit_start = unit_end_indices[unit_idx]
                unit_end = unit_end_indices[unit_idx + 1]

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_correlation_fk_multi(bond_lengths_list,
                               bond_angles_rad_list,
                               const double[:] unit_probs,
                               const int[:, :] rotation_cdf_indices,
                               const double[:, :] rotation_cdf_x,
                               const double[:, :] rotation_cdf_y,
                               rotation_types_list,
                               ris_types_list,
                               const int[:, :] ris_angle_indices,
                               const double[:, :] ris_angles,
                               const double[:, :] ris_probs,
                               int n_samples,
                               int n_repeat_units):
    """
    Batch correlation calculation for multi-component polymer chains.
    
    Parameters:
    -----------
    bond_lengths_list : list of arrays, each array is bond lengths for one unit type
    bond_angles_rad_list : list of arrays, bond angles in radians for each unit type
    unit_probs : probabilities for selecting each unit type
    rotation_cdf_indices : (n_rot_types, 2) indices for CDF lookup
    rotation_cdf_x, rotation_cdf_y : CDF data for rotation sampling
    rotation_types_list : list of arrays, rotation type IDs for each unit type
    ris_types_list : list of arrays or None, RIS type IDs for each unit type
    ris_angle_indices : (n_ris_types, 2) indices for RIS lookup
    ris_angles, ris_probs : RIS angle data
    n_samples : number of chains to generate
    n_repeat_units : number of repeat units per chain
    
    Returns:
    --------
    correlations : (n_samples, n_repeat_units - 1)
    """
    cdef int n_unit_types = len(bond_lengths_list)
    cdef int max_bonds_per_unit = max([len(bl) for bl in bond_lengths_list])
    cdef int max_total_bonds = max_bonds_per_unit * n_repeat_units
    cdef int n_corr = n_repeat_units - 1
    
    # Output array
    cdef cnp.ndarray[double, ndim=2] results = np.empty((n_samples, n_corr), dtype=np.float64)
    cdef double[:, :] results_view = results
    
    # Prepare unit type data
    cdef cnp.ndarray[double, ndim=2] all_bond_lengths = np.zeros((n_unit_types, max_bonds_per_unit), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] all_bond_angles = np.zeros((n_unit_types, max_bonds_per_unit), dtype=np.float64)
    cdef cnp.ndarray[int, ndim=2] all_rotation_types = np.zeros((n_unit_types, max_bonds_per_unit), dtype=np.int32)
    cdef cnp.ndarray[int, ndim=2] all_ris_types = np.full((n_unit_types, max_bonds_per_unit), -1, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1] bonds_per_unit = np.zeros(n_unit_types, dtype=np.int32)
    
    cdef int i, j, n_bonds
    for i in range(n_unit_types):
        n_bonds = len(bond_lengths_list[i])
        bonds_per_unit[i] = n_bonds
        all_bond_lengths[i, :n_bonds] = bond_lengths_list[i]
        all_bond_angles[i, :n_bonds] = bond_angles_rad_list[i]
        all_rotation_types[i, :n_bonds] = rotation_types_list[i]
        if ris_types_list is not None:
            all_ris_types[i, :n_bonds] = ris_types_list[i]
    
    # Build CDF for unit selection
    cdef cnp.ndarray[double, ndim=1] unit_cdf = np.cumsum(unit_probs)
    
    # Working arrays
    cdef cnp.ndarray[int, ndim=1] selected_units = np.empty(n_repeat_units, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1] unit_end_indices = np.empty(n_repeat_units + 1, dtype=np.int32)
    cdef cnp.ndarray[double, ndim=1] chain_bond_lengths = np.empty(max_total_bonds, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] chain_bond_angles = np.empty(max_total_bonds, dtype=np.float64)
    cdef cnp.ndarray[int, ndim=1] chain_rotation_types = np.empty(max_total_bonds, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1] chain_ris_types = np.empty(max_total_bonds, dtype=np.int32)
    cdef cnp.ndarray[double, ndim=1] dihedral_angles = np.empty(max_total_bonds, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] unit_vecs = np.empty((n_repeat_units, 3), dtype=np.float64)
    
    cdef int sample_idx, unit_idx, unit_type, bond_idx, total_bonds
    cdef double u, dot_prod
    cdef unsigned int seed
    
    srand(<unsigned int>time(NULL))
    
    for sample_idx in range(n_samples):
        # Select random units for this chain and build unit_end_indices
        total_bonds = 0
        unit_end_indices[0] = 0
        
        for unit_idx in range(n_repeat_units):
            u = <double>rand() / <double>RAND_MAX
            for j in range(n_unit_types):
                if u <= unit_cdf[j]:
                    selected_units[unit_idx] = j
                    break
            
            unit_type = selected_units[unit_idx]
            n_bonds = bonds_per_unit[unit_type]
            
            # Copy unit data to chain
            for bond_idx in range(n_bonds):
                chain_bond_lengths[total_bonds] = all_bond_lengths[unit_type, bond_idx]
                chain_bond_angles[total_bonds] = all_bond_angles[unit_type, bond_idx]
                chain_rotation_types[total_bonds] = all_rotation_types[unit_type, bond_idx]
                chain_ris_types[total_bonds] = all_ris_types[unit_type, bond_idx]
                total_bonds += 1
            
            unit_end_indices[unit_idx + 1] = total_bonds
        
        # Generate dihedral angles
        seed = <unsigned int>(sample_idx * 123456 + time(NULL) % 1000000)
        generate_dihedral_angles(chain_rotation_types[:total_bonds],
                                rotation_cdf_indices,
                                rotation_cdf_x,
                                rotation_cdf_y,
                                chain_ris_types[:total_bonds],
                                ris_angle_indices,
                                ris_angles,
                                ris_probs,
                                dihedral_angles[:total_bonds],
                                seed)
        
        # Build chain unit vectors using variable-length units
        build_chain_unit_vectors_fk_multi(chain_bond_lengths[:total_bonds],
                                         chain_bond_angles[:total_bonds],
                                         dihedral_angles[:total_bonds],
                                         unit_end_indices,
                                         n_repeat_units,
                                         unit_vecs)
        
        # Compute correlation from unit vectors
        for i in range(1, n_repeat_units):
            dot_prod = 0.0
            for j in range(3):
                dot_prod += unit_vecs[0, j] * unit_vecs[i, j]
            results_view[sample_idx, i - 1] = dot_prod
    
    return results

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_end_to_end_multi(bond_lengths_list,
                          bond_angles_rad_list,
                          const double[:] unit_probs,
                          const int[:, :] rotation_cdf_indices,
                          const double[:, :] rotation_cdf_x,
                          const double[:, :] rotation_cdf_y,
                          rotation_types_list,
                          ris_types_list,
                          const int[:, :] ris_angle_indices,
                          const double[:, :] ris_angles,
                          const double[:, :] ris_probs,
                          int n_samples,
                          int n_repeat_units):
    """
    Batch end-to-end distance calculation for multi-component polymer chains.
    """
    cdef int n_unit_types = len(bond_lengths_list)
    cdef int max_bonds_per_unit = max([len(bl) for bl in bond_lengths_list])
    cdef int max_total_bonds = max_bonds_per_unit * n_repeat_units
    cdef int n_points = n_repeat_units + 1
    
    # Output array
    cdef cnp.ndarray[double, ndim=2] results = np.empty((n_samples, n_points), dtype=np.float64)
    cdef double[:, :] results_view = results
    
    # Prepare unit type data (same as batch_correlation_fk_multi)
    cdef cnp.ndarray[double, ndim=2] all_bond_lengths = np.zeros((n_unit_types, max_bonds_per_unit), dtype=np.float64)
    cdef cnp.ndarray[double, ndim=2] all_bond_angles = np.zeros((n_unit_types, max_bonds_per_unit), dtype=np.float64)
    cdef cnp.ndarray[int, ndim=2] all_rotation_types = np.zeros((n_unit_types, max_bonds_per_unit), dtype=np.int32)
    cdef cnp.ndarray[int, ndim=2] all_ris_types = np.full((n_unit_types, max_bonds_per_unit), -1, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1] bonds_per_unit = np.zeros(n_unit_types, dtype=np.int32)
    
    cdef int i, j, n_bonds
    for i in range(n_unit_types):
        n_bonds = len(bond_lengths_list[i])
        bonds_per_unit[i] = n_bonds
        all_bond_lengths[i, :n_bonds] = bond_lengths_list[i]
        all_bond_angles[i, :n_bonds] = bond_angles_rad_list[i]
        all_rotation_types[i, :n_bonds] = rotation_types_list[i]
        if ris_types_list is not None:
            all_ris_types[i, :n_bonds] = ris_types_list[i]
    
    cdef cnp.ndarray[double, ndim=1] unit_cdf = np.cumsum(unit_probs)
    
    # Working arrays
    cdef cnp.ndarray[int, ndim=1] selected_units = np.empty(n_repeat_units, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1] unit_end_indices = np.empty(n_repeat_units + 1, dtype=np.int32)
    cdef cnp.ndarray[double, ndim=1] chain_bond_lengths = np.empty(max_total_bonds, dtype=np.float64)
    cdef cnp.ndarray[double, ndim=1] chain_bond_angles = np.empty(max_total_bonds, dtype=np.float64)
    cdef cnp.ndarray[int, ndim=1] chain_rotation_types = np.empty(max_total_bonds, dtype=np.int32)
    cdef cnp.ndarray[int, ndim=1] chain_ris_types = np.empty(max_total_bonds, dtype=np.int32)
    cdef cnp.ndarray[double, ndim=1] dihedral_angles = np.empty(max_total_bonds, dtype=np.float64)
    
    cdef int sample_idx, unit_idx, unit_type, bond_idx, total_bonds, idx
    cdef double u, dx, dy, dz
    cdef unsigned int seed
    cdef cnp.ndarray[double, ndim=2] positions
    
    srand(<unsigned int>time(NULL))
    
    for sample_idx in range(n_samples):
        # Select random units and build chain
        total_bonds = 0
        unit_end_indices[0] = 0
        
        for unit_idx in range(n_repeat_units):
            u = <double>rand() / <double>RAND_MAX
            for j in range(n_unit_types):
                if u <= unit_cdf[j]:
                    selected_units[unit_idx] = j
                    break
            
            unit_type = selected_units[unit_idx]
            n_bonds = bonds_per_unit[unit_type]
            
            for bond_idx in range(n_bonds):
                chain_bond_lengths[total_bonds] = all_bond_lengths[unit_type, bond_idx]
                chain_bond_angles[total_bonds] = all_bond_angles[unit_type, bond_idx]
                chain_rotation_types[total_bonds] = all_rotation_types[unit_type, bond_idx]
                chain_ris_types[total_bonds] = all_ris_types[unit_type, bond_idx]
                total_bonds += 1
            
            unit_end_indices[unit_idx + 1] = total_bonds
        
        # Generate dihedral angles
        seed = <unsigned int>(sample_idx * 123456 + time(NULL) % 1000000)
        generate_dihedral_angles(chain_rotation_types[:total_bonds],
                                rotation_cdf_indices,
                                rotation_cdf_x,
                                rotation_cdf_y,
                                chain_ris_types[:total_bonds],
                                ris_angle_indices,
                                ris_angles,
                                ris_probs,
                                dihedral_angles[:total_bonds],
                                seed)
        
        # Build full chain
        positions = build_full_chain_fk(chain_bond_lengths[:total_bonds],
                                       chain_bond_angles[:total_bonds],
                                       dihedral_angles[:total_bonds],
                                       n_repeat_units)
        
        # Calculate R² at unit endpoints
        for j in range(n_points):
            idx = unit_end_indices[j]
            dx = positions[idx, 0]
            dy = positions[idx, 1]
            dz = positions[idx, 2]
            results_view[sample_idx, j] = dx*dx + dy*dy + dz*dz
    
    return results

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def build_chain_copolymer_cy(
#     cnp.ndarray[cnp.double_t, ndim=1] all_l,
#     cnp.ndarray[cnp.double_t, ndim=1] theta,
#     cnp.ndarray[cnp.int64_t, ndim=1] unit_lengths,
#     cnp.ndarray[cnp.double_t, ndim=1] dihedrals
# ):
#     cdef Py_ssize_t n_bonds = all_l.shape[0]
#     cdef Py_ssize_t i, j, k

#     cdef cnp.ndarray[cnp.double_t, ndim=2] vectors = np.zeros((n_bonds, 3), dtype=np.float64)

#     cdef cnp.ndarray[cnp.double_t, ndim=2] coords_full = np.zeros((n_bonds + 1, 3), dtype=np.float64)

#     cdef double c_phi, s_phi, c_theta, s_theta
#     cdef double R[3][3]
#     cdef double R_step[3][3]
#     cdef double tmp[3][3]

#     R[0][0] = 1.0; R[0][1] = 0.0; R[0][2] = 0.0
#     R[1][0] = 0.0; R[1][1] = 1.0; R[1][2] = 0.0
#     R[2][0] = 0.0; R[2][1] = 0.0; R[2][2] = 1.0

#     for j in range(n_bonds):
#         c_phi = cos(dihedrals[j])
#         s_phi = sin(dihedrals[j])
#         c_theta = cos(theta[j])
#         s_theta = sin(theta[j])

#         R_step[0][0] = c_theta
#         R_step[1][0] = s_theta
#         R_step[2][0] = 0.0

#         R_step[0][1] = -s_theta * c_phi
#         R_step[1][1] =  c_theta * c_phi
#         R_step[2][1] =  s_phi

#         R_step[0][2] =  s_theta * s_phi
#         R_step[1][2] = -c_theta * s_phi
#         R_step[2][2] =  c_phi

#         for i in range(3):
#             for k in range(3):
#                 tmp[i][k] = (
#                     R[i][0] * R_step[0][k] +
#                     R[i][1] * R_step[1][k] +
#                     R[i][2] * R_step[2][k]
#                 )

#         for i in range(3):
#             for k in range(3):
#                 R[i][k] = tmp[i][k]

#         vectors[j, 0] = R[0][0] * all_l[j]
#         vectors[j, 1] = R[1][0] * all_l[j]
#         vectors[j, 2] = R[2][0] * all_l[j]

#     for j in range(n_bonds):
#         coords_full[j + 1, 0] = coords_full[j, 0] + vectors[j, 0]
#         coords_full[j + 1, 1] = coords_full[j, 1] + vectors[j, 1]
#         coords_full[j + 1, 2] = coords_full[j, 2] + vectors[j, 2]

#     cdef cnp.ndarray[cnp.int64_t, ndim=1] end_idx
#     end_idx = np.concatenate(([0], np.cumsum(unit_lengths)))

#     return coords_full[end_idx]


@cython.boundscheck(False)
@cython.wraparound(False)
def build_chain_copolymer_cy(
    const double[:] all_l not None,
    const double[:] theta not None,
    const LONG64[:] unit_lengths not None,
    const double[:] dihedrals not None
):
    cdef INDEX_T n_bonds = all_l.shape[0]
    cdef INDEX_T n_units = unit_lengths.shape[0]

    cdef double[:, ::1] result = np.zeros((n_units + 1, 3), dtype=np.float64)
    
    cdef INDEX_T i, j = 0
    cdef LONG64 current_idx = 0

    cdef double R00 = 1.0, R01 = 0.0, R02 = 0.0
    cdef double R10 = 0.0, R11 = 1.0, R12 = 0.0
    cdef double R20 = 0.0, R21 = 0.0, R22 = 1.0

    cdef double px = 0.0, py = 0.0, pz = 0.0

    result[0, 0] = px
    result[0, 1] = py
    result[0, 2] = pz

    cdef double c_phi, s_phi, c_theta, s_theta
    cdef double bond_length
    cdef double S00, S01, S02, S10, S11, S12, S20, S21, S22
    cdef double T00, T01, T02, T10, T11, T12, T20, T21, T22

    for i in range(n_units):
        for _ in range(unit_lengths[i]):

            c_phi = cos(dihedrals[j])
            s_phi = sin(dihedrals[j])
            c_theta = cos(theta[j])
            s_theta = sin(theta[j])
            bond_length = all_l[j]

            S00 = c_theta
            S10 = s_theta
            S20 = 0.0
            S01 = -s_theta * c_phi
            S11 = c_theta * c_phi
            S21 = s_phi
            S02 = s_theta * s_phi
            S12 = -c_theta * s_phi
            S22 = c_phi

            T00 = R00 * S00 + R01 * S10 + R02 * S20
            T01 = R00 * S01 + R01 * S11 + R02 * S21
            T02 = R00 * S02 + R01 * S12 + R02 * S22
            T10 = R10 * S00 + R11 * S10 + R12 * S20
            T11 = R10 * S01 + R11 * S11 + R12 * S21
            T12 = R10 * S02 + R11 * S12 + R12 * S22
            T20 = R20 * S00 + R21 * S10 + R22 * S20
            T21 = R20 * S01 + R21 * S11 + R22 * S21
            T22 = R20 * S02 + R21 * S12 + R22 * S22

            R00, R01, R02 = T00, T01, T02
            R10, R11, R12 = T10, T11, T12
            R20, R21, R22 = T20, T21, T22

            px += R00 * bond_length
            py += R10 * bond_length
            pz += R20 * bond_length
            
            j += 1

        result[i + 1, 0] = px
        result[i + 1, 1] = py
        result[i + 1, 2] = pz

    return np.asarray(result)