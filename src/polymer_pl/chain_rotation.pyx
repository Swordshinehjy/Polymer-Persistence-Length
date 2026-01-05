
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport cos, sin, sqrt

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void rotation_matrix_fast(const double[:] axis, double angle, double[:, :] rot_mat) noexcept:
    """Fast rotation matrix calculation without GIL."""
    cdef double norm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
    cdef double ax = axis[0] / norm
    cdef double ay = axis[1] / norm
    cdef double az = axis[2] / norm

    cdef double half_angle = angle / 2.0
    cdef double sin_half = sin(half_angle)

    cdef double a = cos(angle / 2.0)
    cdef double b = -ax * sin_half
    cdef double c = -ay * sin_half
    cdef double d = -az * sin_half

    cdef double aa = a * a
    cdef double bb = b * b
    cdef double cc = c * c
    cdef double dd = d * d
    cdef double bc = b * c
    cdef double ad = a * d
    cdef double ac = a * c
    cdef double ab = a * b
    cdef double bd = b * d
    cdef double cd = c * d

    rot_mat[0, 0] = aa + bb - cc - dd
    rot_mat[0, 1] = 2 * (bc + ad)
    rot_mat[0, 2] = 2 * (bd - ac)
    rot_mat[1, 0] = 2 * (bc - ad)
    rot_mat[1, 1] = aa + cc - bb - dd
    rot_mat[1, 2] = 2 * (cd + ab)
    rot_mat[2, 0] = 2 * (bd + ac)
    rot_mat[2, 1] = 2 * (cd - ab)
    rot_mat[2, 2] = aa + dd - bb - cc

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void apply_rotation(double[:, :] pts, int start_idx, const double[:, :] rot_mat, 
                        const double[:] pivot) noexcept:
    """Apply rotation matrix to points starting from start_idx."""
    cdef int i, j, k
    cdef double temp[3]
    cdef int n_pts = pts.shape[0]

    for i in range(start_idx, n_pts):
        # Translate to pivot
        for j in range(3):
            temp[j] = pts[i, j] - pivot[j]

        # Apply rotation
        for j in range(3):
            pts[i, j] = pivot[j]
            for k in range(3):
                pts[i, j] += rot_mat[j, k] * temp[k]

@cython.boundscheck(False)
@cython.wraparound(False)
def randomRotate_cython(const double[:, :] base_pts, const double[:] angles):
    """Cythonized version of randomRotate_vectorized."""
    cdef int n_pts = base_pts.shape[0]
    cdef int n_angles = angles.shape[0]
    cdef cnp.ndarray[double, ndim=2] pts = np.copy(base_pts)
    cdef double[:, :] pts_view = pts
    cdef double[:] vec = np.empty(3, dtype=np.float64)
    cdef double[:, :] rot_mat = np.empty((3, 3), dtype=np.float64)
    cdef double[:] pivot = np.empty(3, dtype=np.float64)
    cdef int idx
    cdef double angle_rad

    for idx in range(n_angles):
        if angles[idx] != 0:
            # Calculate axis vector
            vec[0] = pts_view[idx, 0] - pts_view[idx - 1, 0]
            vec[1] = pts_view[idx, 1] - pts_view[idx - 1, 1]
            vec[2] = pts_view[idx, 2] - pts_view[idx - 1, 2]

            # Store pivot point
            pivot[0] = pts_view[idx, 0]
            pivot[1] = pts_view[idx, 1]
            pivot[2] = pts_view[idx, 2]

            # Convert angle to radians
            angle_rad = angles[idx] * 3.141592653589793 / 180.0

            # Compute rotation matrix
            rotation_matrix_fast(vec, angle_rad, rot_mat)

            # Apply rotation to all subsequent points
            apply_rotation(pts_view, idx + 1, rot_mat, pivot)

    return pts

@cython.boundscheck(False)
@cython.wraparound(False)
def cosVals_cython(const double[:, :] pts, int length):
    """Cythonized version of cosVals."""
    cdef int n_pts = pts.shape[0]
    cdef int n_values = (n_pts - length) // length
    cdef cnp.ndarray[double, ndim=1] cos_vals = np.empty(n_values, dtype=np.float64)
    cdef double[:] cos_view = cos_vals

    cdef double[:] v2 = np.empty(3, dtype=np.float64)
    cdef double dot_prod, norm_v, comp, cos_val
    cdef int i, k, j

    # First vector (reference)
    v2[0] = pts[length, 0] - pts[0, 0]
    v2[1] = pts[length, 1] - pts[0, 1]
    v2[2] = pts[length, 2] - pts[0, 2]
    cdef double norm_v2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2])

    for i in range(n_values):
        k = (i + 1) * length

        # Current vector
        dot_prod = 0.0
        norm_v = 0.0
        for j in range(3):
            comp = pts[k, j] - pts[k - length, j]
            dot_prod += comp * v2[j]
            norm_v += comp * comp

        norm_v = sqrt(norm_v)

        # Cosine with clipping
        cos_val = dot_prod / (norm_v * norm_v2)
        if cos_val > 1.0:
            cos_val = 1.0
        elif cos_val < -1.0:
            cos_val = -1.0

        cos_view[i] = cos_val

    return cos_vals

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_cosVals_cython(const double[:, :] ch, const double[:, :] all_angles, 
                        int length):
    """Process a batch of angle sets and compute cosine values."""
    cdef int n_samples = all_angles.shape[0]
    cdef int n_pts = ch.shape[0]
    cdef int n_cos = (n_pts - 2) // length

    cdef cnp.ndarray[double, ndim=2] results = np.empty((n_samples, n_cos), dtype=np.float64)
    cdef double[:, :] results_view = results
    cdef int i, j
    cdef cnp.ndarray[double, ndim=2] rotated_pts
    cdef cnp.ndarray[double, ndim=1] cos_vals

    for i in range(n_samples):
        rotated_pts = randomRotate_cython(ch, all_angles[i, :])
        cos_vals = cosVals_cython(rotated_pts, length)
        for j in range(n_cos):
            results_view[i, j] = cos_vals[j]

    return results

@cython.boundscheck(False)
@cython.wraparound(False)
def batch_r2_cython(const double[:, :] ch, const double[:, :] all_angles,
                        int length):
    """Process a batch of angle sets and compute cosine values."""
    cdef int n_samples = all_angles.shape[0]
    cdef int n_pts = ch.shape[0]
    cdef int n_r = n_pts // length + 1

    cdef cnp.ndarray[double, ndim=2] results = np.empty((n_samples, n_r), dtype=np.float64)
    cdef double[:, :] results_view = results
    cdef int i, j, idx
    cdef cnp.ndarray[double, ndim=2] rotated_pts
    cdef double dx, dy, dz

    for i in range(n_samples):
        rotated_pts = randomRotate_cython(ch, all_angles[i, :])
        for j in range(n_r):
            idx = j * length
            dx = rotated_pts[idx, 0]
            dy = rotated_pts[idx, 1]
            dz = rotated_pts[idx, 2]
            results_view[i, j] = dx * dx + dy * dy + dz * dz

    return results
