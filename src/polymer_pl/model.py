from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.constants as sc
from joblib import Parallel, delayed
from numpy.linalg import eigvals
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from . import tool

try:
    from . import chain_rotation
except ImportError:
    print(
        "Warning: chain_rotation module not found. Cython functions will not be available."
    )
    chain_rotation = None


class PolymerPersistence:
    """
    Calculates the persistence length of a polymer chain based on its
    molecular structure and dihedral angle potentials.

    This class encapsulates the calculations for determining the persistence
    length from bond lengths, bond angles, and rotational potentials
    using the matrix transformation method or the Monte Carlo method.
    """

    def __init__(self,
                 bond_lengths,
                 bond_angles_deg,
                 temperature=300.0,
                 rotation_types=None,
                 rotation_labels=None,
                 ris_types=None,
                 ris_labels=None,
                 fitting_method='interpolation',
                 param_n=5):
        """
        Initializes the PolymerPersistence model.

        Args:
            bond_lengths (list or np.ndarray): The lengths of the bonds in the repeat unit.
            bond_angles_deg (list or np.ndarray): The deflection angles between bonds in degrees.
            temperature (int, optional): The temperature in Kelvin. Defaults to 300.
            rotaion_types (list or np.ndarray, optional): An array of integers mapping each bond to a
                                                 specific rotational potential profile. A value of 0
                                                 indicates a fixed bond with no rotation.
            rotation_labels (dict, optional): A dictionary mapping rotation_types to data files.
            ris_types (list or np.ndarray, optional): An array of integers mapping each bond to ris model.
            ris_labels (dict, optional): A dictionary mapping ris_types to data files.
            fitting_method (str, optional): The method used for fitting the data. 'interpolation', 'cosine' or 'fourier'.
            param_n (int, optional): The order for fitting the data.
        """
        self.bond_lengths = np.asarray(
            bond_lengths) if bond_lengths is not None else None
        self.bond_angles_rad = np.deg2rad(np.array(bond_angles_deg))
        self.rotation_types = np.array(rotation_types)
        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

        # Default labels mapping rotation_types to data files
        self.rotation_labels = rotation_labels
        for rot_id in self.rotation_labels:
            rot = self.rotation_labels[rot_id]
            if 'data' in rot or 'fitf' in rot and 'label' not in rot:
                self.rotation_labels[rot_id]['label'] = f"dihedral {rot_id}"
            if 'loc' in rot and 'label' not in rot:
                file_path = self.rotation_labels[rot_id]['loc']
                self.rotation_labels[rot_id]['label'] = Path(file_path).stem
        self.ris_labels = ris_labels
        self.ris_types = ris_types
        # Initialize RIS data if needed
        if self.ris_types is not None:
            self.ris_types = np.array(self.ris_types)
            if not hasattr(self, 'ris_data') or self.ris_data is None:
                self.ris_data = {}
                for ris_id, info in self.ris_labels.items():
                    try:
                        if 'data' in info:
                            risdata = np.asarray(info['data'])
                            angles, energies = risdata[:, 0], risdata[:, 1]
                        elif 'loc' in info:
                            angles, energies = tool.read_ris_data(
                                Path(info['loc']))
                        self.ris_data[ris_id] = (angles, energies)
                    except FileNotFoundError:
                        print(
                            f"Warning: RIS data file not found. Skipping RIS type {ris_id}."
                        )
                        continue
        # --- Internal cache for lazy evaluation ---
        self._Mmat = None
        self._A_list = None
        self._G_unit = None
        self._G_free_unit = None
        self._lambda_max = None
        self._correlation_length = None
        self._correlation_length_wlc = None
        self._unit_length_wlc = None
        self._computational_data = {}
        self._full_data = {}
        self.fitting_method = fitting_method
        self.param_n = param_n

    @staticmethod
    def _update_dihedral(data):
        data = np.asarray(data)
        data[:, 1] -= data[:, 1].min()
        if data[:, 0].max() == 360 and data[:, 0].min() == 0:
            return data
        else:
            all_angles = np.concatenate(
                ((data[:, 0] + 360) % 360, (-data[:, 0] + 360) % 360))
            all_energies = np.concatenate((data[:, 1], data[:, 1]))
            combined = np.column_stack((all_angles, all_energies))
            _, index = np.unique(combined[:, 0], axis=0, return_index=True)
            combined = combined[index]
            zero_mask = np.isclose(combined[:, 0], 0.0, atol=1e-8)
            if np.any(zero_mask):
                energy0 = combined[zero_mask, 1][0]
                combined = np.vstack((combined, [360.0, energy0]))
                combined = combined[np.argsort(combined[:, 0])]
            return combined

    def _read_data(self, file_name: Path):
        """Reads and processes dihedral angle data from a file."""
        delimiter = ',' if file_name.suffix == '.csv' else None
        data = np.loadtxt(file_name, delimiter=delimiter)
        data = np.reshape(data, (-1, 2))
        return self._update_dihedral(data)

    def _prepare_computational_data(self):
        """Sets up interpolation functions from data files."""
        if self._computational_data:
            return

        for rot_id, info in self.rotation_labels.items():
            try:
                if 'fitf' in info:
                    self._computational_data[rot_id] = {
                        'fitf': info['fitf'],
                        **info
                    }
                    continue
                elif 'data' in info:
                    data = self._update_dihedral(info['data'])
                elif 'loc' in info:
                    data = self._read_data(Path(info['loc']))
                else:
                    raise ValueError(
                        f"Either 'data' or 'loc' must be provided for rotation type {rot_id}."
                    )
                x, y = data[:, 0], data[:, 1]
                if self.fitting_method == 'interpolation':
                    fitf = interp1d(x,
                                    y,
                                    kind='cubic',
                                    fill_value="extrapolate")
                elif self.fitting_method == 'cosine':
                    p = np.polynomial.polynomial.polyfit(
                        np.cos(np.deg2rad(x)), y, self.param_n)
                    fitf = (lambda p_val: lambda z: np.polynomial.polynomial.
                            polyval(np.cos(np.deg2rad(z)), p_val))(p)
                elif self.fitting_method == 'fourier':
                    rad = np.deg2rad(x)
                    a = np.column_stack(
                        [np.cos(n * rad) for n in range(self.param_n + 1)])
                    coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
                    fitf = (
                        lambda c, ord_val: lambda z: np.sum([
                            c[n] * np.cos(n * np.deg2rad(z))
                            for n in range(ord_val + 1)
                        ],
                                                            axis=0))(
                                                                coeffs,
                                                                self.param_n)
                self._computational_data[rot_id] = {'fitf': fitf, **info}
            except FileNotFoundError:
                print(
                    f"Warning: Data file not found. Skipping rotation type {rot_id}."
                )
                continue

    def _prepare_full_data(self):
        """Sets up interpolation functions from data files."""
        if self._full_data:
            return
        for rot_id, info in self.rotation_labels.items():
            try:
                if 'fitf' in info:
                    fitf = info['fitf']
                    data = np.empty((0, 2))
                else:
                    if 'data' in info:
                        data = self._update_dihedral(info['data'])
                    elif 'loc' in info:
                        data = self._read_data(Path(info['loc']))
                    else:
                        raise ValueError(
                            f"Either 'data' or 'loc' must be provided for rotation type {rot_id}."
                        )
                    x, y = data[:, 0], data[:, 1]
                    if self.fitting_method == 'interpolation':
                        fitf = interp1d(x,
                                        y,
                                        kind='cubic',
                                        fill_value="extrapolate")
                    elif self.fitting_method == 'cosine':
                        p = np.polynomial.polynomial.polyfit(
                            np.cos(np.deg2rad(x)), y, self.param_n)
                        fitf = (
                            lambda p_val: lambda z: np.polynomial.polynomial.
                            polyval(np.cos(np.deg2rad(z)), p_val))(p)
                    elif self.fitting_method == 'fourier':
                        rad = np.deg2rad(x)
                        a = np.column_stack(
                            [np.cos(n * rad) for n in range(self.param_n + 1)])
                        coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
                        fitf = (lambda c, ord_val: lambda z: np.sum([
                            c[n] * np.cos(n * np.deg2rad(z))
                            for n in range(ord_val + 1)
                        ],
                                                                    axis=0)
                                )(coeffs, self.param_n)
                norm_val, _ = quad(lambda x: np.exp(-fitf(x) / self.kTval),
                                   0,
                                   360,
                                   limit=1000)
                x_values = np.linspace(0, 360, 1000)
                prob_vals = np.exp(-fitf(x_values) / self.kTval) / norm_val
                cum_dist = cumulative_trapezoid(prob_vals, x_values, initial=0)
                norm_cdf = cum_dist / cum_dist[-1]
                unique_cdf_vals, unique_indices = np.unique(norm_cdf,
                                                            return_index=True)
                corresponding_x_vals = x_values[unique_indices]
                inv_cdf = interp1d(unique_cdf_vals,
                                   corresponding_x_vals,
                                   kind='cubic',
                                   fill_value="extrapolate")
                self._full_data[rot_id] = {
                    'fitf': fitf,
                    'original': data,
                    'prob_vals': prob_vals,
                    "inv_cdf": inv_cdf,
                    "x_values": x_values,
                    "cum_dist": cum_dist,
                    **info
                }
            except FileNotFoundError:
                print(
                    f"Warning: Data file not found. Skipping rotation type {rot_id}."
                )
                continue

    def _compute_rotation_integrals(self, fitf, limit=1000):
        """Calculates the Boltzmann-averaged <cos(phi)> and <sin(phi)>."""

        def exp_energy(phi_deg):
            return np.exp(-fitf(phi_deg) / self.kTval)

        Z, _ = quad(exp_energy, 0, 360, limit=limit)
        if Z == 0:
            return 0.0, 0.0

        cos_avg, _ = quad(
            lambda phi: np.cos(np.deg2rad(phi)) * exp_energy(phi),
            0,
            360,
            limit=limit)
        sin_avg, _ = quad(
            lambda phi: np.sin(np.deg2rad(phi)) * exp_energy(phi),
            0,
            360,
            limit=limit)
        return cos_avg / Z, sin_avg / Z

    def _calculate_Mmat(self):
        """Constructs the overall transformation matrix M for the repeat unit."""
        self._prepare_computational_data()

        M = len(self.rotation_types)
        A_list = []
        integral_cache = {}
        ris_cache = {}

        for i in range(M):
            rot_id = int(self.rotation_types[i])
            ris_id = int(
                self.ris_types[i]) if self.ris_types is not None else 0
            theta = float(self.bond_angles_rad[i])

            if rot_id == 0 and ris_id == 0:
                m_i, s_i = 1.0, 0.0  # Fixed bond
            elif rot_id != 0:
                # Continuous rotation model
                if rot_id not in self._computational_data:
                    print(
                        f"Warning: No data for rotation ID {rot_id}. Assuming a rigid bond (m=1, s=0)."
                    )
                    m_i, s_i = 1.0, 0.0
                else:
                    if rot_id not in integral_cache:
                        fitf = self._computational_data[rot_id]['fitf']
                        integral_cache[
                            rot_id] = self._compute_rotation_integrals(fitf)
                    m_i, s_i = integral_cache[rot_id]
            elif ris_id != 0:
                if ris_id not in self.ris_data:
                    print(
                        f"Warning: No data for RIS ID {ris_id}. Assuming a rigid bond (m=1, s=0)."
                    )
                    m_i, s_i = 1.0, 0.0
                else:
                    if ris_id not in ris_cache:
                        angles_deg, energies = self.ris_data[ris_id]
                        m_i, s_i = tool.compute_ris_rotation_integrals(
                            angles_deg, energies, self.kTval)
                        ris_cache[ris_id] = (m_i, s_i)
                    else:
                        m_i, s_i = ris_cache[ris_id]
            else:
                # Default case (should not reach here)
                m_i, s_i = 1.0, 0.0

            # Dihedral rotation matrix (around x-axis)
            R_x = np.array([[1, 0.0, 0.0], [0.0, m_i, -s_i], [0.0, s_i, m_i]])

            # Bond angle deflection matrix (around z-axis)
            c, s = np.cos(theta), np.sin(theta)
            R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])

            A_list.append(R_z @ R_x)

        # Multiply all transformation matrices for the repeat unit
        Mmat = np.eye(3)
        if len(A_list) > 1:
            for A in A_list[1:]:
                Mmat = Mmat @ A
        Mmat = Mmat @ A_list[0]
        self._Mmat = Mmat
        self._A_list = A_list

    def build_G_unit(self):
        """
        Builds the unit transfer matrix G_unit for the polymer chain.
        Returns:
            np.ndarray: The unit transfer matrix G_unit.
        """
        # Flory Generator Matrix
        # in our bond_angle and rotation definition,
        # G_i include l_i and T_{i+1}
        # v_i . v_{i+1} correlation is defined by T_{i+1}
        num_bonds = len(self.bond_lengths)
        if self._A_list is None:
            self._calculate_Mmat()
        avg_matrices = self._A_list
        G_unit = np.eye(5)
        for i in range(num_bonds):
            # current bond length
            l_vec = np.array([self.bond_lengths[i], 0.0, 0.0])
            l_sq = self.bond_lengths[i]**2
            # matrix for the next bond (periodic boundary)
            next_idx = (i + 1) % num_bonds
            T_next = avg_matrices[next_idx]
            # build G_i
            # [ 1  2l^T T_next  l^2 ]
            # [ 0    T_next      l  ]
            # [ 0      0         1  ]
            G_i = np.zeros((5, 5))
            G_i[0, 0] = 1.0
            G_i[0, 1:4] = 2 * l_vec.T @ T_next
            G_i[0, 4] = l_sq

            G_i[1:4, 1:4] = T_next
            G_i[1:4, 4] = l_vec
            G_i[4, 4] = 1.0

            G_unit = G_unit @ G_i
        self._G_unit = G_unit

    def calculate_characteristic_ratio(self):
        """
        Calculates the Characteristic Ratio C_infinity using the algebraic 
        steady-state solution of the Flory generator matrix.
        Formula:
            Delta_R2 = G[0,4] + G[0, 1:4] @ (I - G[1:4, 1:4])^-1 @ G[1:4, 4]
            C_inf = Delta_R2 / l_unit^2
        """
        # Build the generator matrix for the repeating unit
        if self._G_unit is None:
            self.build_G_unit()
        G_unit = self._G_unit
        # Extract submatrices
        # M: rotation matrix (3x3)
        M = G_unit[1:4, 1:4]
        # p: end-to-end vector of unit (3x1)
        p = G_unit[1:4, 4]
        # n: correlation row vector (1x3)
        n_vec = G_unit[0, 1:4]
        # s: scalar within unit
        s = G_unit[0, 4]
        # Check for rigid rod case (if eigenvalue is close to 1, matrix is not invertible)
        # Considering floating-point errors, treat as infinite if close to 1
        if np.abs(self._lambda_max - 1.0) < 1e-10:
            return np.inf
        # Algebraically solve for the increment Delta <R^2> in the long-chain limit
        # (I - M)^-1
        I = np.eye(3)
        try:
            inv_I_minus_M = np.linalg.inv(I - M)
        except np.linalg.LinAlgError:
            return np.inf
        # Delta R^2 = s + n * (I-M)^-1 * p
        delta_R2 = s + n_vec @ inv_I_minus_M @ p
        C_inf = delta_R2 / s
        return C_inf

    def calculate_persistence_length(self):
        """
        Calculates the geometric persistence length l_p.
        Definition: The projection of the average end-to-end vector of an 
        infinite chain onto the direction of the first bond.
        This is more accurate for discrete chains than the eigenvalue method 
        (-1/ln(lambda)), which describes correlation decay rate rather than physical length
        Important: use geometric average of all possible translations of the structure
        """
        if self._A_list is None:
            self._calculate_Mmat()
        num_bonds = len(self.bond_lengths)
        precomputed_Gi = []
        for idx in range(num_bonds):
            next_idx = (idx + 1) % num_bonds
            l_vec = np.array([self.bond_lengths[idx], 0.0, 0.0])
            T_next = self._A_list[next_idx]
            G_i = np.zeros((4, 4))
            G_i[0:3, 0:3] = T_next
            G_i[0:3, 3] = l_vec
            G_i[3, 3] = 1.0
            precomputed_Gi.append(G_i)
        lp_list = []
        for start_idx in range(num_bonds):
            G = np.eye(4)
            for i in range(num_bonds):
                idx = (start_idx + i) % num_bonds
                G = G @ precomputed_Gi[idx]
            M = G[0:3, 0:3]
            p = G[0:3, 3]
            try:
                # p + M @ p + M^2 @ p + ...  = (I - M)^-1 @ p
                R_avg = np.linalg.solve(np.eye(3) - M, p)
                lp = np.dot(R_avg, p) / np.linalg.norm(p)
            except np.linalg.LinAlgError:
                lp = np.inf
            lp_list.append(lp)
        lp = np.array(lp_list)
        if np.all(np.isinf(lp)):
            return np.inf
        else:
            return np.mean(lp)

    def _wormlike_chain_approximation(self):
        """
        Calculates the effective unit length (alpha) and effect correlation length (Np)
        for a WLC model by matching both the asymptotic slope and intercept 
        of the discrete chain's exact <R^2>.
        """
        if self._G_unit is None:
            self.build_G_unit()
        G = self._G_unit
        # Extract sub-blocks from the Generator Matrix
        s = G[0, 4]  # scalar: bond length squared term
        n_vec = G[0, 1:4]  # row vector: related to 2*l*T correlations
        p_vec = G[1:4, 4]  # col vector: bond vector
        M = G[1:4, 1:4]  # matrix: rotation/transfer matrix
        I = np.eye(3)
        # Calculate (I - M)^-1
        try:
            inv_I_M = np.linalg.inv(I - M)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Matrix (I - M) is singular. Chain might be perfectly rigid or linear."
            )
        # accurate equation for <R2> = A*N + B + n^T @ (I - M)^-2 @ M^N @ p
        # delta_R2 = 2*Np*alpha^2 when n -> infinity
        # A = s + n^T @ (I - M)^-1 @ p (slope)
        A = s + n_vec @ inv_I_M @ p_vec
        # intercept = -2* Np * alpha^2 when n -> infinity
        # B = - n^T @ (I - M)^-2 @ p (intercept)
        # Note: (I-M)^-2 is inv_I_M @ inv_I_M
        B = -n_vec @ (inv_I_M @ inv_I_M) @ p_vec
        if np.abs(A) < 1e-10:
            return 0.0  # Avoid division by zero for zero-length chains
        Np = -B / A
        # decay term n^T @ (I - M)^-2 @ M^N @ p can be accurately described by three eigenvalues
        # M @ v_i = lambda_i * v_i, u_i^T @ M = lambda_i * u_i^T, u_i^T @ v_i = delta_ij
        # M^N = ∑ lambda_i^N * v_i * u_i^T
        # n^T @ (I - M)^-2 @ M^N @ p = ∑ (n^T @ (I - M)^-2 @ v_i)*(u_i^T @ p) * lambda_i^N
        # alpha = sqrt( A / (2 * Np) )
        alpha_sq = A / (2 * Np)
        alpha = np.sqrt(alpha_sq)
        self._unit_length_wlc = alpha
        self._correlation_length_wlc = Np

    def run_calculation(self):
        """
        Runs the full calculation to find the correlation length.
        This method populates the result attributes.
        """
        if self._Mmat is None:
            self._calculate_Mmat()

        eigs = eigvals(self._Mmat)
        self._lambda_max = float(np.max(np.abs(eigs)))

        if self._lambda_max >= 1.0:
            self._correlation_length = np.inf
        else:
            self._correlation_length = -1.0 / np.log(self._lambda_max)

    @property
    def correlation_length(self):
        """The correlation length."""
        if self._correlation_length is None:
            self.run_calculation()
        return self._correlation_length

    @property
    def lambda_max(self):
        """The largest absolute eigenvalue of the transformation matrix."""
        if self._lambda_max is None:
            self.run_calculation()
        return self._lambda_max

    @property
    def matrix(self):
        """The overall transformation matrix for the repeat unit."""
        if self._Mmat is None:
            self._calculate_Mmat()
        return self._Mmat

    @property
    def generator_matrix(self):
        """The generator matrix for the repeat unit."""
        if self._G_unit is None:
            self.build_G_unit()
        return self._G_unit

    @property
    def c_inf(self):
        """The characteristic ratio."""
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
        return self.calculate_characteristic_ratio()

    @property
    def average_unit_vector(self):
        """The average unit vector."""
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
        if self._G_unit is None:
            self.build_G_unit()
        return self._G_unit[1:4, 4]

    @property
    def average_unit_length(self):
        return np.linalg.norm(self.average_unit_vector)

    @property
    def mean_square_unit_length(self):
        """The mean square unit length."""
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
        if self._G_unit is None:
            self.build_G_unit()
        return self._G_unit[0, 4]

    @property
    def kuhn_length(self):
        """The Kuhn length."""
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
        return self.c_inf * np.sqrt(self.mean_square_unit_length)

    @property
    def persistence_length(self):
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
        return self.calculate_persistence_length()

    @property
    def persistence_length_units(self):
        """The persistence length in repeat units."""
        return self.persistence_length / np.sqrt(self.mean_square_unit_length)

    @property
    def effective_unit_length_wlc(self):
        """The effective unit length for a worm-like chain (WLC) model."""
        if self._unit_length_wlc is None:
            self._wormlike_chain_approximation()
        return self._unit_length_wlc

    @property
    def correlation_length_wlc(self):
        """The correlation length for a worm-like chain (WLC) model."""
        if self._correlation_length_wlc is None:
            self._wormlike_chain_approximation()
        return self._correlation_length_wlc

    @property
    def persistence_length_wlc(self):
        """The persistence length for a worm-like chain (WLC) model."""
        try:
            wlc = self.effective_unit_length_wlc * self.correlation_length_wlc
        except:
            wlc = np.inf
        return wlc

    def plot_dihedral_potentials(self):
        """Plot dihedral potentials and their probability distributions."""
        if not self._full_data:
            self._prepare_full_data()
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        for key, data in self._full_data.items():
            plt.plot(data['original'][:, 0],
                     data['original'][:, 1],
                     f"{data['color']}",
                     marker="o",
                     linestyle="None",
                     label=data['label'])
            plt.plot(data['x_values'],
                     data['fitf'](data['x_values']),
                     color=f"{data['color']}",
                     linestyle="--")
        tool.format_subplot("Dihedral Angle [Deg.]",
                            "Dihedral Potential (kJ/mol)",
                            "Dihedral Potentials")
        plt.subplot(1, 3, 2)
        for key, data in self._full_data.items():
            plt.plot(data['x_values'],
                     data['prob_vals'],
                     color=f"{data['color']}",
                     linestyle="-",
                     label=data['label'])
        tool.format_subplot("Angle [deg.]", "Probability",
                            "Probability Distributions")
        plt.subplot(1, 3, 3)
        for key, data in self._full_data.items():
            plt.plot(data['cum_dist'] / data['cum_dist'][-1],
                     data['x_values'],
                     color=f"{data['color']}",
                     linestyle="-",
                     label=data['label'])
        tool.format_subplot("Probability", "Dihedral Angle [deg.]",
                            "Cumulative Probability Distributions")

        plt.tight_layout()
        plt.show()

    def report(self):
        """Prints a summary of the calculation results."""
        # Ensure calculation has been run
        corr = self.correlation_length
        lam = self.lambda_max
        print("-------------- Calculation Report -------------")
        print(f"Temperature: {self.temperature} K")
        print(f"Max Eigenvalue (lambda_max): {lam:.12f}")
        print(f"Correlation Length: {corr:.6f}")
        if self.bond_lengths is not None:
            print(
                f"Persistence Length Geometric (Angstroms): {self.persistence_length:.6f}"
            )
            print(
                f"Persistence Length WLC (Angstroms): {self.persistence_length_wlc:.6f}"
            )
            print(f"Conformational Parameter: {self.conformational_param:.6f}")
        print("-----------------------------------------------")

    def generate_chain(self, n_repeat_units):
        """Generate a polymer chain with n_repeat_units."""
        bonds = self.bond_lengths if self.bond_lengths is not None else np.array(
            [1.0] * len(self.bond_angles_rad))
        l_array = np.tile(bonds, n_repeat_units)
        vectors = np.hstack((l_array[:, None], np.zeros(
            (l_array.shape[0], 2))))
        all_angle = np.tile(self.bond_angles_rad, n_repeat_units)
        angles = np.cumsum(all_angle)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        rotated_x = vectors[:, 0] * cos_angles - vectors[:, 1] * sin_angles
        rotated_y = vectors[:, 0] * sin_angles + vectors[:, 1] * cos_angles
        rotated_z = vectors[:, 2]
        segments = np.column_stack((rotated_x, rotated_y, rotated_z))
        return np.cumsum(np.vstack((np.array([[0, 0, 0]]), segments)), axis=0)

    def pre_generate_angles(self, n_samples, flat_rotation, flat_ris):
        """Original independent sampling method."""
        self._prepare_full_data()
        num_positions = len(flat_rotation)
        rng = np.random.default_rng()
        rand_vals = rng.random((n_samples, num_positions))
        angles_per_position = np.zeros((n_samples, num_positions))

        for rot_type, data_type in self._full_data.items():
            mask = flat_rotation == rot_type
            if np.any(mask):
                inv_cdf = data_type['inv_cdf']
                angles_per_position[:, mask] = inv_cdf(rand_vals[:, mask])

        if flat_ris is not None:
            for ris_id, (ang_deg, energies) in self.ris_data.items():
                mask = (flat_ris == ris_id)
                if not np.any(mask):
                    continue

                boltz = np.exp(-energies / self.kTval)
                prob = boltz / boltz.sum()

                sampled = rng.choice(ang_deg,
                                     size=(n_samples, np.sum(mask)),
                                     p=prob)
                angles_per_position[:, mask] = sampled
        return angles_per_position

    def calc_mean_square_end_to_end_distance(self,
                                             n_repeat_units=20,
                                             n_samples=150000,
                                             return_data=False):
        """Plots the mean square end-to-end distance as a function of repeat units from 1 to N.
        Args:
            N (int): Maximum number of repeat units to plot
            return_data (bool): If True, returns the mean square end-to-end distance values as a list
        """
        if self.bond_lengths is None or chain_rotation is None:
            raise ValueError("Bond lengths and chain_rotation must be set.")
        n_repeats = np.arange(0, n_repeat_units + 1)
        length = len(self.bond_angles_rad)
        ch = self.generate_chain(n_repeat_units)
        batch_size = 1000
        n_batches = n_samples // batch_size
        n_jobs = psutil.cpu_count(logical=False)
        flat_rotation = np.concatenate([
            [0], self.rotation_types[np.arange(len(ch) - 1) % length]
        ])[:-1].astype(np.int64)
        flat_ris = None
        if self.ris_types is not None:
            flat_ris = np.concatenate([
                [0], self.ris_types[np.arange(len(ch) - 1) % length]
            ])[:-1].astype(np.int64)
        all_angles = self.pre_generate_angles(n_samples, flat_rotation,
                                              flat_ris)
        r2List = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(chain_rotation.batch_r2_cython)(
                np.ascontiguousarray(ch, dtype=np.float64),
                np.ascontiguousarray(all_angles[i * batch_size:(i + 1) *
                                                batch_size],
                                     dtype=np.float64), length)
            for i in range(n_batches))
        r2List = np.vstack(r2List)
        msd_values = np.mean(r2List, axis=0)

        plt.figure(figsize=(6, 5))
        plt.plot(n_repeats, msd_values, linewidth=2, color='blue', marker='o')
        tool.format_subplot("Number of Repeat Units (N)",
                            "Mean Square End-to-End Distance (Å²)",
                            "Monte Carlo Simulation of <R²>")
        plt.tight_layout()
        plt.show()

        if return_data:
            return msd_values

    def calculate_correlation_length_mc(self,
                                        n_repeat_units=20,
                                        n_samples=150000):
        """
        Calculate correlation length using Monte Carlo sampling.
        
        Parameters:
        -----------
        n_repeat_units : int, optional
            Number of repeat units in the polymer chain. Default is 20.
        n_samples : int, optional
            Number of Monte Carlo samples to generate. Default is 150,000.
        method: str, Sampling method to use. Options:
            - 'independent': Original independent sampling (fast, noisy)
            - 'stratified': Stratified sampling (reduces noise in flexible systems)
        **sampling_kwargs : dict
            Additional parameters for sampling method (e.g., burnin, thin for MCMC)
        Returns:
        --------
        float
            The calculated correlation length in units of repeat units.
        """
        if chain_rotation is None:
            print(
                "Error: chain_rotation module not available. Monte Carlo simulation cannot be performed."
            )
            return None

        # Generate the base chain
        ch = self.generate_chain(n_repeat_units)
        length = len(self.bond_angles_rad)

        # Prepare rotation mapping for the chain
        flat_rotation = np.concatenate([
            [0], self.rotation_types[np.arange(len(ch) - 1) % length]
        ])[:-1].astype(np.int64)
        flat_ris = None
        if self.ris_types is not None:
            flat_ris = np.concatenate([
                [0], self.ris_types[np.arange(len(ch) - 1) % length]
            ])[:-1].astype(np.int64)

        # Pre-generate all angles
        all_angles = self.pre_generate_angles(n_samples, flat_rotation,
                                              flat_ris)

        print(f"Calculating {n_samples} samples...")
        print(f"Using {psutil.cpu_count(logical=False)} CPU cores")

        # Parallel computation using Cython optimized versionable_memory * 0.5 / sampl
        batch_size = 1000
        n_batches = n_samples // batch_size
        n_jobs = psutil.cpu_count(logical=False)

        cosList2 = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(self._batch_cosVals_optimized)(
                ch, all_angles[i * batch_size:(i + 1) *
                               batch_size], flat_rotation, length)
            for i in range(n_batches))
        cosList2 = np.vstack(cosList2)

        # Calculate correlation length
        corr2 = np.mean(cosList2, axis=0)
        repeat_units = np.arange(len(corr2))
        start_idx = 1
        end_idx = 10

        # Fit exponential decay to correlation function
        p = np.polynomial.polynomial.polyfit(repeat_units[start_idx:end_idx],
                                             np.log(corr2[start_idx:end_idx]),
                                             1)

        corr_length = -1 / p[1]

        print(f"\nMonte Carlo Result:")
        print(f"slope: {p[1]:.6f}")
        print(f"Correlation Length: {corr_length:.2f}")

        return corr_length

    def _batch_cosVals_optimized(self, ch, all_angles, flat_rotation, length):
        """Batch processing function optimized with Cython."""
        if chain_rotation is None:
            raise ImportError("chain_rotation module not available")
        return chain_rotation.batch_cosVals_cython(
            np.ascontiguousarray(ch, dtype=np.float64),
            np.ascontiguousarray(all_angles, dtype=np.float64), length)

    def calculate_contact_map_mc(
        self,
        n_repeat_units=20,
        n_samples=20000,
    ):
        ch = self.generate_chain(n_repeat_units)
        length = len(self.bond_angles_rad)

        flat_rotation = np.concatenate([
            [0], self.rotation_types[np.arange(len(ch) - 1) % length]
        ])[:-1].astype(np.int64)
        flat_ris = None
        if self.ris_types is not None:
            flat_ris = np.concatenate([
                [0], self.ris_types[np.arange(len(ch) - 1) % length]
            ])[:-1].astype(np.int64)

        all_angles = self.pre_generate_angles(n_samples, flat_rotation,
                                              flat_ris)
        c = np.zeros((n_repeat_units, n_repeat_units))
        unit_idx = np.arange(0, n_repeat_units * length + 1, length)
        for i in range(n_samples):
            pos = chain_rotation.randomRotate_cython(ch, all_angles[i])
            r = pos[unit_idx]
            u = r[1:] - r[:-1]
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            c[:len(u), :len(u)] += u @ u.T
        return c / n_samples

    def plot_contact_map_mc(self, n_units=20, n_samples=100):
        plt.figure(figsize=(6, 5))
        im = plt.imshow(self.calculate_contact_map_mc(n_units, n_samples),
                        vmin=-1,
                        vmax=1,
                        cmap="coolwarm",
                        interpolation="bicubic")
        plt.xlabel("i", fontsize=14, fontfamily="Helvetica")
        plt.ylabel("j", fontsize=14, fontfamily="Helvetica")
        plt.xticks(fontsize=14, fontfamily="Helvetica")
        plt.yticks(fontsize=14, fontfamily="Helvetica")
        plt.title("Contact Map", fontsize=16, fontfamily="Helvetica")
        cbar = plt.colorbar(im)
        cbar.set_label(r"$\hat v_i \cdot \hat v_j$",
                       fontsize=14,
                       fontfamily="Helvetica")
        cbar.ax.tick_params(labelsize=14)
        plt.setp(cbar.ax.get_yticklabels(), fontfamily="Helvetica")
        plt.minorticks_on()
        plt.tight_layout()
        plt.show()

    def plot_correlation_function(self,
                                  n_repeat_units=20,
                                  n_samples=150000,
                                  start_idx=1,
                                  end_idx=10,
                                  return_data=False):
        """
        Plot the correlation function and its exponential fit.
        
        Parameters:
        -----------
        n_repeat_units : int, optional
            Number of repeat units in the polymer chain. Default is 20.
        n_samples : int, optional
            Number of Monte Carlo samples to generate. Default is 150,000.
        start_idx : int, optional
            Starting index for fitting. Default is 1.
        end_idx : int, optional
            Ending index for fitting. Default is 10.
        method: str, Sampling method to use. Options:
            - 'independent': Original independent sampling (fast, noisy)
            - 'stratified': Stratified sampling (reduces noise in flexible systems)
        **sampling_kwargs : dict
            Additional parameters for sampling method (e.g., burnin, thin for MCMC)
        """
        try:
            if chain_rotation is None:
                print(
                    "Error: chain_rotation module not available. Plot cannot be generated."
                )
                return

            # Generate the base chain
            ch = self.generate_chain(n_repeat_units)
            length = len(self.bond_angles_rad)

            # Prepare rotation mapping for the chain
            flat_rotation = np.concatenate([
                [0], self.rotation_types[np.arange(len(ch) - 1) % length]
            ])[:-1].astype(np.int64)
            flat_ris = None
            if self.ris_types is not None:
                flat_ris = np.concatenate([
                    [0], self.ris_types[np.arange(len(ch) - 1) % length]
                ])[:-1].astype(np.int64)

            # Pre-generate all angles
            all_angles = self.pre_generate_angles(n_samples, flat_rotation,
                                                  flat_ris)

            # Parallel computation using Cython optimized version
            batch_size = 1000
            n_batches = n_samples // batch_size
            n_jobs = psutil.cpu_count(logical=False)

            cosList2 = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(self._batch_cosVals_optimized)(
                    ch, all_angles[i * batch_size:(i + 1) *
                                   batch_size], flat_rotation, length)
                for i in range(n_batches))
            cosList2 = np.vstack(cosList2)

            # Calculate correlation function
            corr2 = np.mean(cosList2, axis=0)
            repeat_units = np.arange(len(corr2))

            # Validate indices
            end_idx = min(end_idx, len(corr2) - 1)
            if start_idx >= end_idx:
                print("Error: Invalid index range for fitting.")
                return

            # Check for valid correlation values
            if np.any(corr2[start_idx:end_idx] <= 0):
                print(
                    "Warning: Some correlation values are non-positive, log transformation may fail."
                )

            # Fit exponential decay to correlation function
            p = np.polynomial.polynomial.polyfit(
                repeat_units[start_idx:end_idx],
                np.log(corr2[start_idx:end_idx]), 1)

            if p[1] == 0:
                print(
                    "Warning: Zero slope in fit, correlation length undefined."
                )
                corr_length = np.inf
            else:
                corr_length = -1 / p[1]

            # Plotting
            plt.figure(figsize=(6, 5))
            plt.plot(repeat_units[start_idx:end_idx],
                     np.log(corr2[start_idx:end_idx]),
                     'bo',
                     label='Log Correlation')
            plt.plot(repeat_units[start_idx:end_idx],
                     np.polynomial.polynomial.polyval(
                         repeat_units[start_idx:end_idx], p),
                     'b--',
                     linewidth=2,
                     alpha=0.7,
                     label=f'Np = {corr_length:.5f}')
            tool.format_subplot("Repeat Units", r'Ln[$<V_0 \cdot V_n>$]',
                                "Log of Correlation Function")
            plt.show()
            if return_data:
                return corr2

        except Exception as e:
            print(f"Error in plot_correlation_function: {str(e)}")
            return

    def calculate_exact_r2(self, n_repeats):
        """
        Calculates the exact mean square end-to-end distance <R^2> 
        matching the specific forward kinematics of the Monte Carlo simulation.
        
        Args:
            n_repeats (int): Number of repeat units.
            
        Returns:
            float: The exact <R^2>.
        """
        if self._G_unit is None:
            self.build_G_unit()
        G_unit = self._G_unit
        # G_chain = (G_unit)^n
        G_chain = np.linalg.matrix_power(G_unit, n_repeats)

        # result at [0, 4]
        return G_chain[0, 4]

    def calc_mean_square_end_to_end_transfer_matrix(self,
                                                    n_repeat_unit=20,
                                                    return_data=False,
                                                    plot=True):
        """
        Calculates and plots the exact mean square end-to-end distance <R^2>
        as a function of the number of repeat units n from 0 to n_repeat_unit.
        
        Args:
            n_repeat_unit (int): Maximum number of repeat units to calculate.
            return_data (bool, optional): Whether to return the R2 data. Defaults to False.
            plot (optional): Whether to plot the results. Defaults to True.
            
        Returns:
            r2_array.
        """
        n_array = np.arange(n_repeat_unit + 1)
        r2_array = np.zeros(len(n_array))
        if self._G_unit is None:
            self.build_G_unit()
        G_unit = self._G_unit
        # Calculate R^2 for each n
        for i, n in enumerate(n_array):
            if n == 0:
                r2_array[i] = 0.0
            else:
                G_chain = np.linalg.matrix_power(G_unit, n)
                r2_array[i] = G_chain[0, 4]
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(n_array, r2_array, 'bo-', linewidth=2)
            tool.format_subplot("Number of Repeat Units (N)",
                                "Mean Square End-to-End Distance (Å²)",
                                "Transfer Matrix Simulation of <R²>")
            plt.tight_layout()
            plt.show()
        if return_data:
            return r2_array

    def temperature_scan(self, T_list, plot=False):
        """
        T_list: iterable of temperatures (K)
        Returns: dict with keys 'T', 'corr', 'Mmat'
        """
        Ts = np.asarray(T_list, dtype=np.float64)
        results = {'T': Ts, 'corr': [], 'Mmat': []}

        kT_orig = self.kTval
        for T in Ts:
            self.kTval = sc.R * T / 1000.0  # kJ/mol
            # self._computational_data only restore fitf
            # fitf is independent of kT
            try:
                self._calculate_Mmat()
                results['Mmat'].append(self._Mmat)
            except Exception:
                raise ValueError("Failed to compute M matrix")
            # find lambda_max and lp
            eigs = eigvals(self._Mmat)
            lambda_max = float(np.max(np.abs(eigs)))
            if lambda_max >= 1.0:
                corr = np.inf
            else:
                corr = -1.0 / np.log(lambda_max)
            results['corr'].append(corr)
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(Ts, results['corr'], 'o-')
            tool.format_subplot("Temperature (K)", "Correlation Length",
                                "Temperature Scan")
            plt.show()
        # restore original
        self.kTval = kT_orig
        self._calculate_Mmat()
        return results

    def persistence_length_Tscan(self, T_list, plot=False):
        """
        T_list: iterable of temperatures (K)
        Returns: dict with keys 'T', 'corr', 'Mmat'
        """
        Ts = np.atleast_1d(T_list).astype(np.float64)
        results = {'T': Ts, 'lp': [], 'lp_wlc': [], 'G_unit': []}

        kT_orig = self.kTval
        for T in Ts:
            self.kTval = sc.R * T / 1000.0  # kJ/mol
            # self._computational_data only restore fitf
            # fitf is independent of kT
            try:
                self._calculate_Mmat()
                self.build_G_unit()
                results['lp'].append(self.persistence_length)
                self._wormlike_chain_approximation()
                results['lp_wlc'].append(self.persistence_length_wlc)
                results['G_unit'].append(self._G_unit)
            except Exception:
                raise ValueError("Failed to compute generator matrix")
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(Ts, results['lp'], 'bo-', label='Lp')
            plt.plot(Ts, results['lp_wlc'], 'rD-', label='Lp_wlc')
            tool.format_subplot("Temperature (K)", "Persistence Length (Å)",
                                "Temperature Scan")
            plt.show()
        # restore original
        self.kTval = kT_orig
        self._calculate_Mmat()
        self.build_G_unit()
        self._wormlike_chain_approximation()
        return results

    def build_G_free_unit(self):
        """
        calculate the generator matrix for all freely rotating bonds, independent of temperature
        """
        A_list = []
        num_bonds = len(self.rotation_types)
        for i in range(num_bonds):
            rot_id = int(self.rotation_types[i])
            theta = float(self.bond_angles_rad[i])
            c, s = np.cos(theta), np.sin(theta)
            if rot_id == 0:
                R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])
                A_list.append(R_z)
            else:
                A_list.append(
                    np.array([[c, 0, 0.0], [s, 0, 0.0], [0.0, 0.0, 0]]))
        G_free = np.eye(5)
        for i in range(num_bonds):
            l_vec = np.array([self.bond_lengths[i], 0.0, 0.0])
            l_sq = self.bond_lengths[i]**2
            next_idx = (i + 1) % num_bonds
            T_next = A_list[next_idx]
            G_i = np.zeros((5, 5))
            G_i[0, 0] = 1.0
            G_i[0, 1:4] = 2 * l_vec.T @ T_next
            G_i[0, 4] = l_sq
            G_i[1:4, 1:4] = T_next
            G_i[1:4, 4] = l_vec
            G_i[4, 4] = 1.0
            G_free = G_free @ G_i
        self._G_free_unit = G_free

    @property
    def conformational_param(self):
        """
        sigma = sqrt(<R^2>/<R^2>_free) with infinite chain length
        """
        if self._G_unit is None:
            self.build_G_unit()
        G = self._G_unit
        s = G[0, 4]
        n = G[0, 1:4]
        p = G[1:4, 4]
        M = G[1:4, 1:4]
        if self._G_free_unit is None:
            self.build_G_free_unit()
        G_free = self._G_free_unit
        s1 = G_free[0, 4]
        n1 = G_free[0, 1:4]
        p1 = G_free[1:4, 4]
        M1 = G_free[1:4, 1:4]
        try:
            inv_I_minus_M = np.linalg.inv(np.eye(3) - M)
            inv_I_minus_M1 = np.linalg.inv(np.eye(3) - M1)
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error inverting (I - M): {e}")
            return 1.0
        delta_r2 = s + n @ inv_I_minus_M @ p
        delta_r2_free = s1 + n1 @ inv_I_minus_M1 @ p1
        return np.sqrt(delta_r2 / delta_r2_free)
