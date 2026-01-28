from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from numpy.linalg import eigvals
from scipy.integrate import quad
from scipy.interpolate import interp1d
from . import tool

try:
    from . import chain_rotation
except ImportError:
    print(
        "Warning: chain_rotation module not found. Cython functions will not be available."
    )
    chain_rotation = None


class PolymerPersistenceDependentDefelection:
    """
    Calculates the persistence length of a polymer chain based on its
    molecular structure and dihedral angle potentials.

    This class encapsulates the calculations for determining the persistence
    length from bond lengths, bond angles, and rotational potentials
    using the matrix transformation method or the Monte Carlo method.
    """

    def __init__(self,
                 bond_lengths,
                 bond_angles_file,
                 temperature=300.0,
                 rotation_types=None,
                 rotation_labels=None,
                 deflection_types=None,
                 fitting_method='interpolation',
                 param_n=5):
        """
        Initializes the PolymerPersistenceDependentDefelection model.

        Args:
            bond_lengths (list or np.ndarray): The lengths of the bonds in the repeat unit.
            bond_angles_file (Path): The path to the file containing the dihedraal-dependent bond angles.
            temperature (int, optional): The temperature in Kelvin. Defaults to 300.
            rotation_types (list or np.ndarray, optional): An array of integers mapping each bond to a
                                                 specific rotational potential profile. A value of 0
                                                 indicates a fixed bond with no rotation.
            rotation_labels (dict, optional): A dictionary mapping rotation_types to data files.
            fitting_method (str, optional): The method used for fitting the data. 'interpolation', 'cosine' or 'fourier'.
            param_n (int, optional): The order for fitting the data.
        """
        self.bond_lengths = np.asarray(
            bond_lengths) if bond_lengths is not None else None
        self.bond_angle_file = bond_angles_file
        self.rotation_types = np.array(rotation_types)
        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

        # Default labels mapping rotation_types to data files
        self.rotation_labels = rotation_labels
        ris_list = []
        for rot_id, info in self.rotation_labels.items():
            if 'type' not in info:
                self.rotation_labels[rot_id]['type'] = 'continuous'
            if info.get('type') == 'ris':
                ris_list.append(rot_id)
            if 'data' in info or 'fitf' in info and 'label' not in info:
                self.rotation_labels[rot_id]['label'] = f"dihedral {rot_id}"
            if 'loc' in info and 'label' not in info:
                file_path = self.rotation_labels[rot_id]['loc']
                self.rotation_labels[rot_id]['label'] = Path(file_path).stem
        self.ris_data = {}
        for rot_id, info in self.rotation_labels.items():
            if info.get('type') == 'ris':
                try:
                    if 'data' in info:
                        risdata = np.asarray(info['data'])
                        angles, energies = risdata[:, 0], risdata[:, 1]
                    elif 'loc' in info:
                        angles, energies = tool.read_ris_data(Path(
                            info['loc']))
                    self.ris_data[rot_id] = (angles, energies)
                except FileNotFoundError:
                    print(
                        f"Warning: RIS data file not found. Skipping RIS type {rot_id}."
                    )
                    continue
        if deflection_types is None:
            self.deflection_types = np.array(rotation_types).copy()
            for ris_id in ris_list:
                mask0 = self.deflection_types == ris_id
                self.deflection_types[mask0] = 0
            mask = self.deflection_types == 0
            self.deflection_types[mask] = np.roll(self.deflection_types, 1)[mask]
        else:
            self.deflection_types = np.array(deflection_types)
        # --- Internal cache for lazy evaluation ---
        self._Mmat = None
        self._A_list = None
        self._G_unit = None
        self._lambda_max = None
        self._correlation_length = None
        self._correlation_length_wlc = None
        self._unit_length_wlc = None
        self._computational_data = {}
        self.fitting_method = fitting_method
        self.param_n = param_n
        self._avg_angles = None

    @staticmethod
    def _update_angle(data):
        data = np.asarray(data)
        if data[:, 0].max() == 360 and data[:, 0].min() == 0:
            return data
        else:
            all_angles = np.concatenate(
                ((data[:, 0] + 360) % 360, (-data[:, 0] + 360) % 360))
            all_energies = np.concatenate((data[:, 1:], data[:, 1:]))
            combined = np.column_stack((all_angles, all_energies))
            _, index = np.unique(combined[:, 0], axis=0, return_index=True)
            combined = combined[index]
            zero_mask = np.isclose(combined[:, 0], 0.0, atol=1e-8)
            if np.any(zero_mask):
                energy0 = combined[zero_mask, 1:][0]
                row_360 = np.concatenate(([360.0], energy0))
                combined = np.vstack((combined, row_360))
                combined = combined[np.argsort(combined[:, 0])]
            return combined

    def _read_data(self, file_name: Path):
        """Reads and processes dihedral angle data from a file."""
        delimiter = ',' if file_name.suffix == '.csv' else None
        data = np.loadtxt(file_name, delimiter=delimiter)
        return self._update_angle(data)

    def _minimize_dihedral(self, data):
        new_energy = data[:, 1] - data[:, 1].min()
        return np.column_stack((data[:, 0], new_energy))

    def _fit_deflection(self, file_name: Path):
        """Fits the deflection angle data."""
        data = self._read_data(file_name)
        deflection_func = []
        for i in range(1, data.shape[1]):
            deflection_func.append(
                interp1d(data[:, 0],
                         data[:, i],
                         kind='cubic',
                         fill_value="extrapolate"))
        return deflection_func

    def _prepare_computational_data(self):
        """Sets up interpolation functions from data files."""
        if self._computational_data:
            return

        for rot_id, info in self.rotation_labels.items():
            if info.get('type') == 'ris':
                continue
            try:
                if 'fitf' in info:
                    self._computational_data[rot_id] = {
                        'fitf': info['fitf'],
                        **info
                    }
                    continue
                elif 'data' in info:
                    data = self._update_angle(info['data'])
                elif 'loc' in info:
                    data = self._read_data(Path(info['loc']))
                else:
                    raise ValueError(
                        f"Either 'data' or 'loc' must be provided for rotation type {rot_id}."
                    )
                data = self._minimize_dihedral(data)
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
        # print("Computational data prepared.")

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

    def _compute_deflection_integrals(self,
                                      fitf_deflection,
                                      fitf_rotation,
                                      limit=1000):
        """Calculates the Boltzmann-averaged <cos(theta)> and <sin(theta)>."""

        def w(phi_deg):
            return np.exp(-fitf_rotation(phi_deg) / self.kTval)

        Z = quad(w, 0, 360, limit=limit)[0]
        if Z == 0:
            return 0.0, 0.0
        cos_avg = quad(lambda phi_deg: np.cos(
            np.deg2rad(fitf_deflection(phi_deg))) * w(phi_deg) / Z,
                       0,
                       360,
                       limit=1000)[0]
        sin_avg = quad(lambda phi_deg: np.sin(
            np.deg2rad(fitf_deflection(phi_deg))) * w(phi_deg) / Z,
                       0,
                       360,
                       limit=1000)[0]
        angle_avg = quad(
            lambda phi_deg: fitf_deflection(phi_deg) * w(phi_deg) / Z,
            0,
            360,
            limit=1000)[0]
        return cos_avg, sin_avg, angle_avg

    def _calculate_Mmat(self):
        """Constructs the overall transformation matrix M for the repeat unit."""
        self._prepare_computational_data()
        # list deflection functions
        fitf_deflection = self._fit_deflection(Path(self.bond_angle_file))

        M = len(self.rotation_types)
        # deflection using rotation types
        A_list = []
        avg_angles = []
        integral_cache = {}
        ris_cache = {}

        for i in range(M):
            rot_id = int(self.rotation_types[i])
            fitf_angle = fitf_deflection[i]
            deflection_id = self.deflection_types[i]
            if deflection_id == 0:
                angle_avg = quad(
                    lambda phi_deg: fitf_angle(phi_deg), 0, 360,
                    limit=1000)[0] / 360
                c, s = np.cos(np.deg2rad(angle_avg)), np.sin(
                    np.deg2rad(angle_avg))
            else:
                fitf = self._computational_data[deflection_id]['fitf']
                c, s, angle_avg = self._compute_deflection_integrals(
                    fitf_angle, fitf)
            if rot_id == 0:
                m_i, s_i = 1.0, 0.0  # Fixed bond
            else:
                rot_info = self.rotation_labels.get(rot_id, {})
                is_ris = rot_info.get('type') == 'ris'
                if is_ris:
                    # RIS model
                    if rot_id not in self.ris_data:
                        print(
                            f"Warning: No data for RIS ID {rot_id}. Assuming a rigid bond (m=1, s=0)."
                        )
                        m_i, s_i = 1.0, 0.0
                    else:
                        if rot_id not in ris_cache:
                            angles_deg, energies = self.ris_data[rot_id]
                            m_i, s_i = tool.compute_ris_rotation_integrals(
                                angles_deg, energies, self.kTval)
                            ris_cache[rot_id] = (m_i, s_i)
                        else:
                            m_i, s_i = ris_cache[rot_id]
                else:
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

            # Dihedral rotation matrix (around x-axis)
            R_x = np.array([[1, 0.0, 0.0], [0.0, m_i, -s_i], [0.0, s_i, m_i]])
            # Bond angle deflection matrix (around z-axis)
            R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])

            A_list.append(R_z @ R_x)
            avg_angles.append(angle_avg)

        # Multiply all transformation matrices for the repeat unit
        Mmat = np.eye(3)
        if len(A_list) > 1:
            for A in A_list[1:]:
                Mmat = Mmat @ A
        Mmat = Mmat @ A_list[0]
        self._Mmat = Mmat
        self._avg_angles = avg_angles
        self._A_list = A_list

    def build_G_unit(self):
        """
        Builds the unit transfer matrix G_unit for the polymer chain.
        
        Returns:
            np.ndarray: The unit transfer matrix G_unit.
        """
        num_bonds = len(self.bond_lengths)
        if self._A_list is None:
            self._calculate_Mmat()
        avg_matrices = self._A_list
        G_unit = np.eye(5)
        for i in range(num_bonds):
            # current bond length
            l_vec = np.array([self.bond_lengths[i], 0.0, 0.0])
            l_sq = self.bond_lengths[i]**2
            next_idx = (i + 1) % num_bonds
            T_next = avg_matrices[next_idx]
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
        if self._G_unit is None:
            self.build_G_unit()
        if self._lambda_max is None:
            self.run_calculation()
        G_unit = self._G_unit
        M = G_unit[1:4, 1:4]
        p = G_unit[1:4, 4]
        n_vec = G_unit[0, 1:4]
        s = G_unit[0, 4]
        if np.abs(self._lambda_max - 1.0) < 1e-10:
            return np.inf
        I = np.eye(3)
        try:
            inv_I_minus_M = np.linalg.inv(I - M)
        except np.linalg.LinAlgError:
            return np.inf
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
        s = G[0, 4]
        n_vec = G[0, 1:4]
        p_vec = G[1:4, 4]
        M = G[1:4, 1:4]
        I = np.eye(3)
        try:
            inv_I_M = np.linalg.inv(I - M)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Matrix (I - M) is singular. Chain might be perfectly rigid or linear."
            )
        A = s + n_vec @ inv_I_M @ p_vec
        B = -n_vec @ (inv_I_M @ inv_I_M) @ p_vec
        if np.abs(A) < 1e-10:
            return
        Np = -B / A
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
    def average_angles(self):
        if self._avg_angles is None:
            self._calculate_Mmat()
        return self._avg_angles

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
        print("-----------------------------------------------")

    def report_average_angles(self, round=2):
        avg = self.average_angles
        print("---- Average Deflection Angles ----")
        print(", ".join([f"{x:.{round}f}" for x in avg]))
        print("-----------------------------------")

    def plot_deflection_angles(self):
        original = self._read_data(Path(self.bond_angle_file))
        fit_angle = self._fit_deflection(Path(self.bond_angle_file))
        colors = plt.get_cmap('tab20')
        x = np.linspace(0, 360, 721)
        for i in range(len(fit_angle)):
            fitf = fit_angle[i]
            color = colors((i % 20) / 20.0)
            plt.plot(original[:, 0],
                     original[:, i + 1],
                     linestyle='none',
                     c=color)
            plt.plot(x, fitf(x), label=f"{i+1}", c=color)
        tool.format_subplot("Dihedral Angle (°)", "Deflection Angle (°)",
                            "Variable Deflection Angle")
        plt.show()

    def temperature_scan(self, T_list, plot=False):
        """
        T_list: iterable of temperatures (K)
        Returns: dict with keys 'T', 'corr', 'Mmat'
        """
        Ts = np.asarray(T_list, dtype=np.float64)
        results = {'T': Ts, 'corr': [], 'Mmat': [], 'average_angles': []}

        kT_orig = self.kTval
        for T in Ts:
            self.kTval = sc.R * T / 1000.0  # kJ/mol
            # recompute M matrix (vectorized)
            try:
                self._calculate_Mmat()
                results['Mmat'].append(self._Mmat)
                results['average_angles'].append(self._avg_angles)
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
