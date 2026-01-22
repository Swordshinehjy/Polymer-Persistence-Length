from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from numpy.linalg import eigvals
from scipy.integrate import dblquad, quad
from scipy.interpolate import (RectBivariateSpline, RegularGridInterpolator,
                               interp1d)
from . import tool


class PolymerPersistenceDependentDihedral:
    """
    Calculates the persistence length of a polymer chain with both independent
    and dependent dihedral angle potentials using dblquad for 2D integration.
    
    This class handles cases where adjacent dihedral angles are coupled
    and have a 2D potential energy surface U(phi1, phi2).
    """

    def __init__(self,
                 bond_lengths,
                 bond_angles_deg,
                 temperature=300.0,
                 rotation_types=None,
                 rotation_labels=None,
                 coupled_pairs=None,
                 coupled_labels=None,
                 fitting_method='interpolation',
                 param_n=5,
                 vectorized=False,
                 method='linear'):
        """
        Initializes the PolymerPersistenceDependentDihedral model with dependent dihedrals.
        
        Args:
            bond_lengths (list or np.ndarray): The lengths of the bonds in the repeat unit.
            bond_angles_deg (list or np.ndarray): The deflection angles between bonds in degrees.
            temperature (int, optional): The temperature in Kelvin. Defaults to 300.
            rotation_types (list or np.ndarray, optional): An array of integers mapping each bond to a
                                                 specific rotational potential profile. A value of 0
                                                 indicates a fixed bond with no rotation.
            rotation_labels (dict, optional): A dictionary mapping rotation_types to data files.
            coupled_pairs (list of tuples, optional): List of tuples indicating which dihedrals are coupled.
                                                     Each tuple contains indices of coupled dihedrals.
            coupled_labels (dict, optional): Dictionary mapping coupled pair indices to 2D data files.
            fitting_method (str, optional): The method used for fitting the data. 'interpolation', 'cosine' or 'fourier'.
            param_n (int, optional): The order for fitting the data.
        """
        self.bond_lengths = np.asarray(
            bond_lengths) if bond_lengths is not None else None
        self.bond_angles_rad = np.deg2rad(np.array(bond_angles_deg))
        self.rotation_types = np.array(
            rotation_types) if rotation_types is not None else None
        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

        # For dependent dihedrals
        self.coupled_pairs = coupled_pairs if coupled_pairs is not None else []
        self.coupled_labels = coupled_labels if coupled_labels is not None else {}

        # Default labels mapping rotation_types to data files
        self.rotation_labels = rotation_labels if rotation_labels is not None else {}
        for rot_id, info in self.rotation_labels.items():
            if 'type' not in info:
                self.rotation_labels[rot_id]['type'] = 'continuous'
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

        # Process coupled labels
        for cp_id in self.coupled_labels:
            cp = self.coupled_labels[cp_id]
            if 'loc' in cp and 'label' not in cp:
                file_path = cp['loc']
                self.coupled_labels[cp_id]['label'] = Path(file_path).stem
        self.vectorized = vectorized
        self.method = method

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
        self._coupled_data = {}
        self._full_data = {}
        self.fitting_method = fitting_method
        self.param_n = param_n

        # Build mapping of dihedral index to its type
        if self.rotation_types is not None:
            self.dihedral_types = {}
            for i, rot_type in enumerate(self.rotation_types):
                self.dihedral_types[i] = {
                    'type': 'independent',
                    'rot_id': int(rot_type),
                    'coupled_with': None
                }

            # Update coupled dihedrals
            for cp_idx, (i, j) in enumerate(self.coupled_pairs):
                if i < len(self.rotation_types) and j < len(
                        self.rotation_types):
                    self.dihedral_types[i] = {
                        'type': 'coupled',
                        'pair_id': cp_idx,
                        'coupled_with': j,
                        'position': 0  # First in the pair
                    }
                    self.dihedral_types[j] = {
                        'type': 'coupled',
                        'pair_id': cp_idx,
                        'coupled_with': i,
                        'position': 1  # Second in the pair
                    }

    @staticmethod
    def _read_2d_data(file_name: Path):
        """Reads 2D dihedral angle data from a file."""
        delimiter = ',' if file_name.suffix == '.csv' else None
        data = np.loadtxt(file_name, delimiter=delimiter)
        data = data - data.min()
        # Matrix format - assume square and equally spaced
        phi = np.linspace(0, 360, data.shape[0])
        psi = np.linspace(0, 360, data.shape[1])
        return phi, psi, data

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

    def _prepare_independent_data(self):
        """Sets up interpolation functions for independent dihedrals."""
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

    def _get_phi_psi_energy(self, info):
        if 'data' in info:
            # Assume data is in format (phi1, phi2, energy)
            data = np.asarray(info['data'])
            energy_grid = data - data.min()
            phi1 = np.linspace(0, 360, data.shape[0])
            phi2 = np.linspace(0, 360, data.shape[1])
        elif 'loc' in info:
            phi1, phi2, energy_grid = self._read_2d_data(Path(info['loc']))
        else:
            raise ValueError(
                f"Either 'data' or 'loc' must be provided for coupled pair.")
        return phi1, phi2, energy_grid

    def _prepare_coupled_data(self):
        """Sets up interpolation functions for coupled dihedrals."""
        for cp_id, info in self.coupled_labels.items():
            try:
                phi1, phi2, energy_grid = self._get_phi_psi_energy(info)
                # Create 2D interpolation function for dblquad
                # We need a function that takes (phi, psi) and returns energy
                interp = RegularGridInterpolator((phi1, phi2),
                                                 energy_grid,
                                                 method=self.method,
                                                 bounds_error=False,
                                                 fill_value=None)

                # Create a wrapper function for dblquad
                def energy_func(phi, psi):
                    # Convert to array for interpolation
                    return interp([[phi, psi]])[0]

                # Store the interpolation function and original data
                self._coupled_data[cp_id] = {
                    'energy_func': energy_func,
                    'interp': interp,
                    'phi1': phi1,
                    'phi2': phi2,
                    'energy_grid': energy_grid,
                    **info
                }

            except FileNotFoundError:
                print(
                    f"Warning: Data file not found. Skipping coupled pair {cp_id}."
                )
                continue

    def _prepare_computational_data(self):
        """Sets up interpolation functions for all data."""
        if self._computational_data and self._coupled_data:
            return

        # Prepare independent dihedrals
        if self.rotation_labels:
            self._prepare_independent_data()

        # Prepare coupled dihedrals
        if self.coupled_labels:
            self._prepare_coupled_data()

    def _compute_independent_rotation_integrals(self, fitf, limit=1000):
        """Calculates the Boltzmann-averaged <cos(phi)> and <sin(phi)> for independent dihedrals."""

        def w(phi_deg):
            return np.exp(-fitf(phi_deg) / self.kTval)

        Z, _ = quad(w, 0, 360, limit=limit)
        if Z == 0:
            return 0.0, 0.0

        cos_avg, _ = quad(lambda x: np.cos(np.deg2rad(x)) * w(x),
                          0,
                          360,
                          limit=limit)
        sin_avg, _ = quad(lambda x: np.sin(np.deg2rad(x)) * w(x),
                          0,
                          360,
                          limit=limit)
        return cos_avg / Z, sin_avg / Z

    def _compute_coupled_rotation_integrals(self, cp_id):
        """
        Calculates the Boltzmann-averaged <cos(phi)>, <sin(phi)>, <cos(psi)>, <sin(psi)> 
        for coupled dihedrals using dblquad.
        
        Args:
            cp_id: Coupled pair identifier
        
        Returns:
            Tuple of (cos_phi, sin_phi, cos_psi, sin_psi)
        """
        if cp_id not in self._coupled_data:
            return 1.0, 0.0, 1.0, 0.0  # Default to rigid bonds

        cp_data = self._coupled_data[cp_id]
        energy_func = cp_data['energy_func']

        # Define Boltzmann factor function
        def boltzmann_factor(phi, psi):
            return np.exp(-energy_func(phi, psi) / self.kTval)

        # Compute normalization constant Z
        Z, z_err = dblquad(boltzmann_factor,
                           0,
                           360,
                           lambda x: 0,
                           lambda x: 360,
                           epsabs=1e-6,
                           epsrel=1e-6)

        if Z == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Compute <cos(phi)>
        def integrand_cos_phi(phi, psi):
            return np.cos(np.deg2rad(phi)) * boltzmann_factor(phi, psi)

        cos_phi, cos_phi_err = dblquad(integrand_cos_phi,
                                       0,
                                       360,
                                       lambda x: 0,
                                       lambda x: 360,
                                       epsabs=1e-6,
                                       epsrel=1e-6)
        cos_phi_avg = cos_phi / Z

        # Compute <sin(phi)>
        def integrand_sin_phi(phi, psi):
            return np.sin(np.deg2rad(phi)) * boltzmann_factor(phi, psi)

        sin_phi, sin_phi_err = dblquad(integrand_sin_phi,
                                       0,
                                       360,
                                       lambda x: 0,
                                       lambda x: 360,
                                       epsabs=1e-6,
                                       epsrel=1e-6)
        sin_phi_avg = sin_phi / Z

        # Compute <cos(psi)>
        def integrand_cos_psi(phi, psi):
            return np.cos(np.deg2rad(psi)) * boltzmann_factor(phi, psi)

        cos_psi, cos_psi_err = dblquad(integrand_cos_psi,
                                       0,
                                       360,
                                       lambda x: 0,
                                       lambda x: 360,
                                       epsabs=1e-6,
                                       epsrel=1e-6)
        cos_psi_avg = cos_psi / Z

        # Compute <sin(psi)>
        def integrand_sin_psi(phi, psi):
            return np.sin(np.deg2rad(psi)) * boltzmann_factor(phi, psi)

        sin_psi, sin_psi_err = dblquad(integrand_sin_psi,
                                       0,
                                       360,
                                       lambda x: 0,
                                       lambda x: 360,
                                       epsabs=1e-6,
                                       epsrel=1e-6)
        sin_psi_avg = sin_psi / Z

        return cos_phi_avg, sin_phi_avg, cos_psi_avg, sin_psi_avg

    def _calculate_Mmat(self):
        """Constructs the overall transformation matrix M for the repeat unit."""
        self._prepare_computational_data()

        M = len(self.rotation_types)
        A_list = []
        integral_cache = {}
        ris_cache = {}
        coupled_cache = {}  # Cache for coupled dihedral averages

        for i in range(M):
            rot_id = int(self.rotation_types[i])
            theta = float(self.bond_angles_rad[i])

            # Check if this dihedral is part of a coupled pair
            dihedral_info = self.dihedral_types.get(i, {})
            if dihedral_info.get('type') == 'coupled':
                cp_id = dihedral_info['pair_id']
                position = dihedral_info['position']

                # Compute averages for the coupled pair if not already done
                if cp_id not in coupled_cache:
                    cos_phi, sin_phi, cos_psi, sin_psi = self._compute_coupled_rotation_integrals_fast(
                        cp_id
                    ) if self.vectorized else self._compute_coupled_rotation_integrals(
                        cp_id)
                    coupled_cache[cp_id] = {
                        0: (cos_phi, sin_phi),  # First dihedral in pair
                        1: (cos_psi, sin_psi)  # Second dihedral in pair
                    }

                # Get the appropriate values based on position
                m_i, s_i = coupled_cache[cp_id][position]

            elif rot_id == 0:
                m_i, s_i = 1.0, 0.0  # Fixed bond
            else:
                # Continuous rotation model
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
                    if rot_id not in self._computational_data:
                        print(
                            f"Warning: No data for rotation ID {rot_id}. Assuming a rigid bond (m=1, s=0)."
                        )
                        m_i, s_i = 1.0, 0.0
                    else:
                        if rot_id not in integral_cache:
                            fitf = self._computational_data[rot_id]['fitf']
                            integral_cache[
                                rot_id] = self._compute_independent_rotation_integrals(
                                    fitf)
                        m_i, s_i = integral_cache[rot_id]

            R_x = np.array([[1, 0.0, 0.0], [0.0, m_i, -s_i], [0.0, s_i, m_i]])

            c, s = np.cos(theta), np.sin(theta)
            R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])

            A_list.append(R_z @ R_x)
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
            return 0.0
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
        """The maximum eigenvalue of the transformation matrix."""
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

    def report(self):
        """Prints a summary of the calculation results."""
        # Ensure calculation has been run
        corr = self.correlation_length
        lam = self.lambda_max
        print("-------------- Calculation Report -------------")
        print(f"Temperature: {self.temperature} K")
        print(f"kT value: {self.kTval:.6f} kJ/mol")
        print(f"Max Eigenvalue (lambda_max): {lam:.12f}")
        print(f"Correlation Length: {corr:.6f}")
        if self.bond_lengths is not None:
            print(f"Average unit length: {self.average_unit_length:.6f} Å")
            print(
                f"Persistence Length Geometric (Angstroms): {self.persistence_length:.6f}"
            )
            print(
                f"Persistence Length WLC (Angstroms): {self.persistence_length_wlc:.6f}"
            )
            print(f"Conformational Parameter: {self.conformational_param:.6f}")
        print(f"Number of coupled dihedral pairs: {len(self.coupled_pairs)}")
        print("-----------------------------------------------")

    def visualize_coupled_potential(self,
                                    plot_type='2d',
                                    cmap='viridis',
                                    smooth=True,
                                    target_res=1000,
                                    sampling=10):
        """
        Visualize the 2D potential energy surface for a coupled dihedral pair.
        """
        for cp_id, info in self.coupled_labels.items():
            try:
                phi1, phi2, energy_grid = self._get_phi_psi_energy(info)
                if smooth:
                    k = 1 if self.method == 'linear' else 3
                    interp_func = RectBivariateSpline(phi1,
                                                      phi2,
                                                      energy_grid,
                                                      kx=k,
                                                      ky=k)
                    phi1_new = np.linspace(0, 360, target_res, endpoint=False)
                    phi2_new = np.linspace(0, 360, target_res, endpoint=False)
                    energy_grid_new = interp_func(phi1_new, phi2_new)
                    phi1, phi2, energy_grid = phi1_new, phi2_new, energy_grid_new
                if plot_type == '2d':
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(energy_grid,
                                   origin='lower',
                                   extent=[
                                       phi1.min(),
                                       phi1.max(),
                                       phi2.min(),
                                       phi2.max()
                                   ],
                                   cmap=cmap,
                                   interpolation='bicubic',
                                   aspect='equal')
                elif plot_type == '2d_contour':
                    fig, ax = plt.subplots(figsize=(6, 5))
                    x_grid, y_grid = np.meshgrid(phi1, phi2)
                    im = ax.contour(x_grid,
                                    y_grid,
                                    energy_grid,
                                    cmap=cmap,
                                    levels=20,
                                    extend='both')
                elif plot_type == '3d':
                    fig = plt.figure(figsize=(8, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    x_grid, y_grid = np.meshgrid(phi1, phi2)
                    im = ax.plot_surface(x_grid,
                                         y_grid,
                                         energy_grid,
                                         cmap=cmap,
                                         linewidth=0,
                                         antialiased=True,
                                         rstride=sampling,
                                         cstride=sampling)
                    ax.view_init(elev=30, azim=45)
                    ax.set_zlabel('Energy (kJ/mol)', fontsize=12)
                else:
                    raise ValueError("plot_type must be '2d' or '3d'")
                ax.set_xlabel('Dihedral 1 (degrees)', fontsize=14)
                ax.set_ylabel('Dihedral 2 (degrees)', fontsize=14)
                ax.set_title(
                    f'Potential Energy Surface for Coupled Pair {cp_id}',
                    fontsize=16)
                cbar = plt.colorbar(im)
                cbar.set_label('Energy (kJ/mol)', fontsize=14)
                ax.minorticks_on()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error plotting coupled pair {cp_id}: {e}")

    def visualize_coupled_boltzmann_distribution(self,
                                                 temperature=300.0,
                                                 plot_type='2d',
                                                 cmap='viridis',
                                                 smooth=True,
                                                 target_res=1000,
                                                 sampling=10):
        """
        Visualize the 2D potential energy surface for a coupled dihedral pair.
        """
        for cp_id, info in self.coupled_labels.items():
            try:
                phi1, phi2, energy_grid = self._get_phi_psi_energy(info)
                if smooth:
                    k = 1 if self.method == 'linear' else 3
                    interp_func = RectBivariateSpline(phi1,
                                                      phi2,
                                                      energy_grid,
                                                      kx=k,
                                                      ky=k)
                    phi1_new = np.linspace(0, 360, target_res, endpoint=False)
                    phi2_new = np.linspace(0, 360, target_res, endpoint=False)
                    energy_grid_new = interp_func(phi1_new, phi2_new)
                    phi1, phi2, energy_grid = phi1_new, phi2_new, energy_grid_new
                kt = sc.R * temperature / 1000
                boltzmann_dist = np.exp(-energy_grid / kt)
                boltzmann_dist /= np.sum(boltzmann_dist)
                if plot_type == '2d':
                    fig, ax = plt.subplots(figsize=(6, 5))
                    im = ax.imshow(boltzmann_dist,
                                   origin='lower',
                                   extent=[
                                       phi1.min(),
                                       phi1.max(),
                                       phi2.min(),
                                       phi2.max()
                                   ],
                                   cmap=cmap,
                                   interpolation='bicubic',
                                   aspect='equal')
                elif plot_type == '2d_contour':
                    fig, ax = plt.subplots(figsize=(6, 5))
                    x_grid, y_grid = np.meshgrid(phi1, phi2)
                    im = ax.contour(x_grid,
                                    y_grid,
                                    boltzmann_dist,
                                    cmap=cmap,
                                    levels=20,
                                    extend='both')
                elif plot_type == '3d':
                    fig = plt.figure(figsize=(8, 7))
                    ax = fig.add_subplot(111, projection='3d')
                    x_grid, y_grid = np.meshgrid(phi1, phi2)
                    im = ax.plot_surface(x_grid,
                                         y_grid,
                                         boltzmann_dist,
                                         cmap=cmap,
                                         linewidth=0,
                                         antialiased=True,
                                         rstride=sampling,
                                         cstride=sampling,
                                         shade=False,
                                         alpha=0.9)
                    ax.view_init(elev=30, azim=45)
                    ax.set_zlabel('Probability', fontsize=12)
                else:
                    raise ValueError("plot_type must be '2d' or '3d'")
                ax.set_xlabel('Dihedral 1 (degrees)', fontsize=14)
                ax.set_ylabel('Dihedral 2 (degrees)', fontsize=14)
                ax.set_title(
                    f'Boltzmann Distribution for Coupled Pair {cp_id}',
                    fontsize=16)
                cbar = plt.colorbar(im)
                cbar.set_label('Probability', fontsize=14)
                ax.minorticks_on()
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Error plotting coupled pair {cp_id}: {e}")

    def _upsample_energy_grid(self, cp_id, n_phi=720, n_psi=720):
        """
        Upsample a coarse 2D energy grid to a fine grid using cubic interpolation.

        Args:
            cp_id: coupled pair id
            n_phi, n_psi: number of points after upsampling (e.g., 360, 720, 1000)

        Returns:
            phi_fine, psi_fine, E_fine (new high-resolution energy grid)
        """

        cp = self._coupled_data[cp_id]
        E = cp["energy_grid"]
        phi1 = cp["phi1"]
        phi2 = cp["phi2"]

        # Define fine grid
        phi_fine = np.linspace(phi1.min(), phi1.max(), n_phi, endpoint=False)
        psi_fine = np.linspace(phi2.min(), phi2.max(), n_psi, endpoint=False)

        phi_fine_2d, psi_fine_2d = np.meshgrid(phi_fine,
                                               psi_fine,
                                               indexing="ij")

        # Build interpolator
        interp = cp["interp"]  # already a RegularGridInterpolator

        # Flatten the evaluation points
        pts = np.column_stack([phi_fine_2d.ravel(), psi_fine_2d.ravel()])

        # Interpolate
        E_fine = interp(pts).reshape(n_phi, n_psi)

        return phi_fine, psi_fine, E_fine

    def _compute_coupled_rotation_integrals_vectorized(self,
                                                       cp_id,
                                                       temperatures,
                                                       n_phi=721,
                                                       n_psi=721):
        """
        Vectorized + upsampled (fine grid) version.
        """

        # 1) Upsample coarse grid first
        phi, psi, E = self._upsample_energy_grid(cp_id,
                                                 n_phi=n_phi,
                                                 n_psi=n_psi)

        phi2d, psi2d = np.meshgrid(phi, psi, indexing='ij')
        phi2d_rad = np.deg2rad(phi2d)
        psi2d_rad = np.deg2rad(psi2d)

        dphi = (phi[-1] - phi[0]) / (len(phi) - 1)
        dpsi = (psi[-1] - psi[0]) / (len(psi) - 1)

        # 2) Vectorized Boltzmann factors
        kT = sc.R * np.asarray(temperatures) / 1000.0
        kT = kT[:, None, None]

        boltz = np.exp(-E[None, :, :] / kT)

        Z = boltz.sum(axis=(1, 2)) * dphi * dpsi

        cos_phi = ((np.cos(phi2d_rad)[None, :, :] * boltz).sum(axis=(1, 2)) *
                   dphi * dpsi / Z)
        sin_phi = ((np.sin(phi2d_rad)[None, :, :] * boltz).sum(axis=(1, 2)) *
                   dphi * dpsi / Z)
        cos_psi = ((np.cos(psi2d_rad)[None, :, :] * boltz).sum(axis=(1, 2)) *
                   dphi * dpsi / Z)
        sin_psi = ((np.sin(psi2d_rad)[None, :, :] * boltz).sum(axis=(1, 2)) *
                   dphi * dpsi / Z)

        return {
            "cos_phi": cos_phi,
            "sin_phi": sin_phi,
            "cos_psi": cos_psi,
            "sin_psi": sin_psi,
        }

    def _compute_coupled_rotation_integrals_fast(self, cp_id):
        # backward compatibility: single temperature
        T = self.temperature
        r = self._compute_coupled_rotation_integrals_vectorized(cp_id, [T])
        return (r["cos_phi"][0], r["sin_phi"][0], r["cos_psi"][0],
                r["sin_psi"][0])

    def temperature_scan(self, T_list, plot=False):
        """
        Computes the rotation integrals for a list of temperatures and
        returns the results.

        Args:
            T_list (list): A list of temperatures.
            plot (bool, optional): If True, plots the results. Defaults to False.

        Returns:
            dict: A dictionary containing the computed rotation integrals for each temperature.
        """
        Ts = np.asarray(T_list, dtype=np.float64)
        results = {'T': Ts, 'corr': [], 'Mmat': []}

        kT_orig = self.kTval
        vectorized_orig = self.vectorized
        self.vectorized = True
        for T in Ts:
            self.kTval = sc.R * T / 1000.0  # kJ/mol
            # recompute M matrix (vectorized)
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
        self.vectorized = vectorized_orig
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
            if rot_id == 0 or self.rotation_labels[rot_id]['type'] == 'ris':
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
