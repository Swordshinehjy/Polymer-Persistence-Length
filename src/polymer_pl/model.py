from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.constants as sc
from joblib import Parallel, delayed
from numpy.linalg import eigvals
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from scipy.linalg import fractional_matrix_power
from typing import List, Tuple, Union

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
        # --- Internal cache for lazy evaluation ---
        self._Mmat = None
        self._lambda_max = None
        self._lp_in_repeats = None
        self._computational_data = {}
        self._full_data = {}
        self.fitting_method = fitting_method
        self.param_n = param_n

    @staticmethod
    def _read_ris_data(file_name: Path):
        delimiter = ',' if file_name.suffix == '.csv' else None
        data = np.loadtxt(file_name, delimiter=delimiter)
        data = np.reshape(data, (-1, 2))
        data = np.unique(data, axis=0)
        return data[:, 0], data[:, 1]

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

    def _compute_ris_rotation_integrals(self, angles_deg, energies):
        """Compute rotation integrals for RIS model using discrete states."""
        angles_rad = np.deg2rad(angles_deg)

        boltzmann_weights = np.exp(-energies / self.kTval)
        Z = np.sum(boltzmann_weights)

        probabilities = boltzmann_weights / Z
        m_i = np.sum(probabilities * np.cos(angles_rad))
        s_i = np.sum(probabilities * np.sin(angles_rad))
        return m_i, s_i

    def _calculate_Mmat(self):
        """Constructs the overall transformation matrix M for the repeat unit."""
        self._prepare_computational_data()

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
                            angles, energies = self._read_ris_data(
                                Path(info['loc']))
                        self.ris_data[ris_id] = (angles, energies)
                    except FileNotFoundError:
                        print(
                            f"Warning: RIS data file not found. Skipping RIS type {ris_id}."
                        )
                        continue

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
                        m_i, s_i = self._compute_ris_rotation_integrals(
                            angles_deg, energies)
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

            A_list.append(R_x @ R_z)

        # Multiply all transformation matrices for the repeat unit
        Mmat = np.eye(3)
        for A in A_list:
            Mmat = A @ Mmat
        self._Mmat = Mmat

    def run_calculation(self):
        """
        Runs the full calculation to find the persistence length.
        This method populates the result attributes.
        """
        if self._Mmat is None:
            self._calculate_Mmat()

        eigs = eigvals(self._Mmat)
        self._lambda_max = float(np.max(np.abs(eigs)))

        if self._lambda_max >= 1.0:
            self._lp_in_repeats = np.inf
        else:
            self._lp_in_repeats = -1.0 / np.log(self._lambda_max)

    @property
    def persistence_length_repeats(self):
        """The persistence length in units of repeat units."""
        if self._lp_in_repeats is None:
            self.run_calculation()
        return self._lp_in_repeats

    @property
    def persistence_length(self):
        """The persistence length."""
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
        return self.persistence_length_repeats * np.sum(self.bond_lengths)

    @property
    def kuhn_length(self):
        """The Kuhn length."""
        return self.persistence_length * 2

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

    def format_subplot(self, xlabel, ylabel, title):
        """Format subplot with consistent styling."""
        plt.xlabel(xlabel, fontsize=16, fontfamily="Helvetica")
        plt.ylabel(ylabel, fontsize=16, fontfamily="Helvetica")
        plt.xticks(fontsize=14, fontfamily="Helvetica")
        plt.yticks(fontsize=14, fontfamily="Helvetica")
        # Add legend only if there are labeled elements
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(fontsize=14, prop={'family': 'Helvetica'})
        plt.grid(True, alpha=0.3)
        plt.minorticks_on()
        plt.title(title, fontsize=18, fontfamily="Helvetica")

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
        self.format_subplot("Dihedral Angle [Deg.]",
                            "Dihedral Potential (kJ/mol)",
                            "Dihedral Potentials")
        plt.subplot(1, 3, 2)
        for key, data in self._full_data.items():
            plt.plot(data['x_values'],
                     data['prob_vals'],
                     color=f"{data['color']}",
                     linestyle="-",
                     label=data['label'])
        self.format_subplot("Angle [deg.]", "Probability",
                            "Probability Distributions")
        plt.subplot(1, 3, 3)
        for key, data in self._full_data.items():
            plt.plot(data['cum_dist'] / data['cum_dist'][-1],
                     data['x_values'],
                     color=f"{data['color']}",
                     linestyle="-",
                     label=data['label'])
        self.format_subplot("Probability", "Dihedral Angle [deg.]",
                            "Cumulative Probability Distributions")

        plt.tight_layout()
        plt.show()

    def report(self):
        """Prints a summary of the calculation results."""
        # Ensure calculation has been run
        lp = self.persistence_length_repeats
        lam = self.lambda_max
        print("---- Persistence Length Calculation Report ----")
        print(f"Temperature: {self.temperature} K")
        print(f"Max Eigenvalue (lambda_max): {lam:.12f}")
        print(f"Persistence Length (in repeat units): {lp:.6f}")
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

    def pre_generate_angles(self, n_samples, flat_rotation):
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

        all_angles = self.pre_generate_angles(n_samples, flat_rotation)
        r2List = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(chain_rotation.batch_r2_cython)(
                np.ascontiguousarray(ch, dtype=np.float64),
                np.ascontiguousarray(all_angles[i * batch_size:(i + 1) *
                                                batch_size],
                                     dtype=np.float64),
                np.ascontiguousarray(flat_rotation, dtype=np.int64), length)
            for i in range(n_batches))
        r2List = np.vstack(r2List)
        msd_values = np.mean(r2List, axis=0)

        plt.figure(figsize=(6, 5))
        plt.plot(n_repeats, msd_values, linewidth=2, color='blue', marker='o')
        self.format_subplot("Number of Repeat Units (N)",
                            "Mean Square End-to-End Distance (Å²)",
                            "<R²> vs. Number of Repeat Units")
        plt.tight_layout()
        plt.show()

        if return_data:
            return msd_values

    def calculate_persistence_length_mc(self,
                                        n_repeat_units=20,
                                        n_samples=150000):
        """
        Calculate persistence length using Monte Carlo sampling.
        
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
            The calculated persistence length in units of repeat units.
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

        # Pre-generate all angles
        all_angles = self.pre_generate_angles(n_samples, flat_rotation)

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

        # Calculate persistence length
        corr2 = np.mean(cosList2, axis=0)
        repeat_units = np.arange(len(corr2))
        start_idx = 1
        end_idx = 10

        # Fit exponential decay to correlation function
        p = np.polynomial.polynomial.polyfit(repeat_units[start_idx:end_idx],
                                             np.log(corr2[start_idx:end_idx]),
                                             1)

        persistence_length = -1 / p[1]

        print(f"\nMonte Carlo Result:")
        print(f"slope: {p[1]:.6f}")
        print(f"Persistence Length: {persistence_length:.2f}")

        return persistence_length

    def _batch_cosVals_optimized(self, ch, all_angles, flat_rotation, length):
        """Batch processing function optimized with Cython."""
        if chain_rotation is None:
            raise ImportError("chain_rotation module not available")
        return chain_rotation.batch_cosVals_cython(
            np.ascontiguousarray(ch, dtype=np.float64),
            np.ascontiguousarray(all_angles, dtype=np.float64),
            np.ascontiguousarray(flat_rotation, dtype=np.int64), length)

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

        all_angles = self.pre_generate_angles(n_samples, flat_rotation)
        c = np.zeros((n_repeat_units, n_repeat_units))
        unit_idx = np.arange(0, n_repeat_units * length + 1, length)
        for i in range(n_samples):
            pos = chain_rotation.randomRotate_cython(ch, all_angles[i],
                                                     flat_rotation)
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

            # Pre-generate all angles
            all_angles = self.pre_generate_angles(n_samples, flat_rotation)

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
                    "Warning: Zero slope in fit, persistence length undefined."
                )
                persistence_length = np.inf
            else:
                persistence_length = -1 / p[1]

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
                     label=f'Np = {persistence_length:.5f}')
            self.format_subplot("Repeat Units", r'Ln[$<V_0 \cdot V_n>$]',
                                "Log of Correlation Function")
            plt.show()
            if return_data:
                return corr2

        except Exception as e:
            print(f"Error in plot_correlation_function: {str(e)}")
            return

    def temperature_scan(self, T_list, plot=False):
        """
        T_list: iterable of temperatures (K)
        Returns: dict with keys 'T', 'lp', 'Mmat'
        """
        Ts = np.asarray(T_list, dtype=np.float64)
        results = {'T': Ts, 'lp': [], 'Mmat': []}

        kT_orig = self.kTval
        for T in Ts:
            self.kTval = sc.R * T / 1000.0  # kJ/mol
            # clear cached integrals because they depend on kT
            # simplest approach: reset computational caches that depend on kT
            self._computational_data = {}
            self._full_data = {}

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
                lp = np.inf
            else:
                lp = -1.0 / np.log(lambda_max)
            results['lp'].append(lp)
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(Ts, results['lp'], 'o-')
            self.format_subplot("Temperature (K)",
                                "Persistence Length (repeat units)",
                                "Temperature Scan")
            plt.show()
        # restore original
        self.kTval = kT_orig
        self._calculate_Mmat()
        return results


def compute_persistence_terpolymer(Mmat, prob):
    """
    Computes persistence length for a terpolymer made of two repeat units 
    appearing with probability prob and 1-prob.

    Parameters:
    -----------
    Mmat : listlike
        List of transformation matrices for each repeat unit type
    prob : listlike
        List of probabilities for each repeat unit type

    Returns:
    --------
    float
        Persistence length in repeat units
    """
    # Type checking
    if not hasattr(Mmat, '__iter__') or not hasattr(prob, '__iter__'):
        raise TypeError("Both Mmat and prob must be iterable (listlike)")

    # Convert to lists to check length
    Mmat_list = list(Mmat)
    prob_list = list(prob)

    if len(Mmat_list) != len(prob_list):
        raise ValueError(
            f"Mmat and prob must have the same length, got {len(Mmat_list)} and {len(prob_list)}"
        )

    # Check that probabilities sum to approximately 1
    prob_sum = sum(prob_list)
    if not np.isclose(prob_sum, 1.0, rtol=1e-3):
        raise ValueError(f"Probabilities must sum to 1.0, got {prob_sum}")

    Mmat_avg = 0
    for mat, p in zip(Mmat_list, prob_list):
        Mmat_avg += p * mat

    eigs = eigvals(Mmat_avg)
    lambda_max = float(np.max(np.abs(eigs)))

    if lambda_max >= 1.0:
        return np.inf, 1.0

    lp_in_repeats = -1.0 / np.log(lambda_max)
    return lp_in_repeats


def compute_persistence_terpolymer_Tscan(polymer_models,
                                         prob_list,
                                         T_list,
                                         plot=True) -> np.ndarray:
    """
    Computes persistence length for a terpolymer across a range of temperatures.
    
    This function integrates temperature_scan and compute_persistence_terpolymer
    to calculate how the persistence length of a terpolymer changes with temperature.
    
    Parameters:
    -----------
    polymer_models : listlike
        List of PolymerPersistence objects for each repeat unit type
    prob_list : listlike
        List of probabilities for each repeat unit type (must sum to 1.0)
        example [[0, 1], [0.5, 0.5], [1, 0]]
    T_list : listlike
        List of temperatures (in Kelvin) to evaluate
    plot : bool, optional
        Whether to plot the 2D results, by default True
        
    Returns:
    --------
    2D numpy array, row: temperature, column: persistence length
    """
    # Type checking
    if not hasattr(polymer_models, '__iter__'):
        raise TypeError("polymer_models must be iterable")

    # Convert to lists
    model_list = list(polymer_models)
    prob = np.asarray(prob_list, dtype=np.float64)  # (P, K)
    Ts = np.atleast_1d(T_list).astype(np.float64)  # (N,)
    P, K = prob.shape
    N = len(Ts)
    if K != len(model_list):
        raise ValueError(
            "prob_list column count must match number of polymer models")

    # Validate probability normalization
    if not np.allclose(prob.sum(axis=1), 1.0, rtol=1e-3):
        raise ValueError("Each probability row must sum to 1.")
    # 1. Collect all M matrices at all temperatures
    #    mat_list[k] = (N, 3, 3)
    mats = np.stack([m.temperature_scan(Ts)['Mmat'] for m in model_list],
                    axis=0)  # (K, N, 3, 3)

    # 2. Weighted combination by prob (vectorized)
    #    For each probability set p (shape P,K):
    #    M_avg[p,n,:,:] = sum_k p[p,k] * mats[k,n,:,:]
    # prob[:, :, None, None] → (P,K,1,1)   broadcast
    # mats[None, :, :, :, :] → (1,K,N,3,3)
    M_avg = (prob[:, :, None, None, None] * mats[None]).sum(
        axis=1)  # (P, N, 3, 3)
    eigs = np.linalg.eigvals(M_avg)  # (P,N,3)
    lambda_max = np.max(np.abs(eigs), axis=-1)  # (P,N)
    lp = np.empty_like(lambda_max)

    mask_bad = lambda_max >= 1.0
    mask_good = ~mask_bad

    lp[mask_good] = -1.0 / np.log(lambda_max[mask_good])
    lp[mask_bad] = np.inf
    lp = lp.T
    if plot:
        if P == 1 and N == 1:
            # report
            print("---- Persistence Length Calculation Report ----")
            print(f"Temperature: {Ts[0]:.2f} K")
            print(f"Max Eigenvalue (lambda_max): {lambda_max[0, 0]:.12f}")
            print(f"Persistence Length (in repeat units): {lp[0, 0]:.6f}")
            print("-----------------------------------------------")
        elif P == 1:
            # 1D plot: persistence vs temperature (single composition)
            lp_1d = lp[:, 0]
            finite_mask = np.isfinite(lp_1d)
            plt.figure(figsize=(6, 5))
            plt.plot(Ts[finite_mask], lp_1d[finite_mask], 'o-')
            if not np.all(finite_mask):
                # Optionally mark infinities (e.g., as flat line or annotation)
                pass
            plt.xlabel("Temperature (K)", fontsize=16, fontfamily="Helvetica")
            plt.ylabel("$N_p$", fontsize=16, fontfamily="Helvetica")
            plt.title("Persistence Length vs Temperature",
                      fontsize=18,
                      fontfamily="Helvetica")
            plt.xticks(fontsize=14, fontfamily="Helvetica")
            plt.yticks(fontsize=14, fontfamily="Helvetica")
            plt.grid(True)
            plt.tight_layout()
            plt.minorticks_on()
            plt.show()
        elif N == 1:
            # Fixed T, vary composition → 1D curve: lp vs composition
            lp_1d = lp[0, :]  # shape (P,)
            # Use first component probability as x-axis (assuming K >= 1)
            x = prob[:, 0]  # probability of first monomer
            finite = np.isfinite(lp_1d)
            plt.figure(figsize=(6, 5))
            plt.plot(x[finite], lp_1d[finite], 'o-')
            plt.xlabel("Probability of Repeat Unit 1",
                       fontsize=16,
                       fontfamily="Helvetica")
            plt.ylabel("$N_p$", fontsize=16, fontfamily="Helvetica")
            plt.title(f"$N_p$ vs Composition (T = {Ts[0]:.2f} K)",
                      fontsize=18,
                      fontfamily="Helvetica")
            plt.xticks(fontsize=14, fontfamily="Helvetica")
            plt.yticks(fontsize=14, fontfamily="Helvetica")
            plt.grid(True)
            plt.tight_layout()
            plt.minorticks_on()
            plt.show()
        else:
            lp_plot = lp.copy()
            lp_plot[np.isinf(lp_plot)] = np.nan  # Mask inf for display
            prob_first_component = prob[:, 0]
            plt.figure(figsize=(6, 5))
            im = plt.imshow(lp_plot,
                            aspect='auto',
                            origin='lower',
                            extent=[
                                prob_first_component.min(),
                                prob_first_component.max(),
                                Ts.min(),
                                Ts.max()
                            ],
                            cmap='viridis',
                            interpolation='bicubic')

            cbar = plt.colorbar(im)
            cbar.set_label("Persistence length",
                           fontsize=14,
                           fontfamily="Helvetica")
            cbar.ax.tick_params(labelsize=14)
            plt.setp(cbar.ax.get_yticklabels(), fontfamily="Helvetica")

            X, Y = np.meshgrid(prob_first_component, Ts)
            if np.any(np.isfinite(lp_plot)):
                CS = plt.contour(X, Y, lp_plot, colors='white', alpha=0.5)
                plt.clabel(CS, inline=True, fontsize=8, fmt="%.1f")
            plt.ylabel("Temperature (K)", fontsize=16, fontfamily="Helvetica")
            plt.xlabel("Probability of Repeat Unit 1",
                       fontsize=16,
                       fontfamily="Helvetica")
            plt.title("Terpolymer Persistence Length",
                      fontsize=18,
                      fontfamily="Helvetica")
            plt.xticks(fontsize=14, fontfamily="Helvetica")
            plt.yticks(fontsize=14, fontfamily="Helvetica")
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()
    return lp


def inverse_data(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    delimiter = ',' if filename.suffix == '.csv' else None
    data = np.loadtxt(filename, delimiter=delimiter)
    data = data[np.argsort(data[:, 0])]
    data_new = np.column_stack((data[:, 0][::-1], data[:, 1]))
    np.savetxt(filename.stem + "-inv.txt", data_new)


def compute_persistence_alternating(model1, model2, temperature, plot=True):
    """
    Compute persistence length for alternating matrices.
    
    Parameters:
    -----------
    model1 : PolymerPersistence
        First model
    model2 : PolymerPersistence
        Second model
    tempeture : float or list
    plot : bool, optional
    Returns:
    --------
    tuple
        (persistence length, maximum eigenvalue)
    """
    # Normalize input temperature to array
    is_scalar = np.isscalar(temperature) or (hasattr(temperature, '__len__')
                                             and len(temperature) == 1)
    T_arr = np.atleast_1d(temperature).astype(np.float64)
    N = len(T_arr)
    # Run temperature scans for both models
    res1 = model1.temperature_scan(T_arr)
    res2 = model2.temperature_scan(T_arr)
    M1_all = np.array(res1["Mmat"])  # shape (N, 3, 3)
    M2_all = np.array(res2["Mmat"])  # shape (N, 3, 3)
    M_combined = np.einsum("nij,njk->nik", M1_all, M2_all)  # (N,3,3)
    # Compute fractional matrix power: (M_B @ M_A)^{1/2}
    M_avg = np.zeros_like(M_combined)
    for i in range(N):
        try:
            M_avg[i] = fractional_matrix_power(M_combined[i], 0.5)
        except Exception as e:
            raise RuntimeError(
                f"Failed to compute matrix square root at T={T_arr[i]} K: {e}")
    # Compute eigenvalues and lambda_max
    eigs = np.linalg.eigvals(M_avg)  # (N, 3)
    lambda_max = np.max(np.abs(eigs), axis=-1)  # (N,)

    lp = np.empty_like(lambda_max)
    mask_good = lambda_max < 1.0
    mask_bad = ~mask_good

    lp[mask_good] = -1.0 / np.log(lambda_max[mask_good])
    lp[mask_bad] = np.inf

    if is_scalar:
        T_val = float(T_arr[0])
        lp_val = lp[0]
        lambda_val = lambda_max[0]
        print("---- Alternating Copolymer Persistence Length Report ----")
        print(f"Temperature: {T_val:.2f} K")
        print(f"Max Eigenvalue (λ_max): {lambda_val:.12f}")
        if np.isinf(lp_val):
            print("Persistence Length: ∞ (rigid or semi-flexible limit)")
        else:
            print(f"Persistence Length (in repeat units): {lp_val:.6f}")
        print("---------------------------------------------------------")

        return lp_val
    else:
        if plot:
            plt.figure(figsize=(6, 5))
            finite_mask = np.isfinite(lp)
            if np.any(finite_mask):
                plt.plot(T_arr[finite_mask],
                         lp[finite_mask],
                         'o-',
                         color='tab:blue')
            if np.any(~finite_mask):
                pass
            plt.xlabel("Temperature (K)", fontsize=14, fontfamily="Helvetica")
            plt.ylabel("$N_p$", fontsize=14, fontfamily="Helvetica")
            plt.title("Alternating Copolymer $N_p$ vs. Temperature",
                      fontsize=18,
                      fontfamily="Helvetica")
            plt.xticks(fontsize=14, fontfamily="Helvetica")
            plt.yticks(fontsize=14, fontfamily="Helvetica")
            plt.grid(True, alpha=0.3)
            plt.minorticks_on()
            plt.tight_layout()
            plt.show()

        return lp


def compare_persistence_results(models: List[PolymerPersistence],
                                labels: List[str],
                                temperature: Union[float, List[float]],
                                property='lp'):
    '''
    Compare persistence results between different models.
    Arguments:
        models: List of persistence models.
        labels: List of labels for the models.
        ts: List of temperature arrays.
        property: Property to compare, e.g., 'lp'.
    '''
    T_arr = np.atleast_1d(temperature).astype(np.float64)
    plt.figure(figsize=(6, 5))
    for model, label in zip(models, labels):
        res = model.temperature_scan(T_arr)
        plt.plot(res['T'], res[property], 'o-', label=label)

    plt.xlabel('Temperature (K)', fontsize=16, fontfamily="Helvetica")
    if property == 'lp':
        ylabel = "$N_p$"
        title = "Persistence Length in Repeat Units"
    else:
        raise ValueError(f"Unknown property: {property}")
    plt.legend()
    plt.ylabel(ylabel, fontsize=16, fontfamily="Helvetica")
    plt.xticks(fontsize=14, fontfamily="Helvetica")
    plt.yticks(fontsize=14, fontfamily="Helvetica")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(fontsize=14, prop={'family': 'Helvetica'})
    plt.grid(True, alpha=0.3)
    plt.minorticks_on()
    plt.title(title, fontsize=18, fontfamily="Helvetica")
    plt.tight_layout()
    plt.show()


# ======================================================================
#                            USAGE EXAMPLE
# ======================================================================
# if __name__ == '__main__':
#     l = [4.289, 1.375, 4.289, 1.459, 2.510, 1.441, 2.510, 1.459]
#     Angle = np.array([-3.8, 38.1, -38.1, 3.8, 15.3,13.3,-13.3,-15.3 ])
#     rotation = np.array([0, 0, 0, 1, 0, 2, 0, 1])
#     labels = {
#         1: {
#             'loc': 'IID-FT.txt',
#             'color': 'b'
#         },
#         2: {
#             'loc': 'FT-FT.txt',
#             'color': 'm'
#         },

#     }

#     ris = np.array([0, 3, 0, 0, 0, 0, 0, 0])
#     ris_label = {
#         3: {'loc': 'EIID-RIS.txt', 'color': 'c'},
#         }

#     a = PolymerPersistence(l, Angle, rotation_types=rotation, rotation_labels=labels, ris_types=ris, ris_labels=ris_label)
#     a.report()
