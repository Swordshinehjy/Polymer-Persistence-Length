from pathlib import Path
import psutil
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from numpy.linalg import eigvals
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

# Import the Cython module (after compilation)
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
    using the matrix transformation method.

    Attributes:
        bond_lengths (np.ndarray): Array of bond lengths.
        bond_angles_rad (np.ndarray): Array of bond angles in radians.
        rotation_types (np.ndarray): Array defining the type of rotational potential for each bond.
        temperature (float): The temperature in Kelvin.
        kTval (float): Boltzmann constant times temperature (in kJ/mol).
        lambda_max (float): The largest eigenvalue of the transformation matrix.
        persistence_length_repeats (float): The calculated persistence length in units of repeat units.
    """

    def __init__(self,
                 bond_lengths,
                 bond_angles_deg,
                 temperature=300.0,
                 rotation_types=None,
                 rotation_labels=None,
                 ris_types=None,
                 ris_labels=None,
                 fittting_method='intepolation',
                 cosine_deg=5):
        """
        Initializes the PolymerPersistence model.

        Args:
            bond_lengths (list or np.ndarray): The lengths of the bonds in the repeat unit.
            bond_angles_deg (list or np.ndarray): The deflection angles between bonds in degrees.
            rotation_types (list or np.ndarray): An array of integers mapping each bond to a
                                                 specific rotational potential profile. A value of 0
                                                 indicates a fixed bond with no rotation.
            temperature (int, optional): The temperature in Kelvin. Defaults to 300.
        """
        self.bond_lengths = np.array(bond_lengths)
        self.bond_angles_rad = np.deg2rad(np.array(bond_angles_deg))
        self.rotation_types = np.array(rotation_types)
        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

        # Default labels mapping rotation_types to data files
        self.rotation_labels = rotation_labels
        for rot_id in self.rotation_labels:
            file_path = self.rotation_labels[rot_id]['loc']
            self.rotation_labels[rot_id]['label'] = Path(file_path).stem
        self.ris_labels = ris_labels
        self.ris_types = ris_types
        if self.ris_labels is not None:
            for ris_id in self.ris_labels:
                file_path = self.ris_labels[ris_id]['loc']
                self.ris_labels[ris_id]['label'] = Path(file_path).stem
        # --- Internal cache for lazy evaluation ---
        self._Mmat = None
        self._lambda_max = None
        self._lp_in_repeats = None
        self._computational_data = {}
        self._full_data = {}
        self.fitting_method = fittting_method
        self.cosine_deg = cosine_deg

    @staticmethod
    def _read_data(file_name):
        """Reads and processes dihedral angle data from a file."""
        data = np.loadtxt(file_name)
        data = np.reshape(data, (-1, 2))
        if data[:, 0].max() - data[:, 0].min() != 360:
            # symmetric, only calculate half of the dihedral potential
            if data[:, 0].min() == 0 or data[:, 0].min(
            ) == 180:  # from 0 to 180 or 180 to 360
                mirrored = np.column_stack((-data[:, 0] + 360, data[:, 1]))
                combined = np.vstack((data, mirrored))
            elif data[:, 0].min() == -180:  # from -180 to 0
                mirrored = np.column_stack((-data[:, 0], data[:, 1]))
                combined = np.vstack((np.column_stack(
                    (mirrored, data[:, 0] + 360, data[:, 1]))))
            combined = np.unique(combined, axis=0)
            return combined[np.argsort(combined[:, 0])]
        else:  # full dihedral potential
            return data[np.argsort(data[:, 0])]

    @staticmethod
    def _read_ris_data(file_name):
        data = np.loadtxt(file_name)
        data = np.reshape(data, (-1, 2))
        data = np.unique(data, axis=0)
        return data[:, 0], data[:, 1]

    def _prepare_computational_data(self):
        """Sets up interpolation functions from data files."""
        if self._computational_data:
            return

        for rot_id, info in self.rotation_labels.items():
            try:
                data = self._read_data(Path(info['loc']))
                x, y = data[:, 0], data[:, 1]
                if self.fitting_method == 'intepolation':
                    fitf = interp1d(x,
                                    y,
                                    kind='cubic',
                                    fill_value="extrapolate")
                else:  # cosine
                    p = np.polynomial.polynomial.polyfit(
                        np.cos(np.deg2rad(x)), y, self.cosine_deg)
                    fitf = (lambda p_val: lambda z: np.polynomial.polynomial.
                            polyval(np.cos(np.deg2rad(z)), p_val))(p)
                self._computational_data[rot_id] = {'fitf': fitf, **info}
            except FileNotFoundError:
                print(
                    f"Warning: Data file {Path(info['loc'])} not found. Skipping rotation type {rot_id}."
                )
                continue
        # print("Computational data prepared.")

    def _prepare_full_data(self):
        """Sets up interpolation functions from data files."""
        if self._full_data:
            return
        for rot_id, info in self.rotation_labels.items():
            try:
                data = self._read_data(Path(info['loc']))
                x, y = data[:, 0], data[:, 1]
                if self.fitting_method == 'intepolation':
                    fitf = interp1d(x,
                                    y,
                                    kind='cubic',
                                    fill_value="extrapolate")
                else:  # cosine
                    p = np.polynomial.polynomial.polyfit(
                        np.cos(np.deg2rad(x)), y, self.cosine_deg)
                    fitf = (lambda p_val: lambda z: np.polynomial.polynomial.
                            polyval(np.cos(np.deg2rad(z)), p_val))(p)
                norm_val, _ = quad(lambda x: np.exp(-fitf(x) / self.kTval), 0,
                                   360)
                x_values = np.linspace(0, 360, 1000)
                prob_vals = np.exp(-fitf(x_values) / self.kTval) / norm_val
                cum_dist = cumulative_trapezoid(prob_vals, x_values, initial=0)
                inv_cdf = interp1d(cum_dist / cum_dist[-1],
                                   x_values,
                                   kind='cubic',
                                   fill_value="extrapolate")
                self._full_data[rot_id] = {
                    'fitf': fitf,
                    'data': data,
                    'prob_vals': prob_vals,
                    "inv_cdf": inv_cdf,
                    "x_values": x_values,
                    "cum_dist": cum_dist,
                    **info
                }
            except FileNotFoundError:
                print(
                    f"Warning: Data file {Path(info['loc'])} not found. Skipping rotation type {rot_id}."
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
                        angles, energies = self._read_ris_data(
                            Path(info['loc']))
                        self.ris_data[ris_id] = (angles, energies)
                    except FileNotFoundError:
                        print(
                            f"Warning: RIS data file {Path(info['loc'])} not found. Skipping RIS type {ris_id}."
                        )
                        continue
        else:
            self.ris_types = None

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
            S = np.array([[1, 0.0, 0.0], [0.0, m_i, -s_i], [0.0, s_i, m_i]])

            # Bond angle deflection matrix (around z-axis)
            c, s = np.cos(theta), np.sin(theta)
            R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])

            A_list.append(S @ R_z)

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

    def compute_mean_square_end_to_end(self, N):
        """Computes the mean square end-to-end distance for a given number of repeat units."""
        if self._lp_in_repeats is None:
            self.run_calculation()
        L_repeat = np.sum(self.bond_lengths)
        if self._lp_in_repeats == np.inf:
            L = N * L_repeat
            return L**2
        else:
            return 2 * self._lp_in_repeats * N * L_repeat**2 * (
                1 - self._lp_in_repeats / N *
                (1 - np.exp(-N / self._lp_in_repeats)))

    def plot_end_to_end_distance(self, N):
        """Plots the mean square end-to-end distance as a function of repeat units from 1 to N.

        Args:
            N (int): Maximum number of repeat units to plot
        """
        # Generate array of N values
        N_values = np.arange(1, N + 1)

        # Calculate mean square end-to-end distance for each N
        msd_values = [self.compute_mean_square_end_to_end(n) for n in N_values]

        # Create the plot
        plt.figure(figsize=(8, 6))
        plt.plot(N_values, msd_values, linewidth=2, color='blue', marker='o')

        # Format the plot
        self.format_subplot("Number of Repeat Units (N)",
                            "Mean Square End-to-End Distance (Å²)",
                            "End-to-End Distance vs. Number of Repeat Units")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def format_subplot(self, xlabel, ylabel, title):
        """Format subplot with consistent styling."""
        plt.xlabel(xlabel, fontsize=16, fontfamily="Helvetica")
        plt.ylabel(ylabel, fontsize=16, fontfamily="Helvetica")
        plt.xticks(fontsize=14, fontfamily="Helvetica")
        plt.yticks(fontsize=14, fontfamily="Helvetica")
        # Add legend only if there are labeled elements
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend(fontsize=14, prop={'family': 'Helvetica'})
        plt.grid(True)
        plt.minorticks_on()
        plt.title(title, fontsize=18, fontfamily="Helvetica")

    def plot_dihedral_potentials(self):
        """Plot dihedral potentials and their probability distributions."""
        if not self._full_data:
            self._prepare_full_data()
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        for key, data in self._full_data.items():
            plt.plot(data['data'][:, 0],
                     data['data'][:, 1],
                     f"{data['color']}o",
                     label=data['label'])
            plt.plot(data['x_values'], data['fitf'](data['x_values']),
                     f"{data['color']}--")
        self.format_subplot("Dihedral Angle [Deg.]",
                            "Dihedral Potential (kJ/mol)",
                            "Dihedral Potentials")
        plt.subplot(1, 3, 2)
        for key, data in self._full_data.items():
            plt.plot(data['x_values'],
                     data['prob_vals'],
                     f"{data['color']}-",
                     label=data['label'])
        self.format_subplot("Angle [deg.]", "Probability",
                            "Probability Distributions")
        plt.subplot(1, 3, 3)
        for key, data in self._full_data.items():
            plt.plot(data['cum_dist'] / data['cum_dist'][-1],
                     data['x_values'],
                     f"{data['color']}-",
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
        print("--- Persistence Length Calculation Report ---")
        print(f"Temperature: {self.temperature} K")
        print(f"Max Eigenvalue (lambda_max): {lam:.12f}")
        print(f"Persistence Length (in repeat units): {lp:.6f}")
        print("-------------------------------------------")

    def generate_chain(self, n_repeat_units):
        """Generate a polymer chain with n_repeat_units."""
        l_array = np.tile(self.bond_lengths, n_repeat_units)
        all_l = np.vstack((l_array, np.zeros((2, l_array.shape[0])))).T
        all_angle = np.tile(self.bond_angles_rad, n_repeat_units)
        angles = np.cumsum(all_angle[1:])
        vectors = all_l[1:]
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        rotated_x = vectors[:, 0] * cos_angles - vectors[:, 1] * sin_angles
        rotated_y = vectors[:, 0] * sin_angles + vectors[:, 1] * cos_angles
        rotated_z = vectors[:, 2]
        segments = np.column_stack((rotated_x, rotated_y, rotated_z))
        return np.cumsum(np.vstack((np.array([[0, 0, 0],
                                              [self.bond_lengths[0], 0,
                                               0]]), segments)),
                         axis=0)

    def pre_generate_angles(self, n_samples, flat_rotation):
        """Pre-generate all dihedral angles for Monte Carlo sampling."""
        self._prepare_full_data()
        length = len(self.bond_lengths)
        angles_per_position = np.zeros((n_samples, len(flat_rotation)))

        for rot_type, data_type in self._full_data.items():
            mask = flat_rotation == rot_type
            if np.any(mask):
                inv_cdf = data_type['inv_cdf']
                rand_vals = np.random.rand(n_samples, mask.sum())
                angles_per_position[:, mask] = inv_cdf(rand_vals)

        return angles_per_position

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
        length = len(self.bond_lengths)

        # Prepare rotation mapping for the chain
        flat_rotation = np.concatenate([
            [0], self.rotation_types[np.arange(len(ch) - 1) % length]
        ])[:-1].astype(np.int64)

        # Pre-generate all angles
        all_angles = self.pre_generate_angles(n_samples, flat_rotation)

        print(f"Calculating {n_samples} samples...")
        print(f"Using {psutil.cpu_count(logical=False)} CPU cores")

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

    def plot_correlation_function(self,
                                  n_repeat_units=20,
                                  n_samples=150000,
                                  start_idx=1,
                                  end_idx=10):
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
        """
        try:
            if chain_rotation is None:
                print(
                    "Error: chain_rotation module not available. Plot cannot be generated."
                )
                return

            # Generate the base chain
            ch = self.generate_chain(n_repeat_units)
            length = len(self.bond_lengths)

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
            plt.figure(figsize=(10, 6))
            plt.plot(repeat_units[start_idx:end_idx],
                     np.log(corr2[start_idx:end_idx]),
                     'bo',
                     markersize=8,
                     label='Log Correlation')
            plt.plot(repeat_units[start_idx:end_idx],
                     np.polynomial.polynomial.polyval(
                         repeat_units[start_idx:end_idx], p),
                     'b--',
                     linewidth=2,
                     alpha=0.7,
                     label=f'Np = {persistence_length:.5f}')
            plt.xlabel("Repeat Units", fontsize=16, fontfamily="Helvetica")
            plt.ylabel(r'Ln[$<V_0 \cdot V_n>$]',
                       fontsize=16,
                       fontfamily="Helvetica")
            plt.xticks(fontsize=14, fontfamily="Helvetica")
            plt.yticks(fontsize=14, fontfamily="Helvetica")
            plt.grid(True, alpha=0.3)
            # Add legend only if there are labeled elements
            if plt.gca().get_legend_handles_labels()[0]:
                plt.legend(fontsize=14, prop={'family': 'Helvetica'})
            plt.title("Log of Correlation Function",
                      fontsize=18,
                      fontfamily="Helvetica")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error in plot_correlation_function: {str(e)}")
            return


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


# ======================================================================
#                            USAGE EXAMPLE
# ======================================================================
# if __name__ == '__main__':
#     l = [4.289, 1.375, 4.289, 1.459, 2.510, 1.441, 2.510, 1.459]
#     Angle = np.array([-3.8, 38.1, -38.1, 3.8, 15.3,13.3,-13.3,-15.3 ])
#     rotation = np.array([0, 0, 0, 1, 0, 2, 0, 1])
#     labels = {
#         1: {
#             'loc':
#             r'E:\huangjy\script\persistence_length\adma.201702115\IID-FT.txt',
#             'color': 'b'
#         },
#         2: {
#             'loc':
#             r'E:\huangjy\script\persistence_length\adma.201702115\FT-FT.txt',
#             'color': 'm'
#         },

#     }

#     ris = np.array([0, 3, 0, 0, 0, 0, 0, 0])
#     ris_label = {
#         3: {'loc': r'E:\huangjy\script\persistence_length\adma.201702115\IID-RIS.txt', 'color': 'c'},
#         # 2: {'label': 'T-RIS', 'color': 'm'},
#         }

#     a = PolymerPersistence(l, Angle, rotation_types=rotation, rotation_labels=labels, ris_types=ris, ris_labels=ris_label)
#     a.report()
# if __name__ == '__main__':
#     # --- Define the molecular chain structure ---
#     # T-bond-DPP-bond-T-bond-T-bond-E-bond-T-bond
#     l = [
#         2.533, 1.432, 3.533, 1.432, 2.533, 1.432, 2.533, 1.433, 1.363, 1.433,
#         2.533, 1.432
#     ]
#     Angle = [
#         -14.92, -10.83, 30.79, -30.79, 10.83, 14.92, -14.91, -13.29, -53.16,
#         53.16, 13.29, 14.91
#     ]

#     # Mapping to data files: 1 -> T-DPP.txt, 2 -> T-T.txt, 3 -> T-E.txt, 0 -> Fixed
#     rotation = [0, 1, 0, 1, 0, 2, 0, 3, 0, 3, 0, 2]
#     data = {
#         1: {
#             'loc':
#             r'E:\huangjy\script\persistence_length\template_transfer_matrix\T-DPP.txt',
#             'color': 'b'
#         },
#         2: {
#             'loc':
#             r'E:\huangjy\script\persistence_length\template_transfer_matrix\T-T.txt',
#             'color': 'm'
#         },
#         3: {
#             'loc':
#             r'E:\huangjy\script\persistence_length\template_transfer_matrix\T-E.txt',
#             'color': 'c'
#         },
#     }
#     # Set temperature
#     temperature = 300  # K

#     # --- Create an instance and run the calculation ---
#     # 1. Initialize the model with your polymer's data
#     polymer_model = PolymerPersistence(
#         bond_lengths=l,
#         bond_angles_deg=Angle,
#         temperature=temperature,
#         rotation_types=rotation,
#         rotation_labels=data,
#     )

#     # 2. Access the results. The calculation is run automatically the first time a result is requested.
#     lp_repeats = polymer_model.persistence_length_repeats

#     # 3. Print a formatted report
#     polymer_model.report()
#     polymer_model.compute_mean_square_end_to_end(N=20)
#     polymer_model.plot_dihedral_potentials()

#     temperatures = np.linspace(300, 700, 9)
#     N_p = [
#         PolymerPersistence(
#             bond_lengths=l,
#             bond_angles_deg=Angle,
#             rotation_types=rotation,
#             temperature=t,
#             rotation_labels=data,
#         ).persistence_length_repeats for t in temperatures
#     ]

#     all_np = [
#         f"{num:.5f} (Temperature = {t:.1f} K)"
#         for num, t in zip(np.array(N_p), temperatures)
#     ]
#     print("Persistence Length in Repeats:", )
#     print("\n".join(all_np))
