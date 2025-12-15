from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from numpy.linalg import eigvals
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.cm as cm

# Import the Cython module (after compilation)
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
                 ris_types=None,
                 ris_labels=None,
                 fittting_method='interpolation',
                 cosine_deg=5):
        """
        Initializes the PolymerPersistence model.

        Args:
            bond_lengths (list or np.ndarray): The lengths of the bonds in the repeat unit.
            bond_angles_file (Path): The path to the file containing the dihedraal-dependent bond angles.
            temperature (int, optional): The temperature in Kelvin. Defaults to 300.
            rotaion_types (list or np.ndarray, optional): An array of integers mapping each bond to a
                                                 specific rotational potential profile. A value of 0
                                                 indicates a fixed bond with no rotation.
            rotation_labels (dict, optional): A dictionary mapping rotation_types to data files.
            ris_types (list or np.ndarray, optional): An array of integers mapping each bond to ris model.
            ris_labels (dict, optional): A dictionary mapping ris_types to data files.
            fittting_method (str, optional): The method used for fitting the data. 'interpolation' or 'cosine'.
            cosine_deg (int, optional): The number of degrees to use for the cosine fitting method.
        """
        self.bond_lengths = np.asarray(
            bond_lengths) if bond_lengths is not None else None
        self.bond_angle_file = bond_angles_file
        self.rotation_types = np.array(rotation_types)
        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

        # Default labels mapping rotation_types to data files
        self.rotation_labels = rotation_labels
        for rot_id in self.rotation_labels:
            rot = self.rotation_labels[rot_id]
            if 'data' in rot and 'label' not in rot:
                self.rotation_labels[rot_id]['label'] = f"dihedral {rot_id}"
            if 'loc' in rot and 'label' not in rot:
                file_path = self.rotation_labels[rot_id]['loc']
                self.rotation_labels[rot_id]['label'] = Path(file_path).stem
        self.ris_labels = ris_labels
        self.ris_types = ris_types
        self.deflection_types = np.array(rotation_types).copy()
        mask = self.rotation_types == 0
        self.deflection_types[mask] = np.roll(self.rotation_types, 1)[mask]
        # --- Internal cache for lazy evaluation ---
        self._Mmat = None
        self._lambda_max = None
        self._lp_in_repeats = None
        self._computational_data = {}
        self.fitting_method = fittting_method
        self.cosine_deg = cosine_deg
        self._avg_angles = None

    @staticmethod
    def _read_ris_data(file_name: Path):
        delimiter = ',' if file_name.suffix == '.csv' else None
        data = np.loadtxt(file_name, delimiter=delimiter)
        data = np.reshape(data, (-1, 2))
        data = np.unique(data, axis=0)
        return data[:, 0], data[:, 1]

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

    def _fit_deflection(self, file_name: Path):
        """Fits the deflection angle data."""
        data = self._update_angle(self._read_data(file_name))
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
            try:
                if 'data' in info:
                    data = self._update_angle(info['data'])
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
                else:  # cosine
                    p = np.polynomial.polynomial.polyfit(
                        np.cos(np.deg2rad(x)), y, self.cosine_deg)
                    fitf = (lambda p_val: lambda z: np.polynomial.polynomial.
                            polyval(np.cos(np.deg2rad(z)), p_val))(p)
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
        # list deflection functions
        fitf_deflection = self._fit_deflection(Path(self.bond_angle_file))
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
        # deflection using rotation types
        A_list = []
        avg_angles = []
        integral_cache = {}
        ris_cache = {}

        for i in range(M):
            rot_id = int(self.rotation_types[i])
            ris_id = int(
                self.ris_types[i]) if self.ris_types is not None else 0
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
                c, s, angle_avg = self._compute_deflection_integrals(fitf_angle, fitf)
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
            R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1]])

            A_list.append(R_x @ R_z)
            avg_angles.append(angle_avg)

        # Multiply all transformation matrices for the repeat unit
        Mmat = np.eye(3)
        for A in A_list:
            Mmat = A @ Mmat
        self._Mmat = Mmat
        self._avg_angles = avg_angles

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

    @property
    def average_angles(self):
        if self._avg_angles is None:
            self._calculate_Mmat()
        return self._avg_angles

    def compute_mean_square_end_to_end(self, N):
        """Computes the mean square end-to-end distance for a given number of repeat units."""
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
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

    def compute_mean_square_Rg(self, N):
        """Computes the mean square radius of gyration for a given number of repeat units."""
        if self.bond_lengths is None:
            raise RuntimeError("Bond lengths not set.")
        if self._lp_in_repeats is None:
            self.run_calculation()
        L_repeat = np.sum(self.bond_lengths)
        if self._lp_in_repeats == np.inf:
            raise RuntimeError(
                "Infinite persistence length not supported for Rg calculation."
            )
        else:
            L_square = L_repeat**2
            Np = self._lp_in_repeats
            rg2 = L_square * (Np * N / 3 - Np**2 + 2 * Np**3 / N -
                              2 * Np**4 / N**2 * (1 - np.exp(-N / Np)))
            return rg2

    def plot_end_to_end_distance(self, N=20):
        """Plots the mean square end-to-end distance as a function of repeat units from 1 to N.

        Args:
            N (int): Maximum number of repeat units to plot
        """
        # Generate array of N values
        N_values = np.arange(1, N + 1)

        # Calculate mean square end-to-end distance for each N
        msd_values = [self.compute_mean_square_end_to_end(n) for n in N_values]

        # Create the plot
        plt.figure(figsize=(6, 5))
        plt.plot(N_values, msd_values, linewidth=2, color='blue', marker='o')

        # Format the plot
        self.format_subplot("Number of Repeat Units (N)",
                            "Mean Square End-to-End Distance (Å²)",
                            "<R²> vs. Number of Repeat Units")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_mean_square_radius_of_gyration(self, N=20):
        """Plots the mean square radius of gyration as a function of repeat units from 1 to N.

        Args:
            N (int): Maximum number of repeat units to plot
        """
        N_values = np.arange(1, N + 1)
        msd_values = [self.compute_mean_square_Rg(n) for n in N_values]
        plt.figure(figsize=(6, 5))
        plt.plot(N_values, msd_values, linewidth=2, color='blue', marker='o')

        self.format_subplot("Number of Repeat Units (N)", "<Rg²> (Å²)",
                            "<Rg²> vs. Number of Repeat Units")
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
        plt.grid(True, alpha=0.3)
        plt.minorticks_on()
        plt.title(title, fontsize=18, fontfamily="Helvetica")

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

    def report_average_angles(self, round=2):
        avg = self.average_angles
        print("---- Average Deflection Angles ----")
        print(", ".join([f"{x:.{round}f}" for x in avg]))
        print("-----------------------------------")

    def plot_deflection_angles(self):
        original = self._update_angle(
            self._read_data(Path(self.bond_angle_file)))
        fit_angle = self._fit_deflection(Path(self.bond_angle_file))
        colors = cm.get_cmap('tab20')
        x = np.linspace(0, 360, 721)
        for i in range(len(fit_angle)):
            fitf = fit_angle[i]
            color = colors((i % 20) / 20.0)
            plt.plot(original[:, 0],
                     original[:, i + 1],
                     linestyle='none',
                     c=color)
            plt.plot(x, fitf(x), label=f"{i+1}", c=color)
        self.format_subplot("Dihedral Angle (°)", "Deflection Angle (°)",
                            "Variable Deflection Angle")
        plt.show()

    def temperature_scan(self, T_list, plot=False):
        """
        T_list: iterable of temperatures (K)
        Returns: dict with keys 'T', 'lp', 'Mmat'
        """
        Ts = np.asarray(T_list, dtype=np.float64)
        results = {'T': Ts, 'lp': [], 'Mmat': []}

        # store original kTval
        kT_orig = self.kTval
        for T in Ts:
            self.temperature = float(T)
            self.kTval = sc.R * self.temperature / 1000.0  # kJ/mol
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
        self.temperature = float(self.temperature)
        self.kTval = kT_orig
        self._calculate_Mmat()
        return results
