from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.constants as sc
from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from . import tool
try:
    from . import chain_rotation_fk as chain_fk
except ImportError:
    print("Warning: chain_rotation_fk module not found.")
    chain_fk = None


class PolymerPersistenceFK:
    """
    Optimized version using Forward Kinematics.
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
        """Initialize the optimized polymer persistence model."""
        if bond_lengths is None or bond_angles_deg is None:
            raise ValueError("Bond lengths and angles must be provided.")
        self.bond_lengths = np.array(bond_lengths)
        self.bond_angles_rad = np.deg2rad(np.array(bond_angles_deg))
        self.rotation_types = np.array(rotation_types)
        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

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
        if self.ris_types is not None:
            self.ris_types = np.array(self.ris_types)
            self.ris_data = {}
            for ris_id, info in self.ris_labels.items():
                try:
                    if 'data' in info:
                        risdata = np.asarray(info['data'])
                        angles, energies = risdata[:, 0], risdata[:, 1]
                    elif 'loc' in info:
                        angles, energies = tool.read_ris_data(Path(
                            info['loc']))
                    self.ris_data[ris_id] = (angles, energies)
                except FileNotFoundError:
                    print(
                        f"Warning: RIS data file not found. Skipping RIS type {ris_id}."
                    )
                    continue
        self._full_data = {}
        self.fitting_method = fitting_method
        self.param_n = param_n

    @staticmethod
    def _update_dihedral(data):
        """Update dihedral data to cover 0-360 degrees."""
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
        """Read and process dihedral angle data from a file."""
        delimiter = ',' if file_name.suffix == '.csv' else None
        data = np.loadtxt(file_name, delimiter=delimiter)
        data = np.reshape(data, (-1, 2))
        return self._update_dihedral(data)

    def _prepare_full_data(self):
        """Set up interpolation functions from data files."""
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

    def pre_generate_angles(self, n_samples, n_total_bonds):
        """
        Generate all dihedral angles at once.
        Returns array of shape (n_samples, n_total_bonds) in RADIANS.
        """
        self._prepare_full_data()

        n_bonds_per_unit = len(self.rotation_types)
        flat_rotation = np.tile(
            self.rotation_types,
            n_samples * n_total_bonds // n_bonds_per_unit + 1)[:n_total_bonds]

        rng = np.random.default_rng()
        rand_vals = rng.random((n_samples, n_total_bonds))
        angles_deg = np.zeros((n_samples, n_total_bonds))

        for rot_type, data_type in self._full_data.items():
            mask = flat_rotation == rot_type
            if np.any(mask):
                inv_cdf = data_type['inv_cdf']
                angles_deg[:, mask] = inv_cdf(rand_vals[:, mask])

        if self.ris_types is not None:
            flat_ris = np.tile(
                self.ris_types, n_samples * n_total_bonds // n_bonds_per_unit +
                1)[:n_total_bonds]
            for ris_id, (ang_deg, energies) in self.ris_data.items():
                mask = (flat_ris == ris_id)
                if not np.any(mask):
                    continue

                boltz = np.exp(-energies / self.kTval)
                prob = boltz / boltz.sum()

                sampled = rng.choice(ang_deg,
                                     size=(n_samples, np.sum(mask)),
                                     p=prob)
                angles_deg[:, mask] = sampled

        return np.deg2rad(angles_deg)

    def calculate_correlation_length_mc(self,
                                        n_repeat_units=20,
                                        n_samples=150000,
                                        use_cython=True):
        """
        Optimized Monte Carlo calculation using forward kinematics.
        Returns:
        --------
        float: Correlation length
        """
        if chain_fk is None:
            print("Warning: chain_rotation_fk Cython module not available.")
            use_cython = False

        n_bonds_per_unit = len(self.bond_angles_rad)
        n_total_bonds = n_bonds_per_unit * n_repeat_units

        # Pre-generate all dihedral angles (in radians)
        print(f"Generating {n_samples} angle sets...")
        all_angles = self.pre_generate_angles(n_samples, n_total_bonds)
        if use_cython:
            print(
                f"Calculating correlations using {psutil.cpu_count(logical=False)} CPU cores..."
            )

            # Batch processing
            batch_size = 1000
            n_batches = n_samples // batch_size
            n_jobs = psutil.cpu_count(logical=False)

            corr_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(chain_fk.batch_correlation_fk)
                (np.ascontiguousarray(self.bond_lengths, dtype=np.float64),
                 np.ascontiguousarray(self.bond_angles_rad, dtype=np.float64),
                 np.ascontiguousarray(all_angles[i * batch_size:(i + 1) *
                                                 batch_size],
                                      dtype=np.float64), n_repeat_units)
                for i in range(n_batches))

            corr_results = np.vstack(corr_results)

            # Average correlation function
            corr_mean = np.mean(corr_results, axis=0)
        else:
            print("Calculating correlations using Python...")
            all_chains = self.build_all_chains_no_cython(
                n_samples, n_repeat_units, all_angles)
            corr_mean = np.mean(self.cosVals(all_chains, n_bonds_per_unit),
                                axis=0)
        repeat_units = np.arange(1, len(corr_mean) + 1)

        # Fit exponential decay
        start_idx = 0
        end_idx = min(10, len(corr_mean))

        # Log-linear fit
        valid_mask = corr_mean[start_idx:end_idx] > 0
        if not np.any(valid_mask):
            print("Warning: No positive correlation values for fitting.")
            return np.inf

        x_fit = repeat_units[start_idx:end_idx][valid_mask]
        y_fit = np.log(corr_mean[start_idx:end_idx][valid_mask])

        p = np.polynomial.polynomial.polyfit(x_fit, y_fit, 1)
        corr_length = -1 / p[1] if p[1] != 0 else np.inf

        print(f"\nOptimized Monte Carlo Result:")
        print(f"Slope: {p[1]:.6f}")
        print(f"Correlation Length: {corr_length:.6f}")

        return corr_length

    @staticmethod
    def cosVals(pts, length):
        k_values = np.arange(length, pts.shape[1], length)
        vectors = pts[:, k_values, :] - pts[:, k_values - length, :]
        v_ref = vectors[:, 0:1, :]
        dots = np.sum(vectors * v_ref, axis=2)
        norms = np.linalg.norm(vectors, axis=2) * np.linalg.norm(v_ref, axis=2)
        return np.clip(dots / norms, -1, 1)

    @staticmethod
    def mean_square_end_to_end(pts, length):
        k_values = np.arange(0, pts.shape[1], length)
        vectors = pts[:, k_values, :] - pts[:, 0:1, :]
        square_norms = np.sum(vectors**2, axis=2)
        return np.mean(square_norms, axis=0)

    def plot_correlation_function(self,
                                  n_repeat_units=20,
                                  n_samples=150000,
                                  start_idx=0,
                                  end_idx=10,
                                  return_data=False,
                                  use_cython=True):
        """Plot correlation function using FK method."""
        if chain_fk is None:
            print("Warning: chain_rotation_fk Cython module not available.")
            use_cython = False

        n_bonds_per_unit = len(self.bond_angles_rad)
        n_total_bonds = n_bonds_per_unit * n_repeat_units

        all_angles = self.pre_generate_angles(n_samples, n_total_bonds)
        if use_cython:
            batch_size = 1000
            n_batches = n_samples // batch_size
            n_jobs = psutil.cpu_count(logical=False)

            corr_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(chain_fk.batch_correlation_fk)
                (np.ascontiguousarray(self.bond_lengths, dtype=np.float64),
                 np.ascontiguousarray(self.bond_angles_rad, dtype=np.float64),
                 np.ascontiguousarray(all_angles[i * batch_size:(i + 1) *
                                                 batch_size],
                                      dtype=np.float64), n_repeat_units)
                for i in range(n_batches))

            corr_results = np.vstack(corr_results)
            corr_mean = np.mean(corr_results, axis=0)
        else:
            all_chains = self.build_all_chains_no_cython(
                n_samples, n_repeat_units, all_angles)
            corr_mean = np.mean(self.cosVals(all_chains, n_bonds_per_unit),
                                axis=0)
        self.plot_correlation(corr_mean, start_idx, end_idx)
        if return_data:
            return corr_mean

    def plot_correlation(self, corr_mean, start_idx, end_idx):
        repeat_units = np.arange(1, len(corr_mean) + 1)

        end_idx = min(end_idx, len(corr_mean))
        valid_mask = corr_mean[start_idx:end_idx] > 0

        if not np.any(valid_mask):
            print("Error: No valid correlation values for fitting.")
            return

        x_fit = repeat_units[start_idx:end_idx][valid_mask]
        y_fit = np.log(corr_mean[start_idx:end_idx][valid_mask])
        p = np.polynomial.polynomial.polyfit(x_fit, y_fit, 1)
        corr_length = -1 / p[1] if p[1] != 0 else np.inf

        plt.figure(figsize=(6, 5))
        plt.plot(x_fit, y_fit, 'bo', label='Log Correlation')
        plt.plot(x_fit,
                 np.polynomial.polynomial.polyval(x_fit, p),
                 'b--',
                 linewidth=2,
                 alpha=0.7,
                 label=f'zeta = {corr_length:.6f}')
        tool.format_subplot("Repeat Units", r'Ln[$<V_0 \cdot V_n>$]',
                            "Log of Correlation Function")
        plt.show()

    def build_all_chains_no_cython(self, n_samples, n_repeat_units,
                                   all_dihedrals):
        """
        Generate full chain positions
        """
        length = self.bond_lengths.shape[0]
        n_bonds = n_repeat_units * length
        l_array = np.tile(self.bond_lengths,
                          n_repeat_units)  # Shape: (n_bonds,)
        theta = np.tile(np.tile(self.bond_angles_rad, n_repeat_units),
                        (n_samples, 1))
        c_phi, s_phi = np.cos(all_dihedrals), np.sin(all_dihedrals)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        R_steps = np.zeros((n_samples, n_bonds, 3, 3))
        R_steps[:, :, 0, 0] = c_theta
        R_steps[:, :, 1, 0] = s_theta
        R_steps[:, :, 0, 1] = -s_theta * c_phi
        R_steps[:, :, 1, 1] = c_theta * c_phi
        R_steps[:, :, 2, 1] = s_phi
        R_steps[:, :, 0, 2] = s_theta * s_phi
        R_steps[:, :, 1, 2] = -c_theta * s_phi
        R_steps[:, :, 2, 2] = c_phi
        R_local = np.eye(3)
        R_current = np.tile(R_local, (n_samples, 1, 1))
        vectors = np.zeros((n_samples, n_bonds, 3))
        for i in range(n_bonds):
            R_current = R_current @ R_steps[:, i]
            vectors[:, i] = R_current[:, :, 0] * l_array[i, None]
        coords = np.zeros((n_samples, n_bonds + 1, 3))
        coords[:, 1:] = np.cumsum(vectors, axis=1)

        return coords

    def plot_chain(self,
                   n_repeat_units,
                   colormap='jet',
                   rotate=False,
                   use_cython=True):
        """
        Plot the polymer chain using the FK method.
        """
        if chain_fk is None:
            print("Warning: chain_rotation_fk Cython module not available.")
            use_cython = False

        n_bonds_per_unit = len(self.bond_angles_rad)
        n_total_bonds = n_bonds_per_unit * n_repeat_units
        if rotate:
            all_angles = self.pre_generate_angles(1, n_total_bonds)[0]
        else:
            all_angles = np.zeros(n_total_bonds)
        if use_cython:
            chain = chain_fk.build_full_chain_fk(
                np.ascontiguousarray(self.bond_lengths, dtype=np.float64),
                np.ascontiguousarray(self.bond_angles_rad, dtype=np.float64),
                np.ascontiguousarray(all_angles, dtype=np.float64),
                n_repeat_units)
        else:
            chain = self.build_all_chains_no_cython(1, n_repeat_units,
                                                    all_angles)[0]

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(chain[:, 0], chain[:, 1], chain[:, 2], s=20)
        colors = plt.get_cmap(colormap)
        for i in range(len(chain) - 1):
            ax.plot(
                [chain[i][0], chain[i + 1][0]], [chain[i][1], chain[i + 1][1]],
                [chain[i][2], chain[i + 1][2]],
                color=colors(i % n_bonds_per_unit * (256 // n_bonds_per_unit)),
                linewidth=2)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    def calc_mean_square_end_to_end_distance(self,
                                             n_repeat_units=20,
                                             n_samples=150000,
                                             return_data=False,
                                             plot=False,
                                             use_cython=True):
        """Plot correlation function using FK method."""
        if chain_fk is None:
            print("Warning: chain_rotation_fk Cython module not available.")
            use_cython = False

        n_bonds_per_unit = len(self.bond_angles_rad)
        n_total_bonds = n_bonds_per_unit * n_repeat_units

        all_angles = self.pre_generate_angles(n_samples, n_total_bonds)
        if use_cython:
            batch_size = 1000
            n_batches = n_samples // batch_size
            n_jobs = psutil.cpu_count(logical=False)

            r2_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(chain_fk.batch_end_to_end)
                (np.ascontiguousarray(self.bond_lengths, dtype=np.float64),
                 np.ascontiguousarray(self.bond_angles_rad, dtype=np.float64),
                 np.ascontiguousarray(all_angles[i * batch_size:(i + 1) *
                                                 batch_size],
                                      dtype=np.float64), n_repeat_units)
                for i in range(n_batches))

            r2_results = np.vstack(r2_results)
            r2 = np.mean(r2_results, axis=0)
        else:
            all_chains = self.build_all_chains_no_cython(
                n_samples, n_repeat_units, all_angles)
            r2 = self.mean_square_end_to_end(all_chains, n_bonds_per_unit)
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(np.arange(n_repeat_units + 1), r2, 'bo-')
            tool.format_subplot("Number of Repeat Units (N)",
                                "Mean Square End-to-End Distance (Å²)",
                                "Forward Kinetics Simulation of <R²>")
            plt.show()
        if return_data:
            return r2

    def calc_end_to_end_distribution(self,
                                     n_repeat_units=20,
                                     n_samples=150000,
                                     bins=100,
                                     use_r2=True,
                                     density=True,
                                     plot=True,
                                     use_cython=True,
                                     return_data=False):
        """
        Calculate the distribution of end-to-end distance (R or R^2)
        for the FULL chain length.

        Parameters
        ----------
        use_r2 : bool
            If True, compute distribution of R^2.
            If False, compute distribution of R = sqrt(R^2).
        density : bool
            If True, normalize histogram to PDF.
        """

        if chain_fk is None:
            print("Warning: chain_rotation_fk Cython module not available.")
            use_cython = False

        n_bonds_per_unit = len(self.bond_angles_rad)
        n_total_bonds = n_bonds_per_unit * n_repeat_units
        all_angles = self.pre_generate_angles(n_samples, n_total_bonds)
        if use_cython:
            batch_size = 1000
            n_batches = n_samples // batch_size
            n_jobs = psutil.cpu_count(logical=False)

            r2_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(chain_fk.batch_end_to_end)
                (np.ascontiguousarray(self.bond_lengths, dtype=np.float64),
                 np.ascontiguousarray(self.bond_angles_rad, dtype=np.float64),
                 np.ascontiguousarray(all_angles[i * batch_size:(i + 1) *
                                                 batch_size],
                                      dtype=np.float64), n_repeat_units)
                for i in range(n_batches))

            r2_results = np.vstack(r2_results)
            # R^2 for full chain
            r2_full = r2_results[:, -1]
        else:
            all_chains = self.build_all_chains_no_cython(
                n_samples, n_repeat_units, all_angles)
            vec = all_chains[:, -1, :] - all_chains[:, 0, :]
            r2_full = np.sum(vec**2, axis=1)
        # R or R^2
        values = r2_full if use_r2 else np.sqrt(r2_full)
        hist, bin_edges = np.histogram(values, bins=bins, density=density)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(bin_centers, hist, 'b-', lw=2)
            xlabel = r"$R^2$ ($\mathrm{\AA}^2$)" if use_r2 else r"$R$ ($\mathrm{\AA}$)"
            ylabel = "Probability Density" if density else "Counts"
            tool.format_subplot(xlabel, ylabel,
                                "End-to-End Distance Distribution")
            plt.show()

        if return_data:
            return bin_centers, hist
