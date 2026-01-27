from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.constants as sc
from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from . import tool
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
try:
    from . import chain_rotation_fk as chain_fk
except ImportError:
    print("Warning: chain_rotation_fk module not found.")
    chain_fk = None


class PolymerPersistenceUnit:
    """
    Optimized version using Forward Kinematics for copolymer chains.
    """

    def __init__(self,
                 unit_dict,
                 bond_dict,
                 angle_dict,
                 rotation_dict,
                 connection,
                 probability,
                 temperature=300.0,
                 rotation_labels=None,
                 fitting_method='interpolation',
                 param_n=5):
        """
        Initialize the polymer persistence model for copolymers.
        
        Parameters:
        -----------
        unit_dict : dict
            Dictionary defining monomer units with 'bond', 'angle', 'rotation' keys
        bond_dict : dict
            Dictionary of bond lengths between units
        angle_dict : dict
            Dictionary of bond angles between units
        rotation_dict : dict
            Dictionary of rotation types between units
        connection : list of lists
            Connection patterns for each position
        probability : list of lists
            Probability distributions for each position
        temperature : float
            Temperature in Kelvin
        rotation_labels : dict
            Labels and data for different rotation types
        fitting_method : str
            Method for fitting dihedral potentials
        param_n : int
            Number of parameters for fitting
        """

        self.unit_dict = unit_dict
        self.bond_dict = bond_dict
        self.angle_dict = angle_dict
        self.rotation_dict = rotation_dict
        self.connection = connection
        self.probability = probability

        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

        self.rotation_labels = rotation_labels if rotation_labels is not None else {}
        for rot_id, info in self.rotation_labels.items():
            if 'type' not in info:
                self.rotation_labels[rot_id]['type'] = 'continuous'
            if ('data' in info or 'fitf' in info) and 'label' not in info:
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

        self._full_data = {}
        self.fitting_method = fitting_method
        self.param_n = param_n
        self._build_rotation_encoding()
        # Precompute pair cache
        self._build_pair_cache()

    def _build_rotation_encoding(self):
        """Build mapping from rotation type strings to integers"""
        # Collect all unique rotation types
        all_rot_types = set()
        for c1 in self.unit_dict.keys():
            for c2 in self.unit_dict.keys():
                key = c1 + c2
                if key in self.rotation_dict:
                    all_rot_types.update([self.rotation_dict[key]])
            all_rot_types.update(self.unit_dict[c1]['rotation'])

        # Create bidirectional mapping
        self.rot_to_int = {
            rot: i
            for i, rot in enumerate(sorted(all_rot_types))
        }
        self.int_to_rot = {i: rot for rot, i in self.rot_to_int.items()}
        print(f"Rotation encoding: {len(self.rot_to_int)} unique types")

    def _build_pair_cache(self):
        """Precompute all possible unit pairs with integer-encoded rotations"""
        all_chars = list(self.unit_dict.keys())
        self.pair_cache = {}

        for c1 in all_chars:
            for c2 in all_chars:
                key = c1 + c2
                if key not in self.bond_dict:
                    continue

                b_list = self.unit_dict[c1]['bond'] + [self.bond_dict[key]]
                a_list = self.unit_dict[c1]['angle'] + self.angle_dict[key]
                r_list = self.unit_dict[c1]['rotation'] + [
                    self.rotation_dict[key]
                ]

                # Encode rotations as integers
                r_int = np.array([self.rot_to_int[r] for r in r_list],
                                 dtype=np.int32)

                self.pair_cache[key] = {
                    'bonds': np.array(b_list),
                    'angles': np.array(a_list),
                    'rotations': r_int,
                    'length': len(b_list)
                }

    def generate_copolymer_chains(self, n_repeats, n_samples):
        """
        Generate structural information for random copolymer chains
        
        Returns list of dicts with integer-encoded rotations and padded arrays
        """
        n_pos = len(self.connection)
        conn_arr = [np.array(c) for c in self.connection]
        chains_matrix = np.empty((n_samples, n_repeats * n_pos), dtype='U1')
        rng = np.random.default_rng()

        # Generate character matrix
        for i in range(n_pos):
            target_cols = np.arange(i, n_repeats * n_pos, n_pos)
            choices = rng.choice(conn_arr[i],
                                 size=(n_samples, n_repeats),
                                 p=self.probability[i])
            chains_matrix[:, target_cols] = choices

        # NEW: Pre-calculate maximum possible bonds across all pair types
        max_bonds_per_pair = max(data['length']
                                 for data in self.pair_cache.values())
        # Worst case: maximum bonds per pair × total number of pairs
        total_max_bonds = max_bonds_per_pair * n_repeats * n_pos

        # Batch generate structural information with padding
        final_results = []
        for row in chains_matrix:
            # Pre-allocate padded arrays
            bonds_pad = np.zeros(total_max_bonds)
            angles_pad = np.zeros(total_max_bonds)
            rotations_pad = np.zeros(total_max_bonds, dtype=np.int32)
            mask = np.zeros(total_max_bonds, dtype=bool)
            unit_lengths = []

            actual_idx = 0
            for r in range(n_repeats):
                unit_start = actual_idx

                for p in range(n_pos):
                    idx = r * n_pos + p
                    curr = row[idx]
                    nxt = row[(idx + 1) % len(row)]
                    data = self.pair_cache[curr + nxt]

                    n_bonds = data['length']
                    bonds_pad[actual_idx:actual_idx + n_bonds] = data['bonds']
                    angles_pad[actual_idx:actual_idx +
                               n_bonds] = data['angles']
                    rotations_pad[actual_idx:actual_idx +
                                  n_bonds] = data['rotations']
                    mask[actual_idx:actual_idx + n_bonds] = True
                    actual_idx += n_bonds

                unit_lengths.append(actual_idx - unit_start)

            # Angle correction on valid portion
            valid_angles = angles_pad[mask]
            if len(valid_angles) > 0:
                valid_angles = np.concatenate([[valid_angles[-1]],
                                               valid_angles[:-1]])
                angles_pad[mask] = valid_angles

            final_results.append({
                'bonds': bonds_pad,
                'angles': np.deg2rad(angles_pad),
                'rotations': rotations_pad,  # Integer encoded
                'mask': mask,
                'unit_lengths': np.array(unit_lengths),
                'chain': ''.join(row),
                'actual_length': actual_idx
            })

        return final_results

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
            if info.get('type') == 'ris':
                continue
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
                    'rot_int':
                    self.rot_to_int[rot_id],  # NEW: Store integer encoding
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

    def pre_generate_angles_copolymer_batch(self, chains_info):
        """
        NEW: Batch generate dihedral angles for multiple chains at once
        
        Parameters:
        -----------
        chains_info : list of dict
            List of chain information dictionaries
            
        Returns:
        --------
        list of np.ndarray: List of dihedral angle arrays (in radians)
        """
        self._prepare_full_data()

        n_chains = len(chains_info)
        max_len = max(info['actual_length'] for info in chains_info)

        # Stack all rotation arrays
        all_rotations = np.zeros((n_chains, max_len), dtype=np.int32)
        all_masks = np.zeros((n_chains, max_len), dtype=bool)

        for i, info in enumerate(chains_info):
            n = info['actual_length']
            all_rotations[i, :n] = info['rotations'][:n]
            all_masks[i, :n] = info['mask'][:n]

        # Pre-allocate result
        angles_deg = np.zeros((n_chains, max_len))

        # Generate random values once
        rng = np.random.default_rng()
        rand_vals = rng.random((n_chains, max_len))

        # Process each rotation type in batch
        for rot_str, data_type in self._full_data.items():
            rot_int = data_type['rot_int']
            mask = (all_rotations == rot_int) & all_masks

            if np.any(mask):
                inv_cdf = data_type['inv_cdf']
                angles_deg[mask] = inv_cdf(rand_vals[mask])

        # Process RIS types in batch
        for rot_str, (ang_deg, energies) in self.ris_data.items():
            rot_int = self.rot_to_int[rot_str]
            mask = (all_rotations == rot_int) & all_masks

            if np.any(mask):
                boltz = np.exp(-energies / self.kTval)
                prob = boltz / boltz.sum()
                n_samples = np.sum(mask)
                sampled = rng.choice(ang_deg, size=n_samples, p=prob)
                angles_deg[mask] = sampled

        # Convert to list of arrays with actual lengths
        result = []
        for i, info in enumerate(chains_info):
            n = info['actual_length']
            result.append(np.deg2rad(angles_deg[i, :n]))

        return result

    def pre_generate_angles_copolymer(self, chain_info):
        """
        Generate dihedral angles for a single copolymer chain (legacy compatibility)
        """
        return self.pre_generate_angles_copolymer_batch([chain_info])[0]

    def build_chain_copolymer(self, chain_info, all_dihedrals):
        """
        Build coordinates for a single copolymer chain (only recording unit endpoints)
        
        Parameters:
        -----------
        chain_info : dict
            Dictionary containing 'bonds', 'angles', 'unit_lengths', 'mask'
        all_dihedrals : np.ndarray
            Array of dihedral angles (in radians)
            
        Returns:
        --------
        np.ndarray: Chain coordinates with shape (n_units+1, 3), containing only unit endpoints
        """
        n = chain_info['actual_length']
        mask = chain_info['mask'][:n]

        all_l = chain_info['bonds'][:n][mask[:n]]
        theta = chain_info['angles'][:n][mask[:n]]
        unit_lengths = chain_info['unit_lengths']
        n_bonds = len(all_l)

        # Calculate unit endpoint indices (including start point 0)
        end_idx = np.concatenate([[0], np.cumsum(unit_lengths)]).astype(int)

        c_phi, s_phi = np.cos(all_dihedrals), np.sin(all_dihedrals)
        c_theta, s_theta = np.cos(theta), np.sin(theta)

        # Build rotation matrices
        vectors = np.zeros((n_bonds, 3))
        R_steps = np.zeros((n_bonds, 3, 3))
        R_steps[:, 0, 0] = c_theta
        R_steps[:, 1, 0] = s_theta
        R_steps[:, 0, 1] = -s_theta * c_phi
        R_steps[:, 1, 1] = c_theta * c_phi
        R_steps[:, 2, 1] = s_phi
        R_steps[:, 0, 2] = s_theta * s_phi
        R_steps[:, 1, 2] = -c_theta * s_phi
        R_steps[:, 2, 2] = c_phi
        R_current = np.eye(3)
        for j in range(n_bonds):
            R_current = R_current @ R_steps[j]
            vectors[j] = R_current[:, 0] * all_l[j]

        # Calculate cumulative positions for all bonds
        coords_full = np.zeros((n_bonds + 1, 3))
        coords_full[1:] = np.cumsum(vectors, axis=0)

        # Return only unit endpoint positions
        return coords_full[end_idx]

    def calculate_correlation_length_mc(self,
                                        n_repeat_units=20,
                                        n_samples=150000,
                                        plot=True,
                                        return_data=False,
                                        use_cython=True):
        """
        Monte Carlo calculation of correlation length using forward kinematics.
        """
        if chain_fk is None and use_cython:
            print('No cython module found.')
            use_cython = False

        print(f"Generating {n_samples} copolymer chains...")
        chains_info = self.generate_copolymer_chains(n_repeat_units, n_samples)

        print("Batch generating dihedral angles...")
        all_dihedrals = self.pre_generate_angles_copolymer_batch(chains_info)

        n_jobs = psutil.cpu_count(logical=False)
        print(f"Building chain coordinates using {n_jobs} parallel jobs...")

        # Build chain coordinates in parallel
        def build_single_chain(chain_info, dihedrals):
            if use_cython:
                n = chain_info['actual_length']
                mask = chain_info['mask'][:n]
                return chain_fk.build_chain_copolymer_cy(
                    chain_info['bonds'][:n][mask[:n]],
                    chain_info['angles'][:n][mask[:n]],
                    chain_info['unit_lengths'], dihedrals)
            else:
                return self.build_chain_copolymer(chain_info, dihedrals)

        all_chains = Parallel(verbose=1, n_jobs=n_jobs)(
            delayed(build_single_chain)(chains_info[i], all_dihedrals[i])
            for i in range(len(chains_info)))

        all_chains = np.array(all_chains)

        print(f"Chain shape: {all_chains.shape}")

        # Calculate correlation
        all_vectors = all_chains[:, 1:, :] - all_chains[:, :-1, :]
        v_ref = all_vectors[:, 0:1, :]
        dots = np.sum(all_vectors * v_ref, axis=2)
        norms = np.linalg.norm(all_vectors, axis=2) * np.linalg.norm(v_ref,
                                                                     axis=2)
        corr_mean = np.mean(np.clip(dots / norms, -1, 1), axis=0)

        repeat_units = np.arange(1, len(corr_mean) + 1)

        # Fit exponential decay
        start_idx = 0
        end_idx = min(10, len(corr_mean))

        valid_mask = corr_mean[start_idx:end_idx] > 0
        if not np.any(valid_mask):
            print("Warning: No positive correlation values for fitting.")
            return np.inf

        x_fit = repeat_units[start_idx:end_idx][valid_mask]
        y_fit = np.log(corr_mean[start_idx:end_idx][valid_mask])

        p = np.polynomial.polynomial.polyfit(x_fit, y_fit, 1)
        corr_length = -1 / p[1] if p[1] != 0 else np.inf

        print(f"\nMonte Carlo Result:")
        print(f"Correlation Length: {corr_length:.6f}")

        if plot:
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

        if return_data:
            return corr_length

    def _square_end_to_end_distance(self, n_repeat_units, n_samples,
                                    use_cython):
        if chain_fk is None and use_cython:
            print('No cython module found.')
            use_cython = False

        chains_info = self.generate_copolymer_chains(n_repeat_units, n_samples)
        all_dihedrals = self.pre_generate_angles_copolymer_batch(chains_info)

        n_jobs = psutil.cpu_count(logical=False)

        def build_single_chain(chain_info, dihedrals):
            if use_cython:
                n = chain_info['actual_length']
                mask = chain_info['mask'][:n]
                return chain_fk.build_chain_copolymer_cy(
                    chain_info['bonds'][:n][mask[:n]],
                    chain_info['angles'][:n][mask[:n]],
                    chain_info['unit_lengths'], dihedrals)
            else:
                return self.build_chain_copolymer(chain_info, dihedrals)

        all_chains = Parallel(verbose=1, n_jobs=n_jobs)(
            delayed(build_single_chain)(chains_info[i], all_dihedrals[i])
            for i in range(len(chains_info)))

        all_chains = np.array(all_chains)

        vec = all_chains[:, :, :] - all_chains[:, 0:1, :]
        r2_results = np.sum(vec**2, axis=2)
        return r2_results

    def calc_mean_square_end_to_end_distance(self,
                                             n_repeat_units=20,
                                             n_samples=150000,
                                             return_data=False,
                                             plot=True,
                                             use_cython=True):
        """
        Calculate mean square end-to-end distance.
        
        Parameters:
        -----------
        n_repeat_units : int
            Number of repeat units
        n_samples : int
            Number of samples
        return_data : bool
            Whether to return the calculated data
        plot : bool
            Whether to plot the results
            
        Returns:
        --------
        np.ndarray (optional): Mean square distances if return_data=True
        """
        r2_results = self._square_end_to_end_distance(n_repeat_units,
                                                      n_samples, use_cython)
        r2 = np.mean(r2_results, axis=0)
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
                                 grid_points=400,
                                 bw_method='scott',
                                 plot=True,
                                 return_data=False,
                                 use_cython=True):
        """
        Calculate smooth end-to-end distance distribution using KDE.

        Parameters
        ----------
        grid_points : int
            Number of points for KDE evaluation grid.
        bw_method : str or float
            Bandwidth for gaussian_kde ('scott', 'silverman', or float).
        """

        # 1. Generate end-to-end distances
        r2_results = self._square_end_to_end_distance(
            n_repeat_units, n_samples, use_cython
        )
        r2_full = r2_results[:, -1]
        values = np.sqrt(r2_full)

        kde = gaussian_kde(values, bw_method=bw_method)

        r_min, r_max = values.min(), values.max()
        r_grid = np.linspace(r_min, r_max, grid_points)
        pdf = kde(r_grid)

        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(r_grid, pdf, 'b-', lw=2)
            xlabel = r"$R$ ($\mathrm{\AA}$)"
            ylabel = "Probability Density"
            tool.format_subplot(
                xlabel, ylabel, "End-to-End Distance Distribution (KDE)"
            )
            plt.show()

        if return_data:
            return r_grid, pdf

    def calc_mean_end_to_end_monte_carlo(self,
                                         n_repeat_units=20,
                                         n_samples=150000,
                                         plot=True,
                                         return_data=False,
                                         use_cython=True):
        """Plots the mean square end-to-end distance as a function of repeat units from 1 to N.
        Args:
           n_repeat_units (int): Maximum number of repeat units to plot
           n_samples (int): Number of samples to use in Monte Carlo simulation
           plot (bool): If True, plots the mean end-to-end distance
           return_data (bool): If True, returns the mean end-to-end distance values as a list
        """
        r2List = self._square_end_to_end_distance(n_repeat_units, n_samples,
                                                  use_cython)
        r = np.mean(np.sqrt(r2List), axis=0)
        n_repeats = np.arange(0, n_repeat_units + 1)
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(n_repeats, r, linewidth=2, color='blue', marker='o')
            tool.format_subplot("Number of Repeat Units (N)",
                                "Mean End-to-End Distance (Å)",
                                "Monte Carlo Simulation of <R>")
            plt.tight_layout()
            plt.show()
        if return_data:
            return r

    def calc_mean_r4_monte_carlo(self,
                                 n_repeat_units=20,
                                 n_samples=150000,
                                 plot=True,
                                 return_data=False,
                                 use_cython=True):
        """Plots the mean square end-to-end distance as a function of repeat units from 1 to N.
        Args:
           n_repeat_units (int): Maximum number of repeat units to plot
           n_samples (int): Number of samples to use in Monte Carlo simulation
           plot (bool): If True, plots the <R^4> values
           return_data (bool): If True, returns the <R^4> as a list
        """
        r2List = self._square_end_to_end_distance(n_repeat_units, n_samples,
                                                  use_cython)
        r4 = np.mean(r2List**2, axis=0)
        n_repeats = np.arange(0, n_repeat_units + 1)
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(n_repeats, r4, linewidth=2, color='blue', marker='o')
            tool.format_subplot("Number of Repeat Units (N)", "<R$^4$> (Å)",
                                "Monte Carlo Simulation of <R$^4$>")
            plt.tight_layout()
            plt.show()
        if return_data:
            return r4

    def wormlikechain_fitting_from_monte_carlo(self,
                                               n_repeat_units=20,
                                               n_samples=150000,
                                               use_cython=True):
        """
        Fit the Worm-like Chain model to Monte Carlo simulation results.

        Returns:
            N_eff (float): Persistence length (in units of repeat units).
            alpha (float): Scaling factor (sqrt of alpha_sq).
        """
        r2_data = self.calc_mean_square_end_to_end_distance(
            n_repeat_units=n_repeat_units,
            n_samples=n_samples,
            return_data=True,
            plot=False,
            use_cython=use_cython)
        n_values = np.arange(len(r2_data))

        def wlc_model(n, N_eff, alpha_sq):
            r2 = np.zeros_like(n, dtype=np.float64)
            mask = n > 0
            n_val = n[mask]
            term = 1 - (N_eff / n_val) * (1 - np.exp(-n_val / N_eff))
            r2[mask] = 2 * N_eff * alpha_sq * n_val * term
            return r2

        p0 = [2, self.generate_copolymer_chains(1, 1)[0]['bonds'].sum()**2]
        bounds = ([0, 0], [np.inf, np.inf])

        try:
            popt, _ = curve_fit(wlc_model,
                                n_values,
                                r2_data,
                                p0=p0,
                                bounds=bounds)
            N_eff_fit, alpha_sq_fit = popt
            alpha_fit = np.sqrt(alpha_sq_fit)
        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            return None, None
        print(f"N_eff_fit: {N_eff_fit:.3f}\nalpha: {alpha_fit:.3f}\n" +
              f"Lp: {N_eff_fit * alpha_fit:.3f} Å")
        n_smooth = np.linspace(0, len(r2_data) - 1, 200)
        r2_fit = wlc_model(n_smooth, *popt)

        plt.figure(figsize=(6, 5))
        plt.plot(n_values, r2_data, 'bo', label='Monte Carlo Data')
        plt.plot(
            n_smooth,
            r2_fit,
            'r-',
            alpha=0.7,
            linewidth=2,
            label=
            f'WLC Fit\n$N_{{eff}}={N_eff_fit:.3f}$\n$\\alpha={alpha_fit:.3f}$')

        tool.format_subplot("Number of Repeat Units (N)",
                            "Mean Square End-to-End Distance (Å$^2$)",
                            "Worm-like Chain Fitting")
        plt.legend()
        plt.tight_layout()
        plt.show()

        return N_eff_fit, alpha_fit
