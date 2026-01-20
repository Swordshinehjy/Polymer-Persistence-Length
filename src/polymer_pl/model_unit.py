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

        # Precompute pair cache
        self._build_pair_cache()

    def _build_pair_cache(self):
        """Precompute all possible unit pairs"""
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

                self.pair_cache[key] = {
                    'bonds': np.array(b_list),
                    'angles': np.array(a_list),
                    'rotations': np.array(r_list),
                    'length': len(b_list)
                }

    def generate_copolymer_chains(self, n_repeats, n_samples):
        """
        Generate structural information for random copolymer chains
        
        Parameters:
        -----------
        n_repeats : int
            Number of repeat units
        n_samples : int
            Number of samples
            
        Returns:
        --------
        list of dict: Structural information for each sample, containing:
            - bonds: Bond length array
            - angles: Bond angle array (in radians)
            - rotations: Rotation type array
            - unit_lengths: Number of bonds per repeat unit
            - chain: Chain sequence string
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

        # Batch generate structural information
        final_results = []
        for row in chains_matrix:
            res = {
                'bonds': [],
                'angles': [],
                'rotations': [],
                'unit_lengths': [],
                'chain': ''.join(row)
            }

            for r in range(n_repeats):
                current_unit_total_len = 0

                for p in range(n_pos):
                    idx = r * n_pos + p
                    curr = row[idx]
                    nxt = row[(idx + 1) % len(row)]

                    data = self.pair_cache[curr + nxt]

                    res['bonds'].extend(data['bonds'])
                    res['angles'].extend(data['angles'])
                    res['rotations'].extend(data['rotations'])
                    current_unit_total_len += data['length']

                res['unit_lengths'].append(current_unit_total_len)

            # Angle correction
            if res['angles']:
                res['angles'] = [res['angles'][-1]] + res['angles'][:-1]

            # Convert to numpy arrays
            res['bonds'] = np.array(res['bonds'])
            res['angles'] = np.deg2rad(np.array(res['angles']))
            res['rotations'] = np.array(res['rotations'])
            res['unit_lengths'] = np.array(res['unit_lengths'])

            final_results.append(res)

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

    def pre_generate_angles_copolymer(self, chain_info):
        """
        Generate dihedral angles for a single copolymer chain
        
        Parameters:
        -----------
        chain_info : dict
            Dictionary containing 'rotations' key
            
        Returns:
        --------
        np.ndarray: Array of dihedral angles (in radians)
        """
        self._prepare_full_data()

        rotation_types = chain_info['rotations']
        n_bonds = len(rotation_types)

        rng = np.random.default_rng()
        rand_vals = rng.random(n_bonds)
        angles_deg = np.zeros(n_bonds)

        for rot_type, data_type in self._full_data.items():
            mask = rotation_types == rot_type
            if np.any(mask):
                inv_cdf = data_type['inv_cdf']
                angles_deg[mask] = inv_cdf(rand_vals[mask])

        for ris_id, (ang_deg, energies) in self.ris_data.items():
            mask = (rotation_types == ris_id)
            if not np.any(mask):
                continue

            boltz = np.exp(-energies / self.kTval)
            prob = boltz / boltz.sum()
            sampled = rng.choice(ang_deg, size=np.sum(mask), p=prob)
            angles_deg[mask] = sampled

        return np.deg2rad(angles_deg)

    def build_chain_copolymer(self, chain_info, all_dihedrals):
        """
        Build coordinates for a single copolymer chain (only recording unit endpoints)
        
        Parameters:
        -----------
        chain_info : dict
            Dictionary containing 'bonds', 'angles', 'unit_lengths'
        all_dihedrals : np.ndarray
            Array of dihedral angles (in radians)
            
        Returns:
        --------
        np.ndarray: Chain coordinates with shape (n_units+1, 3), containing only unit endpoints
        """
        all_l = chain_info['bonds']
        theta = chain_info['angles']
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
        Parameters:
        -----------
        n_repeat_units : int
            Number of repeat units to simulate
        n_samples : int
            Number of Monte Carlo samples
        plot : bool
            Whether to plot the correlation function
        return_data : bool
            Whether to return the correlation data
        Returns:
        --------
        float: Correlation length
        """
        if chain_fk is None and use_cython:
            print('No cython module found.')
            use_cython = False
        print(f"Generating {n_samples} copolymer chains...")
        chains_info = self.generate_copolymer_chains(n_repeat_units, n_samples)
        n_jobs = psutil.cpu_count(logical=False)
        print(f"Building chain coordinates using {n_jobs} parallel jobs...")

        # Build chain coordinates in parallel
        def build_single_chain(chain_info):
            dihedrals = self.pre_generate_angles_copolymer(chain_info)
            if use_cython:
                return chain_fk.build_chain_copolymer_cy(
                    chain_info['bonds'], chain_info['angles'],
                    chain_info['unit_lengths'], dihedrals)
            else:
                return self.build_chain_copolymer(chain_info, dihedrals)

        all_chains = Parallel(verbose=1, n_jobs=n_jobs)(
            delayed(build_single_chain)(chain_info)
            for chain_info in chains_info)

        all_chains = np.array(all_chains)  # shape: (n_samples, n_units+1, 3)

        print(f"Chain shape: {all_chains.shape}")

        # Calculate correlation: using unit endpoint positions
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

    def plot_chain(self, n_repeat_units, colormap='jet', rotate=False):
        """
        Plot a single polymer chain in 3D (only showing unit endpoints).
        
        Parameters:
        -----------
        n_repeat_units : int
            Number of repeat units
        colormap : str
            Colormap for visualization
        rotate : bool
            Whether to use random dihedral angles (True) or all zeros (False)
        """
        chains_info = self.generate_copolymer_chains(n_repeat_units, 1)
        chain_info = chains_info[0]

        if rotate:
            all_angles = self.pre_generate_angles_copolymer(chain_info)
        else:
            all_angles = np.zeros(len(chain_info['rotations']))

        chain = self.build_chain_copolymer(chain_info, all_angles)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(chain[:, 0],
                   chain[:, 1],
                   chain[:, 2],
                   s=50,
                   c='red',
                   marker='o')

        colors = plt.get_cmap(colormap)
        n_units = len(chain) - 1

        for i in range(n_units):
            ax.plot([chain[i][0], chain[i + 1][0]],
                    [chain[i][1], chain[i + 1][1]],
                    [chain[i][2], chain[i + 1][2]],
                    color=colors(i / n_units),
                    linewidth=2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        plt.title('Polymer Chain (Unit Endpoints)')
        plt.show()

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
        if chain_fk is None and use_cython:
            print('No cython module found.')
            use_cython = False
        chains_info = self.generate_copolymer_chains(n_repeat_units, n_samples)
        n_jobs = psutil.cpu_count(logical=False)

        # Build chain coordinates in parallel
        def build_single_chain(chain_info):
            dihedrals = self.pre_generate_angles_copolymer(chain_info)
            if use_cython:
                return chain_fk.build_chain_copolymer_cy(
                    chain_info['bonds'], chain_info['angles'],
                    chain_info['unit_lengths'], dihedrals)
            else:
                return self.build_chain_copolymer(chain_info, dihedrals)

        all_chains = Parallel(verbose=1, n_jobs=n_jobs)(
            delayed(build_single_chain)(chain_info)
            for chain_info in chains_info)

        all_chains = np.array(all_chains)  # shape: (n_samples, n_units+1, 3)

        vec = all_chains[:, :, :] - all_chains[:, 0:1:]
        r2 = np.mean(np.sum(vec**2, axis=2), axis=0)

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
                                     return_data=False,
                                     use_cython=True):
        """
        Calculate the distribution of end-to-end distance (R or R²).
        
        Parameters:
        -----------
        n_repeat_units : int
            Number of repeat units
        n_samples : int
            Number of samples
        bins : int
            Number of histogram bins
        use_r2 : bool
            If True, compute distribution of R². If False, compute R.
        density : bool
            If True, normalize histogram to PDF
        plot : bool
            Whether to plot the distribution
        return_data : bool
            Whether to return the histogram data
            
        Returns:
        --------
        tuple (optional): (bin_centers, hist) if return_data=True
        """
        if chain_fk is None and use_cython:
            print('No cython module found.')
            use_cython = False
        chains_info = self.generate_copolymer_chains(n_repeat_units, n_samples)
        n_jobs = psutil.cpu_count(logical=False)

        # Build chain coordinates in parallel
        def build_single_chain(chain_info):
            dihedrals = self.pre_generate_angles_copolymer(chain_info)
            if use_cython:
                return chain_fk.build_chain_copolymer_cy(
                    chain_info['bonds'], chain_info['angles'],
                    chain_info['unit_lengths'], dihedrals)
            else:
                return self.build_chain_copolymer(chain_info, dihedrals)

        all_chains = Parallel(verbose=1, n_jobs=n_jobs)(
            delayed(build_single_chain)(chain_info)
            for chain_info in chains_info)

        all_chains = np.array(all_chains)

        vec = all_chains[:, -1, :] - all_chains[:, 0, :]
        r2_full = np.sum(vec**2, axis=2)

        # R or R²
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
