from pathlib import Path
from typing import List, Dict, Union
import matplotlib.pyplot as plt
import numpy as np
import psutil
import scipy.constants as sc
from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from . import tool
from itertools import chain
from scipy.optimize import curve_fit
try:
    from . import chain_rotation_fk as chain_fk
except ImportError:
    print("Warning: chain_rotation_fk module not found.")
    chain_fk = None


class PolymerPersistenceMulti():
    """
    Multicomponent system using Forward Kinematics.
    """

    def __init__(self,
                 probs: List,
                 bond_lengths: List,
                 bond_angles_deg: List,
                 temperature: float = 300.0,
                 rotation_types: List = [],
                 rotation_labels: Dict[int, Dict] = {},
                 fitting_method='interpolation',
                 param_n=15):
        """Initialize the optimized polymer persistence model."""
        self.probs = probs
        self.bond_lengths = [np.array(unit) for unit in bond_lengths]
        self.bond_angles_rad = [
            np.deg2rad(np.array(unit_angle)) for unit_angle in bond_angles_deg
        ]

        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

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
        self.rotation_types = []
        self.ris_types = [] if len(ris_list) > 0 else None

        for rotation in rotation_types:
            rot_arr = np.array(rotation)
            if self.ris_types is not None:
                ris_arr = np.zeros_like(rot_arr)
                for ris_id in ris_list:
                    mask = rot_arr == ris_id
                    ris_arr[mask] = rot_arr[mask]
                    rot_arr[mask] = 0
                self.ris_types.append(ris_arr)
            self.rotation_types.append(rot_arr)

        self.ris_data = {}
        if self.ris_types is not None:
            for ris_id, info in self.rotation_labels.items():
                if info.get('type') == 'ris':
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

    def generate_rotation_angles(self, all_rotation, all_ris):
        if not self._full_data:
            self._prepare_full_data()
        rng = np.random.default_rng()
        rand_vals = rng.random(len(all_rotation))
        angles_deg = np.zeros(len(all_rotation))
        for rot_type, data_type in self._full_data.items():
            mask = all_rotation == rot_type
            if np.any(mask):
                inv_cdf = data_type['inv_cdf']
                angles_deg[mask] = inv_cdf(rand_vals[mask])
        if all_ris is not None:
            for ris_id, (ang_deg, energies) in self.ris_data.items():
                mask = (all_ris == ris_id)
                if not np.any(mask):
                    continue
                boltz = np.exp(-energies / self.kTval)
                prob = boltz / boltz.sum()
                sampled = rng.choice(ang_deg, size=np.sum(mask), p=prob)
                angles_deg[mask] = sampled
        return np.deg2rad(angles_deg)

    @staticmethod
    def _generate_single_sample(row, bond_lengths, bond_angles_rad,
                                rotation_types, ris_types,
                                generate_rotation_angles_func):
        l = bond_lengths
        angle = bond_angles_rad
        rotation = rotation_types

        end_idx = np.cumsum(np.array([0] + [len(l[idx]) for idx in row]))
        all_l = np.array(list(chain.from_iterable(l[idx] for idx in row)))
        all_Angle = np.array(
            list(chain.from_iterable(angle[idx] for idx in row)))
        all_rotation = np.array(
            list(chain.from_iterable(rotation[idx] for idx in row)))

        if ris_types is not None:
            all_ris = np.array(
                list(chain.from_iterable(ris_types[idx] for idx in row)))
        else:
            all_ris = None
        all_rx_angles = generate_rotation_angles_func(all_rotation, all_ris)

        c_theta, s_theta = np.cos(all_Angle), np.sin(all_Angle)
        c_phi, s_phi = np.cos(all_rx_angles), np.sin(all_rx_angles)

        n_bonds = len(all_l)
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

        coords = np.zeros((n_bonds + 1, 3))
        coords[1:] = np.cumsum(vectors, axis=0)

        return coords[end_idx]

    def generate_all_coords(self, n_samples, n_repeat_units):

        all_units = np.random.choice(np.arange(len(self.bond_lengths)),
                                     size=(n_samples, n_repeat_units),
                                     p=self.probs)
        n_jobs = psutil.cpu_count(logical=False)
        results = Parallel(verbose=1, n_jobs=n_jobs)(
            delayed(self._generate_single_sample)
            (row, self.bond_lengths, self.bond_angles_rad, self.rotation_types,
             self.ris_types, self.generate_rotation_angles)
            for row in all_units)
        return np.array(results)

    def cosVals_no_cython(self, n_samples, n_repeat_units):
        """
        Calculate cosine values using Python.
        """
        all_coords = self.generate_all_coords(n_samples, n_repeat_units)
        all_vectors = all_coords[:, 1:, :] - all_coords[:, :-1, :]
        v_ref = all_vectors[:, 0:1, :]
        dots = np.sum(all_vectors * v_ref, axis=2)
        norms = np.linalg.norm(all_vectors, axis=2) * np.linalg.norm(v_ref,
                                                                     axis=2)
        return np.mean(np.clip(dots / norms, -1, 1), axis=0)

    def r2_no_cython(self, n_samples, n_repeat_units):
        """
        Calculate R2 values using Python.
        """
        all_coords = self.generate_all_coords(n_samples, n_repeat_units)
        r2 = np.sum(all_coords**2, axis=2)
        return r2

    def _prepare_cython_data(self):
        """
        Prepare all data structures needed for Cython batch processing.
        This should be called once after initialization.
        
        Returns:
        --------
        dict containing all prepared data arrays
        """
        if not self._full_data:
            self._prepare_full_data()

        # 1. Prepare rotation CDF data (skip rotation type 0 - that means use RIS)
        rotation_ids = sorted(
            [rid for rid in self._full_data.keys() if rid != 0])

        if not rotation_ids:
            # If no rotation types, create dummy data
            max_cdf_size = 2
            n_rot_types = 1
            rotation_cdf_x = np.zeros((max_cdf_size, n_rot_types),
                                      dtype=np.float64)
            rotation_cdf_y = np.zeros((max_cdf_size, n_rot_types),
                                      dtype=np.float64)
            rotation_cdf_indices = np.zeros((n_rot_types, 2), dtype=np.int32)
            rotation_id_map = {}
        else:
            max_cdf_size = max(
                len(self._full_data[rid]['x_values']) for rid in rotation_ids)
            n_rot_types = len(rotation_ids)

            rotation_cdf_x = np.zeros((max_cdf_size, n_rot_types),
                                      dtype=np.float64)
            rotation_cdf_y = np.zeros((max_cdf_size, n_rot_types),
                                      dtype=np.float64)
            rotation_cdf_indices = np.zeros((n_rot_types, 2), dtype=np.int32)

            for idx, rot_id in enumerate(rotation_ids):
                x_vals = self._full_data[rot_id]['x_values']
                cdf_vals = self._full_data[rot_id]['cum_dist']
                cdf_normalized = cdf_vals / cdf_vals[-1]

                n = len(x_vals)
                rotation_cdf_x[:n, idx] = x_vals
                rotation_cdf_y[:n, idx] = cdf_normalized
                rotation_cdf_indices[idx, 0] = 0
                rotation_cdf_indices[idx, 1] = n

            rotation_id_map = {
                rid: idx
                for idx, rid in enumerate(rotation_ids)
            }

        # 2. Prepare RIS data (if exists)
        if self.ris_types is not None and self.ris_data:
            ris_ids = sorted(self.ris_data.keys())
            max_ris_size = max(len(self.ris_data[rid][0]) for rid in ris_ids)
            n_ris_types = len(ris_ids)

            ris_angles = np.zeros((max_ris_size, n_ris_types),
                                  dtype=np.float64)
            ris_probs = np.zeros((max_ris_size, n_ris_types), dtype=np.float64)
            ris_angle_indices = np.zeros((n_ris_types, 2), dtype=np.int32)

            for idx, ris_id in enumerate(ris_ids):
                ang_deg, energies = self.ris_data[ris_id]
                boltz = np.exp(-energies / self.kTval)
                prob = boltz / boltz.sum()

                n = len(ang_deg)
                ris_angles[:n, idx] = ang_deg
                ris_probs[:n, idx] = prob
                ris_angle_indices[idx, 0] = 0
                ris_angle_indices[idx, 1] = n

            ris_id_map = {rid: idx for idx, rid in enumerate(ris_ids)}
        else:
            # Dummy arrays if no RIS
            ris_angles = np.zeros((1, 1), dtype=np.float64)
            ris_probs = np.zeros((1, 1), dtype=np.float64)
            ris_angle_indices = np.zeros((1, 2), dtype=np.int32)
            ris_id_map = {}

        return {
            'rotation_cdf_indices': rotation_cdf_indices,
            'rotation_cdf_x': rotation_cdf_x,
            'rotation_cdf_y': rotation_cdf_y,
            'ris_angle_indices': ris_angle_indices,
            'ris_angles': ris_angles,
            'ris_probs': ris_probs,
            'rotation_id_map': rotation_id_map,
            'ris_id_map': ris_id_map
        }

    def _remap_rotation_types(self, rotation_types_list, rotation_id_map,
                              ris_id_map):
        """
        Remap rotation and RIS type IDs to contiguous indices for Cython.
        
        Logic: 
        - If rotation_type == 0, check ris_type (handle in Cython)
        - If rotation_type != 0, map to contiguous index for CDF lookup
        - Keep rotation_type == 0 as is (don't remap to -1)
        """
        remapped_rotation = []
        remapped_ris = [] if self.ris_types is not None else None

        for i, rot_types in enumerate(rotation_types_list):
            remapped_rot = np.zeros(len(rot_types), dtype=np.int32)

            for j, rt in enumerate(rot_types):
                if rt == 0:
                    # Keep as 0, will be handled specially in Cython
                    remapped_rot[j] = 0
                else:
                    # Map to contiguous index (add 1 to avoid conflict with 0)
                    if rt in rotation_id_map:
                        remapped_rot[j] = rotation_id_map[rt] + 1
                    else:
                        raise ValueError(
                            f"Rotation type {rt} not found in rotation_labels")

            remapped_rotation.append(remapped_rot)

            # Map RIS types
            if self.ris_types is not None:
                ris_types = self.ris_types[i]
                remapped_ris_unit = np.zeros(len(ris_types), dtype=np.int32)

                for j, ris_t in enumerate(ris_types):
                    if ris_t in ris_id_map:
                        # Map to contiguous index (add 1 to avoid conflict with 0)
                        remapped_ris_unit[j] = ris_id_map[ris_t] + 1
                    else:
                        # 0 or not found means no RIS
                        remapped_ris_unit[j] = 0

                remapped_ris.append(remapped_ris_unit)

        return remapped_rotation, remapped_ris

    def calculate_correlation_length_mc(self,
                                        n_repeat_units=20,
                                        n_samples=150000,
                                        return_data=False,
                                        plot=False,
                                        use_cython=True):
        """
        Optimized Monte Carlo calculation using forward kinematics for multi-component chains.
        """

        if chain_fk is None:
            print("Warning: chain_rotation_fk Cython module not available.")
            use_cython = False

        if use_cython and chain_fk is not None:
            print(f"Calculating correlations using Cython...")

            # Prepare data for Cython
            cython_data = self._prepare_cython_data()
            remapped_rotation, remapped_ris = self._remap_rotation_types(
                self.rotation_types, cython_data['rotation_id_map'],
                cython_data['ris_id_map'])

            # Split work across CPU cores
            n_jobs = psutil.cpu_count(logical=False)
            samples_per_job = n_samples // n_jobs

            print(
                f"Using {n_jobs} CPU cores, {samples_per_job} samples per core..."
            )

            def run_batch(n_samp):
                return chain_fk.batch_correlation_fk_multi(
                    self.bond_lengths, self.bond_angles_rad,
                    np.array(self.probs, dtype=np.float64),
                    cython_data['rotation_cdf_indices'],
                    cython_data['rotation_cdf_x'],
                    cython_data['rotation_cdf_y'], remapped_rotation,
                    remapped_ris, cython_data['ris_angle_indices'],
                    cython_data['ris_angles'], cython_data['ris_probs'],
                    n_samp, n_repeat_units)

            corr_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(run_batch)(samples_per_job) for _ in range(n_jobs))

            corr_results = np.vstack(corr_results)
            corr_mean = np.mean(corr_results, axis=0)
        else:
            print("Calculating correlations using Python...")
            corr_mean = self.cosVals_no_cython(n_samples, n_repeat_units)

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
        else:
            print(f"\nOptimized Monte Carlo Result:")
            print(f"Slope: {p[1]:.6f}")
            print(f"Correlation Length: {corr_length:.6f}")

        if return_data:
            return corr_length

    def _square_end_to_end_distance(self, n_repeat_units, n_samples,
                                    use_cython):
        """Calculate end-to-end distance for multi-component chains.
            return a list of r2 values
        """
        if chain_fk is None:
            print("Warning: chain_rotation_fk Cython module not available.")
            use_cython = False

        if use_cython and chain_fk is not None:
            print(f"Calculating R² using Cython...")

            cython_data = self._prepare_cython_data()
            remapped_rotation, remapped_ris = self._remap_rotation_types(
                self.rotation_types, cython_data['rotation_id_map'],
                cython_data['ris_id_map'])

            n_jobs = psutil.cpu_count(logical=False)
            samples_per_job = n_samples // n_jobs

            def run_batch(n_samp):
                return chain_fk.batch_end_to_end_multi(
                    self.bond_lengths, self.bond_angles_rad,
                    np.array(self.probs, dtype=np.float64),
                    cython_data['rotation_cdf_indices'],
                    cython_data['rotation_cdf_x'],
                    cython_data['rotation_cdf_y'], remapped_rotation,
                    remapped_ris, cython_data['ris_angle_indices'],
                    cython_data['ris_angles'], cython_data['ris_probs'],
                    n_samp, n_repeat_units)

            r2_results = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(run_batch)(samples_per_job) for _ in range(n_jobs))

            r2_results = np.vstack(r2_results)
        else:
            r2_results = self.r2_no_cython(n_samples, n_repeat_units)
        return r2_results

    def calc_mean_square_end_to_end_distance(self,
                                             n_repeat_units=20,
                                             n_samples=150000,
                                             return_data=False,
                                             plot=False,
                                             use_cython=True):
        """Calculate end-to-end distance for multi-component chains."""

        r2_results = self._square_end_to_end_distance(n_repeat_units,
                                                      n_samples, use_cython)
        r2 = np.mean(r2_results, axis=0)

        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(np.arange(n_repeat_units + 1), r2, 'bo-')
            tool.format_subplot("Number of Repeat Units (N)",
                                "Mean Square End-to-End Distance (Å²)",
                                "Forward Kinematics Simulation of <R²>")
            plt.show()

        if return_data:
            return r2

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
            tool.format_subplot("Number of Repeat Units (N)", "<$R^4$> (Å)",
                                "Monte Carlo Simulation of $<R^4>$")
            plt.tight_layout()
            plt.show()
        if return_data:
            return r4

    def calc_end_to_end_distribution(self,
                                     n_repeat_units=20,
                                     n_samples=150000,
                                     bins=100,
                                     density=True,
                                     plot=True,
                                     return_data=False,
                                     use_cython=True):
        """
        Calculate the distribution of end-to-end distance (R)
        for the FULL chain length.

        Parameters
        ----------
        density : bool
            If True, normalize histogram to PDF.
        """
        r2_results = self._square_end_to_end_distance(n_repeat_units,
                                                      n_samples, use_cython)
        r2_full = r2_results[:, -1]
        values = np.sqrt(r2_full)
        hist, bin_edges = np.histogram(values, bins=bins, density=density)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        if plot:
            plt.figure(figsize=(6, 5))
            plt.plot(bin_centers, hist, 'b-', lw=2)
            xlabel = r"$R$ ($\mathrm{\AA}$)"
            ylabel = "Probability Density" if density else "Counts"
            tool.format_subplot(xlabel, ylabel,
                                "End-to-End Distance Distribution")
            plt.show()

        if return_data:
            return bin_centers, hist

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

        p0 = [2, np.hstack(self.bond_lengths).sum()**2]
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
