from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy.integrate import quad
from scipy.interpolate import interp1d
from . import tool


class PolymerPlanarity:
    """
    Calculates the correlation length of a polymer planarity based on its
    molecular structure and dihedral angle potentials.

    This class encapsulates the calculations for determining the planarity information.
    """

    def __init__(self,
                 temperature=300.0,
                 rotation_types=None,
                 rotation_labels=None,
                 fitting_method='cosine',
                 param_n=15):
        """
        Initializes the Polymer Planarity model.

        Args:
            temperature (int, optional): The temperature in Kelvin. Defaults to 300.
            rotation_types (list or np.ndarray, optional): An array of integers mapping each bond to a
                                                 specific rotational potential profile. A value of 0
                                                 indicates a fixed bond with no rotation.
            rotation_labels (dict, optional): A dictionary mapping rotation_types to data files.
            ris_types (list or np.ndarray, optional): An array of integers mapping each bond to ris model.
            ris_labels (dict, optional): A dictionary mapping ris_types to data files.
            fitting_method (str, optional): The method used for fitting the data. 'interpolation', 'cosine' or 'fourier'.
            param_n (int, optional): The order for fitting the data.
        """
        self.rotation_types = np.array(rotation_types)
        self.temperature = temperature
        self.kTval = sc.R * self.temperature / 1000  # in kJ/mol

        # Default labels mapping rotation_types to data files
        self.rotation_labels = rotation_labels
        for rot_id, info in self.rotation_labels.items():
            if 'data' in info and 'label' not in info:
                self.rotation_labels[rot_id]['label'] = f"dihedral {rot_id}"
            elif 'loc' in info and 'label' not in info:
                file_path = self.rotation_labels[rot_id]['loc']
                self.rotation_labels[rot_id]['label'] = Path(file_path).stem

        self._data = {}
        self.fitting_method = fitting_method
        self.param_n = param_n
        self._cos = None
        self._cos2 = None
        self._cosabs = None

    @staticmethod
    def _update_dihedral(data):
        data = np.asarray(data)
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

    @staticmethod
    def _read_data(file_name: Path):
        """Reads and processes dihedral angle data from a file."""
        delimiter = ',' if file_name.suffix == '.csv' else None
        data = np.loadtxt(file_name, delimiter=delimiter)
        data = np.reshape(data, (-1, 2))
        return data

    def _fitting(self, angles_deg, energies):
        if self.fitting_method == 'interpolation':
            fitf = interp1d(angles_deg,
                            energies,
                            kind='cubic',
                            fill_value="extrapolate")
        elif self.fitting_method == 'cosine':
            p = np.polynomial.polynomial.polyfit(
                np.cos(np.deg2rad(angles_deg)), energies, self.param_n)
            fitf = (lambda p_val: lambda z: np.polynomial.polynomial.polyval(
                np.cos(np.deg2rad(z)), p_val))(p)
        elif self.fitting_method == 'fourier':
            rad = np.deg2rad(angles_deg)
            a = np.column_stack(
                [np.cos(n * rad) for n in range(self.param_n + 1)])
            coeffs, *_ = np.linalg.lstsq(a, energies, rcond=None)
            fitf = (lambda c, ord_val: lambda z: np.sum(
                [c[n] * np.cos(n * np.deg2rad(z)) for n in range(ord_val + 1)],
                axis=0))(coeffs, self.param_n)
        return fitf

    def _compute(self):
        """Sets up interpolation functions from data files."""
        if self._data:
            return

        for rot_id, info in self.rotation_labels.items():
            try:
                if 'data' in info:
                    data = info['data']
                elif 'loc' in info:
                    data = self._read_data(Path(info['loc']))
                else:
                    raise ValueError(
                        f"Either 'data' or 'loc' must be provided for rotation type {rot_id}."
                    )
                if 'type' in info and info['type'] == 'ris':
                    data = np.unique(data, axis=0)
                    angles_deg, energies = data[:, 0], data[:, 1]
                    angles_rad = np.deg2rad(angles_deg)

                    boltzmann_weights = np.exp(-energies / self.kTval)
                    Z = np.sum(boltzmann_weights)
                    probabilities = boltzmann_weights / Z
                    cos = np.sum(probabilities * np.cos(angles_rad))
                    cos2 = np.sum(probabilities * np.cos(angles_rad)**2)
                    cosabs = np.sum(probabilities * np.abs(np.cos(angles_rad)))
                else:  # dihedral
                    data = self._update_dihedral(data)
                    angles_deg, energies = data[:, 0], data[:, 1]
                    fitf = self._fitting(angles_deg, energies)

                    def exp_energy(phi_deg):
                        return np.exp(-fitf(phi_deg) / self.kTval)

                    Z = quad(exp_energy, 0, 360, limit=1000)[0]
                    cos = quad(
                        lambda phi: np.cos(np.deg2rad(phi)) * exp_energy(phi),
                        0,
                        360,
                        limit=1000)[0] / Z
                    cos2 = quad(lambda phi: np.cos(np.deg2rad(phi))**2 *
                                exp_energy(phi),
                                0,
                                360,
                                limit=1000)[0] / Z
                    cosabs = quad(lambda phi: np.abs(np.cos(np.deg2rad(phi))) *
                                  exp_energy(phi),
                                  0,
                                  360,
                                  limit=1000)[0] / Z
                self._data[rot_id] = {
                    'cos': cos,
                    'cos_square': cos2,
                    'cos_abs': cosabs,
                    **info
                }
            except FileNotFoundError:
                print(
                    f"Warning: Data file not found. Skipping rotation type {rot_id}."
                )
                continue
        cos_list = []
        cos2_list = []
        cosabs_list = []
        for i in range(len(self.rotation_types)):
            rot_id = int(self.rotation_types[i])
            if rot_id == 0:
                continue
            cos_list.append(self._data[rot_id]['cos'])
            cos2_list.append(self._data[rot_id]['cos_square'])
            cosabs_list.append(self._data[rot_id]['cos_abs'])
        self._cos = cos_list
        self._cos2 = cos2_list
        self._cosabs = cosabs_list

    @property
    def average_cosine(self):
        if self._cos is None:
            self._compute()
        return np.prod(self._cos)

    @property
    def average_cosine_square(self):
        if self._cos2 is None:
            self._compute()
        return np.prod(self._cos2)

    @property
    def average_absolute_cosine(self):
        if self._cosabs is None:
            self._compute()
        return np.prod(self._cosabs)

    @property
    def conformational_disorder(self):
        """σ² = <cos²φ> - <cosφ>²"""
        if self._cos2 is None:
            self._compute()
        return np.prod(self._cos2) - np.prod(self._cos)**2

    def report(self):
        if self._cos is None:
            self._compute()
        print("{:-^90}".format(" Planarity Report "))
        print(f"  Temperature: {self.temperature:.2f} K")
        for _, info in self._data.items():
            name = info["label"]
            cos = info["cos"]
            cos2 = info["cos_square"]
            cosabs = info["cos_abs"]
            print(
                f"  {name:<20}  <cosφ> = {cos:.8f}  <cos²φ> = {cos2:.8f}  <|cosφ|> = {cosabs:.8f}"
            )
        print(f"  <cosφ> for one repeat unit: {self.average_cosine:.6f}")
        print(
            f"  <cos²φ> for one repeat unit: {self.average_cosine_square:.6f}")
        print(f"  <cos²φ> - <cosφ>²: {self.conformational_disorder:.6f}")
        print(
            f"  <|cosφ|> for one repeat unit: {self.average_absolute_cosine:.6f}"
        )
        print(
            f"  Planarity correlation length -1/Log(<|cosφ|>) (repeat units): {-1/np.log(self.average_absolute_cosine):.6f}"
        )
        print("-" * 90)

    def plot_dihedral_potentials(self):
        x_values = np.linspace(0, 360, 1000)
        all_data = {}
        for rot_id, info in self.rotation_labels.items():
            if 'type' in info and info['type'] == 'ris':
                continue
            try:
                if 'data' in info:
                    data = info['data']
                elif 'loc' in info:
                    data = self._read_data(Path(info['loc']))
                else:
                    raise ValueError(
                        f"Either 'data' or 'loc' must be provided for rotation type {rot_id}."
                    )
                data = self._update_dihedral(data)
                angles_deg, energies = data[:, 0], data[:, 1]
                fitf = self._fitting(angles_deg, energies)

                all_data[rot_id] = {'fitf': fitf, 'original': data, **info}
            except:
                print(
                    f"Warning: Data file not found. Skipping rotation type {rot_id}."
                )
                continue
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        for key, data in all_data.items():
            plt.plot(data['original'][:, 0],
                     data['original'][:, 1],
                     f"{data['color']}",
                     marker="o",
                     linestyle="None",
                     label=data['label'])
            plt.plot(x_values,
                     data['fitf'](x_values),
                     color=f"{data['color']}",
                     linestyle="--")
        tool.format_subplot("Dihedral Angle [Deg.]",
                            "Dihedral Potential (kJ/mol)",
                            "Dihedral Potentials")
        plt.subplot(1, 2, 2)
        for key, data in all_data.items():
            fitf = data['fitf']
            norm_val = quad(lambda x: np.exp(-fitf(x) / self.kTval),
                            0,
                            360,
                            limit=1000)[0]
            prob_vals = np.exp(-fitf(x_values) / self.kTval) / norm_val
            plt.plot(x_values,
                     prob_vals,
                     color=f"{data['color']}",
                     linestyle="-",
                     label=data['label'])
        tool.format_subplot("Angle [deg.]", "Probability",
                            "Probability Distributions")
        plt.tight_layout()
        plt.show()

    def boltzmann_distribution_temperature_scan(self,
                                                T_range,
                                                resolution=1000,
                                                cmap='viridis',
                                                plot_type='imshow'):
        """
        Compute Boltzmann distribution p(phi,T) for a selected rotation type 
        over a temperature range, return 2D array and plot heatmap.

        Args:
            rot_id (int): rotation type ID
            T_range (array-like): temperature values (K)
            resolution (int): number of angle points (default 1000)
            plot_type (str): plotting method - 'imshow' or 'contourf' (default 'imshow')
        """
        for rot_id in self.rotation_labels:
            # Load data and build fit function
            info = self.rotation_labels[rot_id]
            if 'type' in info and info['type'] == 'ris':
                continue
            if 'data' in info:
                data = info['data']
            else:
                data = self._read_data(Path(info['loc']))
            label = info['label']

            data = self._update_dihedral(data)
            angles = np.linspace(0, 360, resolution)
            fitf = self._fitting(data[:, 0], data[:, 1])

            # Compute potential
            V = fitf(angles)

            # Compute distribution matrix: rows = T, cols = angle
            T = np.asarray(T_range)  # shape (N,)
            kT = sc.R * T[:, None] / 1000  # shape (N, 1)
            w = np.exp(-V[None, :] / kT)  # shape (N, 1) * (1, M) = (N, M)
            denom = np.trapezoid(w, angles, axis=1)[:, None]
            p_matrix = w / denom  # shape (N, M)

            plt.figure(figsize=(6, 5))

            if plot_type == 'imshow':
                # Original heatmap plot
                im = plt.imshow(p_matrix,
                                aspect='auto',
                                origin='lower',
                                cmap=cmap,
                                extent=[0, 360, T_range[0], T_range[-1]])

            elif plot_type == 'contourf':
                # Contour plot with filled colors
                # Create meshgrid for contour plot
                X, Y = np.meshgrid(angles, T)
                # For example: levels=20 or levels=np.linspace(0, p_matrix.max(), 30)
                im = plt.contourf(
                    X,
                    Y,
                    p_matrix,
                    levels=50,  # Number of contour levels
                    cmap=cmap,
                    extend='both')  # Extend colors to min/max values

                # plt.contour(X, Y, p_matrix,
                #            levels=10,  # Fewer levels for lines
                #            colors='black',
                #            linewidths=0.5,
                #            alpha=0.5)
            elif plot_type == 'contour':
                X, Y = np.meshgrid(angles, T)
                im = plt.contour(X, Y, p_matrix, levels=10, cmap=cmap)
            else:
                raise ValueError(f"plot_type not supported: {plot_type}")
            cbar = plt.colorbar(im)
            cbar.set_label("Probability", fontsize=14, fontfamily="Helvetica")
            cbar.ax.tick_params(labelsize=14)
            plt.setp(cbar.ax.get_yticklabels(), fontfamily="Helvetica")

            # Additional settings for contourf plot
            if plot_type == 'contourf' or plot_type == 'contour':
                # Set axis limits
                plt.xlim(0, 360)
                plt.ylim(T_range[0], T_range[-1])

            tool.format_subplot("Dihedral Angle (deg.)", "Temperature (K)",
                                f"Boltzmann Distribution {label}", grid=False)
            plt.show()

    def temperature_scan(self, T_range, plot=False, num_points=2000):
        """
        Vectorized version of temperature_scan.
        Replaces loop + quad integration with grid-based matrix operations.
        
        Args:
            T_range (array-like): Range of temperatures.
            plot (bool): Whether to plot results.
            num_points (int): Resolution of the integration grid (default 2000).
        """
        T_arr = np.array(T_range)
        kT_arr = sc.R * T_arr / 1000  # Shape: (N_temps,)

        # Grid for numerical integration (0 to 360 degrees)
        phi_deg = np.linspace(0, 360, num_points)
        phi_rad = np.deg2rad(phi_deg)

        # Pre-compute trigonometric terms on the grid
        # Shape: (N_points, 1) for broadcasting
        cos_phi = np.cos(phi_rad)[:, None]
        cos2_phi = (cos_phi**2)
        cosabs_phi = np.abs(cos_phi)

        # Dictionary to store results for each unique rotation ID
        # structure: {rot_id: {'cos': array_of_len_T, 'cos2': ..., 'cosabs': ...}}
        cached_results = {}

        unique_rot_ids = np.unique(self.rotation_types)
        unique_rot_ids = unique_rot_ids[unique_rot_ids
                                        != 0]  # Skip rigid bonds

        for rot_id in unique_rot_ids:
            rot_id = int(rot_id)
            info = self.rotation_labels.get(rot_id)
            if not info:
                continue
            if 'data' in info:
                data = info['data']
            elif 'loc' in info:
                data = self._read_data(Path(info['loc']))
            else:
                continue

            # Check if RIS (Discrete) or Dihedral (Continuous)
            is_ris = info.get('type') == 'ris'

            if is_ris:
                data = np.unique(data, axis=0)
                angles_deg, energies = data[:, 0], data[:, 1]

                # RIS uses discrete summation, not integration grid
                # energies shape: (N_states, 1), kT shape: (1, N_temps)
                E_matrix = energies[:, None]
                weights = np.exp(-E_matrix /
                                 kT_arr[None, :])  # (N_states, N_temps)

                Z = np.sum(weights, axis=0)

                # Discrete Angles
                ang_rad = np.deg2rad(angles_deg)[:, None]
                probs = weights / Z

                avg_cos = np.sum(probs * np.cos(ang_rad), axis=0)
                avg_cos2 = np.sum(probs * (np.cos(ang_rad)**2), axis=0)
                avg_cosabs = np.sum(probs * np.abs(np.cos(ang_rad)), axis=0)

            else:  # Dihedral - Continuous Integration
                data = self._update_dihedral(data)
                angles_data, energies_data = data[:, 0], data[:, 1]

                # Fit the energy profile once
                fitf = self._fitting(angles_data, energies_data)
                E_grid = fitf(phi_deg)[:, None]  # Shape: (N_points, 1)

                # exp( -E / kT ) -> Broadcast to (N_points, N_temps)
                weights = np.exp(-E_grid / kT_arr[None, :])
                Z = np.trapezoid(weights, x=phi_deg, axis=0)

                avg_cos = np.trapezoid(weights * cos_phi, x=phi_deg,
                                       axis=0) / Z
                avg_cos2 = np.trapezoid(weights * cos2_phi, x=phi_deg,
                                        axis=0) / Z
                avg_cosabs = np.trapezoid(
                    weights * cosabs_phi, x=phi_deg, axis=0) / Z

            cached_results[rot_id] = {
                'cos': avg_cos,
                'cos2': avg_cos2,
                'cosabs': avg_cosabs
            }

        # Initialize result arrays (ones because we use multiplication)
        total_cos = np.ones_like(T_arr)
        total_cos2 = np.ones_like(T_arr)
        total_cosabs = np.ones_like(T_arr)

        for r_type in self.rotation_types:
            r_id = int(r_type)
            if r_id == 0: continue

            if r_id in cached_results:
                total_cos *= cached_results[r_id]['cos']
                total_cos2 *= cached_results[r_id]['cos2']
                total_cosabs *= cached_results[r_id]['cosabs']

        # Calculate derived properties
        sigma2 = total_cos2 - total_cos**2

        # Handle division by zero or log of zero safely
        with np.errstate(divide='ignore', invalid='ignore'):
            corr_length = -1 / np.log(total_cosabs)

        results = {
            'T': T_arr,
            'cos': total_cos,
            'cos2': total_cos2,
            'cosabs': total_cosabs,
            'sigma2': sigma2,
            'planarity_corr_length': corr_length
        }

        if plot:
            plt.figure(figsize=(12, 10))
            params = [('cos', r"<cos$\phi$>", "Average Cosine"),
                      ('cos2', r"<cos²$\phi$>", "Average Cosine Square"),
                      ('sigma2', r"$\sigma$²", "Conformational Disorder"),
                      ('planarity_corr_length', r"-1/log(<|cos$\phi$|>)",
                       "Planarity Correlation Length")]
            for i, (key, ylabel, title) in enumerate(params, 1):
                plt.subplot(2, 2, i)
                plt.plot(results['T'], results[key], 'o-')
                tool.format_subplot("Temperature (K)", ylabel, title)

            plt.tight_layout()
            plt.show()

        return results


def compare_planarity_results(models: List[PolymerPlanarity],
                              labels: List[str],
                              ts: Union[List[float], float],
                              property='planarity_corr_length'):
    """
    Compare planarity results between different models.
    Args:
        models: List of planarity models.
        labels: List of labels for the models.
        ts: List of temperature arrays.
        property: Property to compare, e.g., 'planarity_corr_length', 'cos2'.
    """
    T_arr = np.atleast_1d(ts).astype(np.float64)
    plt.figure(figsize=(6, 5))
    for model, label in zip(models, labels):
        res = model.temperature_scan(T_arr)
        plt.plot(res['T'], res[property], 'o-', label=label)
    if property == 'planarity_corr_length':
        ylabel = r"-1/log(<|cos$\phi$|>)"
        title = "Planarity Correlation Length"
    elif property == 'cos2':
        ylabel = r"<cos²$\phi$>"
        title = "Average Cosine Square"
    tool.format_subplot('Temperature (K)', ylabel, title)
    plt.show()
