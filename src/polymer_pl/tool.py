import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigvals
from pathlib import Path
from scipy.linalg import fractional_matrix_power


def format_subplot(xlabel, ylabel, title, grid=True):
    """Format subplot with consistent styling."""
    plt.xlabel(xlabel, fontsize=16, fontfamily="Helvetica")
    plt.ylabel(ylabel, fontsize=16, fontfamily="Helvetica")
    plt.xticks(fontsize=14, fontfamily="Helvetica")
    plt.yticks(fontsize=14, fontfamily="Helvetica")
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(fontsize=14, prop={'family': 'Helvetica'})
    if grid:
        plt.grid(True, alpha=0.3)
    plt.minorticks_on()
    plt.title(title, fontsize=18, fontfamily="Helvetica")


def read_ris_data(file_name: Path):
    delimiter = ',' if file_name.suffix == '.csv' else None
    data = np.loadtxt(file_name, delimiter=delimiter)
    data = np.reshape(data, (-1, 2))
    data = np.unique(data, axis=0)
    return data[:, 0], data[:, 1]


def compute_ris_rotation_integrals(angles_deg, energies, kTVal):
    """Compute rotation integrals for RIS model using discrete states."""
    angles_rad = np.deg2rad(angles_deg)
    boltzmann_weights = np.exp(-energies / kTVal)
    Z = np.sum(boltzmann_weights)
    probabilities = boltzmann_weights / Z
    m_i = np.sum(probabilities * np.cos(angles_rad))
    s_i = np.sum(probabilities * np.sin(angles_rad))
    return m_i, s_i


def inverse_data(filename):
    if isinstance(filename, str):
        filename = Path(filename)
    delimiter = ',' if filename.suffix == '.csv' else None
    data = np.loadtxt(filename, delimiter=delimiter)
    data = data[np.argsort(data[:, 0])]
    data_new = np.column_stack((data[:, 0][::-1], data[:, 1]))
    np.savetxt(filename.stem + "-inv.txt", data_new, fmt="%.10f")


def rotation_matrix(axis, angle):
    """
        Return the rotation matrix associated with 
        counterclockwise rotation about the given 
        axis by angle radians.
        """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def randomRotate(base_pts, angles, flat_rotation):
    pts = base_pts.copy()
    for idx, angle in enumerate(angles):
        if flat_rotation[idx] != 0:
            vec = pts[idx] - pts[idx - 1]
            axis = vec / np.linalg.norm(vec)
            rot = rotation_matrix(axis, np.deg2rad(angle))
            pts[idx + 1:] = (pts[idx + 1:] - pts[idx]) @ rot.T + pts[idx]
    return pts


def cosVals(pts, length):
    """Python implementation of cosVals function."""
    k_values = np.arange(2, len(pts), length)
    vectors = pts[k_values] - pts[k_values - 1]
    v2 = vectors[0]
    dots = vectors @ v2
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(v2)
    return np.clip(dots / norms, -1, 1)


def dihedralRotate(pts, nb, theta_deg):
    """Rotate points around the bond defined by points nb-1 and nb by theta_deg degrees."""
    theta_rad = np.deg2rad(theta_deg)
    vec = pts[nb] - pts[nb - 1]
    vec_norm = np.linalg.norm(vec)
    axis = vec / vec_norm
    rot = rotation_matrix(axis, theta_rad)
    pts[nb + 1:] = (pts[nb + 1:] - pts[nb]) @ rot.T + pts[nb]
    return pts


def compute_persistence_terpolymer(Mmat, prob):
    """
    Computes correlation length for a terpolymer made of two repeat units 
    appearing with probability prob and 1-prob.

    Mean-field approximation is used to combine the persistence lengths of the
    individual repeat units.
    Args:
    -----------
    Mmat : listlike
        List of transformation matrices for each repeat unit type
    prob : listlike
        List of probabilities for each repeat unit type
    Returns:
    --------
    float
        Correlation length
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
    Computes correlation length for a terpolymer across a range of temperatures.

    Mean-field approximation is used to combine the persistence lengths of the
    individual repeat units.
    
    This function integrates temperature_scan and compute_persistence_terpolymer
    to calculate how the correlation length of a terpolymer changes with temperature.
    
    Args:
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
    2D numpy array, row: temperature, column: correlation length
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
    corr = np.empty_like(lambda_max)

    mask_bad = lambda_max >= 1.0
    mask_good = ~mask_bad

    corr[mask_good] = -1.0 / np.log(lambda_max[mask_good])
    corr[mask_bad] = np.inf
    corr = corr.T
    if plot:
        if P == 1 and N == 1:
            # report
            print("-------------- Calculation Report -------------")
            print(f"Temperature: {Ts[0]:.2f} K")
            print(f"Max Eigenvalue (lambda_max): {lambda_max[0, 0]:.12f}")
            print(f"Correlation Length: {corr[0, 0]:.6f}")
            print("-----------------------------------------------")
        elif P == 1:
            # 1D plot: persistence vs temperature (single composition)
            lp_1d = corr[:, 0]
            finite_mask = np.isfinite(lp_1d)
            plt.figure(figsize=(6, 5))
            plt.plot(Ts[finite_mask], lp_1d[finite_mask], 'o-')
            if not np.all(finite_mask):
                # Optionally mark infinities (e.g., as flat line or annotation)
                pass
            format_subplot("Temperature (K)", "$N_p$",
                           "Correlation Length vs Temperature")
            plt.show()
        elif N == 1:
            # Fixed T, vary composition → 1D curve: lp vs composition
            lp_1d = corr[0, :]  # shape (P,)
            # Use first component probability as x-axis (assuming K >= 1)
            x = prob[:, 0]  # probability of first monomer
            finite = np.isfinite(lp_1d)
            plt.figure(figsize=(6, 5))
            plt.plot(x[finite], lp_1d[finite], 'o-')
            format_subplot("Probability of Repeat Unit 1", "$N_p$",
                           f"$N_p$ vs Composition (T = {Ts[0]:.2f} K)")
            plt.show()
        else:
            lp_plot = corr.copy()
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
            cbar.set_label("Correlation length",
                           fontsize=14,
                           fontfamily="Helvetica")
            cbar.ax.tick_params(labelsize=14)
            plt.setp(cbar.ax.get_yticklabels(), fontfamily="Helvetica")

            X, Y = np.meshgrid(prob_first_component, Ts)
            if np.any(np.isfinite(lp_plot)):
                CS = plt.contour(X, Y, lp_plot, colors='white', alpha=0.5)
                plt.clabel(CS, inline=True, fontsize=8, fmt="%.1f")
            format_subplot("Probability of Repeat Unit 1",
                           "Temperature (K)",
                           "Terpolymer Correlation Length",
                           grid=False)
            plt.show()
    return corr


def compute_persistence_alternating(model1, model2, temperature, plot=True):
    """
    Compute correlation length for alternating matrices.
    
    Args:
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
        (correlation length, maximum eigenvalue)
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

    corr = np.empty_like(lambda_max)
    mask_good = lambda_max < 1.0
    mask_bad = ~mask_good

    corr[mask_good] = -1.0 / np.log(lambda_max[mask_good])
    corr[mask_bad] = np.inf

    if is_scalar:
        T_val = float(T_arr[0])
        lp_val = corr[0]
        lambda_val = lambda_max[0]
        print("---- Alternating Copolymer Correlation Length Report ----")
        print(f"Temperature: {T_val:.2f} K")
        print(f"Max Eigenvalue (λ_max): {lambda_val:.12f}")
        if np.isinf(lp_val):
            print("Correlation Length: ∞ (rigid or semi-flexible limit)")
        else:
            print(f"Correlation Length: {lp_val:.6f}")
        print("---------------------------------------------------------")

        return lp_val
    else:
        if plot:
            plt.figure(figsize=(6, 5))
            finite_mask = np.isfinite(corr)
            if np.any(finite_mask):
                plt.plot(T_arr[finite_mask],
                         corr[finite_mask],
                         'o-',
                         color='tab:blue')
            if np.any(~finite_mask):
                pass
            format_subplot("Temperature (K)", "$N_p$",
                           "Alternating Copolymer $N_p$ vs. Temperature")
            plt.show()

        return corr


def compare_persistence_results(models, labels, temperature, property='corr'):
    """
    Compare persistence results between different models.
    Args:
        models: List of persistence models.
        labels: List of labels for the models.
        ts: List of temperature arrays.
        property: Property to compare, 'corr', 'lp', 'lp_wlc'.
    """
    T_arr = np.atleast_1d(temperature).astype(np.float64)
    plt.figure(figsize=(6, 5))
    if property == 'corr':
        ylabel = "$N_p$"
        title = "Correlation length Vs. Temperature"
        for model, label in zip(models, labels):
            res = model.temperature_scan(T_arr)
            plt.plot(res['T'], res[property], 'o-', label=label)
    elif property == 'lp':
        ylabel = "Persistence length (Å)"
        title = "Persistence length Vs. Temperature"
        for model, label in zip(models, labels):
            res = model.persistence_length_Tscan(T_arr)
            plt.plot(res['T'], res[property], 'o-', label=label)
    elif property == 'lp_wlc':
        ylabel = "Persistence length (Å)"
        title = "Persistence length WLC Vs. Temperature"
        for model, label in zip(models, labels):
            res = model.persistence_length_Tscan(T_arr)
            plt.plot(res['T'], res[property], 'o-', label=label)
    else:
        raise ValueError(f"Unknown property: {property}")
    format_subplot('Temperature (K)', ylabel, title)
    plt.show()


def compute_r2_terpolymer_Tscan(polymer_models,
                                prob_list,
                                T_list,
                                n_repeat_units: int = 20,
                                plot=True) -> np.ndarray:
    """
    Computes mean square end-to-end distance for a terpolymer across a range of temperatures.

    Mean-field approximation is used to combine the persistence lengths of the
    individual repeat units.
    
    This function integrates temperature_scan and compute_persistence_terpolymer
    to calculate how the mean square end-to-end distance of a terpolymer changes with temperature.
    
    Args:
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
    2D numpy array, row: temperature, column: mean square end-to-end distance
    """

    if not hasattr(polymer_models, '__iter__'):
        raise TypeError("polymer_models must be iterable")

    model_list = list(polymer_models)
    prob = np.asarray(prob_list, dtype=np.float64)  # (P, K)
    Ts = np.atleast_1d(T_list).astype(np.float64)  # (N,)
    P, K = prob.shape
    N = len(Ts)
    if K != len(model_list):
        raise ValueError(
            "prob_list column count must match number of polymer models")

    if not np.allclose(prob.sum(axis=1), 1.0, rtol=1e-3):
        raise ValueError("Each probability row must sum to 1.")
    # 1. Collect all M matrices at all temperatures
    #    mat_list[k] = (N, 5, 5)
    mats = np.stack(
        [m.persistence_length_Tscan(Ts)['G_unit'] for m in model_list],
        axis=0)  # (K, N, 5, 5)
    # 2. Weighted combination by prob (vectorized)
    #    For each probability set p (shape P,K):
    #    M_avg[p,n,:,:] = sum_k p[p,k] * mats[k,n,:,:]
    # prob[:, :, None, None] → (P,K,1,1,1)   broadcast
    # mats[None, :, :, :, :] → (1,K,N,5,5)
    G_avg = (prob[:, :, None, None, None] * mats[None]).sum(
        axis=1)  # (P, N, 5, 5)
    # matrix power computation using spectral decomposition error
    # so use np.linalg.matrix_power
    G_avg_flat = G_avg.reshape(P * N, 5, 5)  # (P*N, 5, 5)
    G_chain_flat = np.zeros_like(G_avg_flat)  # (P*N, 5, 5)
    for i in range(P * N):
        G_chain_flat[i] = np.linalg.matrix_power(G_avg_flat[i], n_repeat_units)
    G_chain = G_chain_flat.reshape(P, N, 5, 5)
    r2 = G_chain[:, :, 0, 4]  # (P, N)
    result = r2.T  # (N, P)
    if plot:
        if P == 1 and N == 1:
            # report
            print("-------------- Calculation Report -------------")
            print(f"Temperature: {Ts[0]:.2f} K")
            print(f"Mean Square End-to-End Distance: {result[0, 0]:.6f}")
            print("-----------------------------------------------")
        elif P == 1:
            # 1D plot: persistence vs temperature (single composition)
            lp_1d = result[:, 0]
            finite_mask = np.isfinite(lp_1d)
            plt.figure(figsize=(6, 5))
            plt.plot(Ts[finite_mask], lp_1d[finite_mask], 'o-')
            if not np.all(finite_mask):
                # Optionally mark infinities (e.g., as flat line or annotation)
                pass
            format_subplot("Temperature (K)",
                           "Mean Square End-to-End Distance (Å²)",
                           "<R²> vs Temperature")
            plt.show()
        elif N == 1:
            # Fixed T, vary composition → 1D curve: lp vs composition
            lp_1d = result[0, :]  # shape (P,)
            # Use first component probability as x-axis (assuming K >= 1)
            x = prob[:, 0]  # probability of first monomer
            finite = np.isfinite(lp_1d)
            plt.figure(figsize=(6, 5))
            plt.plot(x[finite], lp_1d[finite], 'o-')
            format_subplot("Probability of Repeat Unit 1",
                           "Mean Square End-to-End Distance (Å²)",
                           f"<R²> vs Composition (T = {Ts[0]:.2f} K)")
            plt.show()
        else:
            lp_plot = result.copy()
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
            cbar.set_label("<R²> (Å²)", fontsize=14, fontfamily="Helvetica")
            cbar.ax.tick_params(labelsize=14)
            plt.setp(cbar.ax.get_yticklabels(), fontfamily="Helvetica")

            X, Y = np.meshgrid(prob_first_component, Ts)
            if np.any(np.isfinite(lp_plot)):
                CS = plt.contour(X, Y, lp_plot, colors='white', alpha=0.5)
                plt.clabel(CS, inline=True, fontsize=8, fmt="%.1f")
            format_subplot("Probability of Repeat Unit 1",
                           "Temperature (K)",
                           "Terpolymer <R²>",
                           grid=False)
            plt.show()
    return result
