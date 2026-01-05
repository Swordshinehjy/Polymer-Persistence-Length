# Persistence Length of Semiflexible Conjugated Polymers

## Introduction

This repository is a combination of [Persistence Length Using Monte Carlo Sampling](https://github.com/Swordshinehjy/DFT_persistence_length_Monte_Carlo_sampling) and [Using Transfer Matrix](https://github.com/Swordshinehjy/persistence_length_transfer_matrix). 
## Transfer Matrix Method
Statistical Averaging of Dihedral Angles — Single-Step Average Rotation Operator
Dihedral angles are random (according to a known potential energy distribution), so for a given position i, we define the **single-step average rotation operator** as:

$$
A_i \equiv \langle Q_i(\phi)\rangle_{p_i} = \int_0^{2\pi} Q_i(\phi)p_i(\phi)d\phi,
$$

where $p_i(\phi)=\dfrac{e^{-V_i(\phi)/k_B T}}{Z_i}$, $Z_i=\int_0^{2\pi}e^{-V_i(\phi)/k_B T} d\phi$.

Since $R_z(\theta_i)$ is independent of $\phi$, the above equation can be written as:

$$
A_i = \bigg(\int_0^{2\pi} R_z(\theta_i)R_x(\phi)p_i(\phi)d\phi\bigg) \equiv R_z(\theta_i) S_i ,
$$

and

$$
\int_0^{2\pi} R_x(\phi)p(\phi)\,d\phi
=
\begin{pmatrix}
1 & 0 & 0\\
0 & \langle\cos\phi\rangle & -\langle\sin\phi\rangle\\
0 & \langle\sin\phi\rangle & \langle\cos\phi\rangle\\
\end{pmatrix},
$$

where

$$
\langle\cos\phi\rangle_i=\frac{\int_0^{2\pi}\cos\phi e^{-V_i(\phi)/k_BT} d\phi}{\int_0^{2\pi}e^{-V_i(\phi)/k_BT} d\phi},\quad\langle\sin\phi\rangle_i=\frac{\int_0^{2\pi}\sin\phi e^{-V_i(\phi)/k_BT} d\phi}{\int_0^{2\pi}e^{-V_i(\phi)/k_BT} d\phi}.
$$

Since each step is the action of a linear operator (with independent dihedral angles), the average transformation for n steps can be written as a product of operators:

$$
\langle t_n \rangle =  A_0 \cdots A_{n-2}A_{n-1}t_0.
$$

the autocorrelation is:

$$
C(n)=\langle t_n\cdot t_0\rangle 
=t_0^{T} \Big( \prod_{i=0}^{n-1} A_i \Big) t_0,
$$

where $\prod_{i=0}^{n-1} A_i \equiv A_0 \cdots A_{n-1}$.

If the chain is **periodic** (a repeating unit has M segments, where $A_{i+M}=A_i$), we shall calculate **the transfer matrix for one repeating unit**:

$$
\mathcal{M}=\prod_{i=0}^{M-1} A_i,
$$

Then, the correlation for $r$ repeating units decays as $\mathcal{M}^r$. Let $\lambda_{\max}$ be the maximum eigenvalue (in modulus) of $\mathcal{M}$, then the persistence length in repeating units ($N_p$) is:

$$
N_p = -\frac{1}{\ln\lambda_{\max}}.
$$

## Monte Carlo Sampling Method
See the original paper [Predicting Chain Dimensions of Semiflexible Polymers from Dihedral Potentials](https://doi.org/10.1021/ma500923r) for details. Note in this repository, the definition of deflection angles is different from the paper (moveing the last angle in the original paper to the first).

## Definition of bond length and deflection angle in the script

*   T-bond-DPP-bond-T-bond-T-bond-E-bond-T-bond
*   l = [2.533, 1.432, 3.533, 1.432, 2.533, 1.432, 2.533, 1.433, 1.363, 1.433, 2.533, 1.432] # in Angstrom
*   2.533 is the bond length of Thiophene (l[0])
*   1.432 is the bond length of first linker (l[1])
*   Angle = np.deg2rad(np.array([-14.92, -10.83, 30.79, -30.79, 10.83, 14.92, -14.91, -13.29, -53.16, 53.16, 13.29, 14.91])) # convert degree to radian
*   labels = {1: {'label': 'T-DPP', 'color': 'b'}, 2: {'label': 'T-T', 'color': 'm'}, 3: {'label': 'T-E', 'color': 'c'}}
*   rotation = np.array([0, 1, 0, 1, 0, 2, 0, 3, 0, 3, 0, 2])
*   l[1] rotated by rotation_type 1 with a deflection angle Angle[1]
### Update
*   Add the Monte Carlo sampling method using Cython
*   Add the rotational isomeric state (RIS) model and the mixed HR-RIS model using the same logic

## Implementation Details

### `_calculate_Mmat` Function

The `_calculate_Mmat` method constructs the overall transformation matrix M for the repeat unit by:

1. Preparing computational data for rotation types and RIS (Rotational Isomeric State) models
2. Iterating through each bond in the repeat unit (M bonds total)
3. For each bond, determining the appropriate rotation model:
   - Fixed bond (rot_id=0, ris_id=0): m_i=1.0, s_i=0.0
   - Continuous rotation model (rot_id≠0): Uses fitted functions to compute rotation integrals
   - RIS model (ris_id≠0): Uses discrete angles and energies to compute rotation integrals
4. Constructing transformation matrices:
   - Dihedral rotation matrix (R_x) around the x-axis with m_i and s_i parameters
   - Bond angle deflection matrix (R_z) around the z-axis using bond angles
5. Multiplying all transformation matrices to obtain the overall transformation matrix Mmat

The function uses caching to improve performance by storing computed integrals for repeated rotation types and RIS IDs.




