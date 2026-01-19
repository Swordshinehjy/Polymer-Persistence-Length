# Persistence Length of Semiflexible Conjugated Polymers

## Introduction

This repository is a combination of [Persistence Length Using Monte Carlo Sampling](https://github.com/Swordshinehjy/DFT_persistence_length_Monte_Carlo_sampling) and [Using Transfer Matrix](https://github.com/Swordshinehjy/persistence_length_transfer_matrix). 
## Transfer Matrix Method
Statistical Averaging of Dihedral Angles — Single-Step Average Rotation Operator
Dihedral angles are random (according to a known potential energy distribution), so for a given position i, we define the **single-step average rotation operator** as:

$$
M_i \equiv \langle Q_i(\phi)\rangle_{p_i} = \int_0^{2\pi} Q_i(\phi)p_i(\phi)d\phi,
$$

where $p_i(\phi)=\dfrac{e^{-V_i(\phi)/k_B T}}{Z_i}$, $Z_i=\int_0^{2\pi}e^{-V_i(\phi)/k_B T} d\phi$.

Since $R_z(\theta_i)$ is independent of $\phi$, the above equation can be written as:

$$
M_i = \bigg(\int_0^{2\pi} R_z(\theta_i)R_x(\phi)p_i(\phi)d\phi\bigg) \equiv R_z(\theta_i) S_i ,
$$

and

$$
\int_0^{2\pi} R_x(\phi)p(\phi)d\phi =
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
\langle t_n \rangle =  M_0 \cdots M_{n-2}M_{n-1}t_0.
$$

the autocorrelation is:

$$
C(n)=\langle t_n\cdot t_0\rangle 
=t_0^{T} \Big( \prod_{i=0}^{n-1} M_i \Big) t_0,
$$

where $\prod_{i=0}^{n-1} M_i \equiv M_0 \cdots M_{n-1}$.

If the chain is **periodic** (a repeating unit has M segments, where $A_{i+M}=A_i$), we shall calculate **the transfer matrix for one repeating unit**:

$$
\mathcal{M}=\prod_{i=0}^{M-1} M_i,
$$

Then, the correlation for $r$ repeating units decays as $\mathcal{M}^r$. Let $\lambda_{\max}$ be the maximum eigenvalue (in modulus) of $\mathcal{M}$, then the persistence length in repeating units ($N_p$) is:

$$
N_p = -\frac{1}{\ln\lambda_{\max}}.
$$
## Calculation of Mean Square End-to-End Distance <R²> and Persistence Length

This project uses Flory's Generator Matrix method to accurately calculate the mean square end-to-end distance <R²> of polymers. This method, based on transfer matrix theory, can efficiently calculate the statistical properties of chain-like polymers.

### Generator Matrix Construction

The generator matrix G_i is a 5×5 matrix with the following structure:

$$
G_i = \begin{pmatrix}
1 & 2\vec{l}_i^T M_{i+1} & l_i^2 \\
0 & M_{i+1} & \vec{l}_i \\
0 & 0 & 1
\end{pmatrix}
$$

where:
- $\vec{l}_i$ is the vector of the i-th bond
- $M_{i+1}$ is the transfer matrix of the (i+1)-th bond
- $l_i$ is the length of the i-th bond
- Due to the definition of angles and rotations (rotate first then draw bond), $l_i$ is related to $M_{i+1}$. In Flory's original definition (draw bond first then rotate), $\vec{l}_i$ is related to $M_i$.

For periodic chains, the generator matrix of one repeating unit G_unit is the product of individual G_i matrices:

$$
G_{unit} = \prod_{i=0}^{M-1} G_i
$$

### Mean Square End-to-End Distance <R²> Calculation

The mean square end-to-end distance can be calculated through powers of the generator matrix:

$$
\langle R^2 \rangle_n = G_{chain}[0,4] = (G_{unit})^n[0,4]
$$

where n is the number of repeating units. This method avoids statistical errors in Monte Carlo sampling and can accurately calculate the mean square end-to-end distance for arbitrary chain lengths.

### Persistence Length Calculation (calculate_persistence_length)

The persistence length describes the degree of directional correlation decay in polymer chains. This project provides two calculation methods:

(1) **Geometric Persistence Length**: Defined as the projection of the average end-to-end vector of an infinitely long chain onto the direction of the first bond

$$
l_p = \langle \vec{R} \cdot \hat{t}_0 \rangle = \left[(I - M)^{-1} \vec{p}\right]_0
$$

where M is the average rotation matrix and p is the average bond vector.

(2) **Worm-Like Chain Approximation Persistence Length**: Defined as the product of effective correlation length and effective unit length obtained using the worm-like chain formula.

The Worm-Like Chain (WLC) model is a continuous model for describing semiflexible polymers. In this project, equivalent WLC model parameters can be obtained by matching the asymptotic behavior of the exact mean square end-to-end distance <R²> of discrete chains.

This method is implemented through the following steps:

① Calculate the mean square end-to-end distance <R²> expression for the discrete chain:

$$
\langle R^2 \rangle = AN + B + \vec{n}^T(I-M)^{-2}M^N\vec{p}
$$

where:
- $A = s + \vec{n}^T(I-M)^{-1}\vec{p}$ (slope)
- $B = -\vec{n}^T(I-M)^{-2}\vec{p}$ (intercept)
- $\vec{n} = G_{unit}[0, 1:4], \vec{p} = G_{unit}[1:4, 4]$ are correlation vector and average unit vector extracted from the generator matrix
- $M$ is the transfer matrix of the unit, $s$ is the value of $G_{unit}[0, 4]$

② Calculate effective correlation length $N_{eff}$ and effective unit length $\alpha$:

$$
N_{eff} = -\frac{B}{A}, \quad \alpha = \sqrt{\frac{A}{2N_p}}
$$

③ Calculate worm-like chain approximation persistence length:

$$
l_p^{WLC} = \alpha \cdot N_{eff}
$$

Compared to the literature [Predicting Chain Dimensions of Semiflexible Polymers from Dihedral Potentials](https://doi.org/10.1021/ma500923r) which directly uses the correlation length $N_p$ as the persistence length in unit count, this project matches the asymptotic behavior of <R²> of discrete chains. $N_{eff}$ more realistically reflects the influence of the other two eigenvalues besides the maximum one, while also avoiding the difficulty of calculating the effective unit length of complex discrete chains.

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
*   Add the rotational isomeric state (RIS) model and the mixed HR-RIS model using the same logic (Add 'type': 'ris' to labels)









