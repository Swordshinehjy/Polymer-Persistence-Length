import bisect
import math
import os
import re
from collections import deque
from math import atan2, cos, degrees, radians, sin
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy.constants as sc


def read_gaussian_output(filename):
    """Read Gaussian output file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # If UTF-8 encoding fails, try other encodings
        with open(filename, 'r', encoding='latin-1') as f:
            content = f.read()
    return content


def parse_modredundant_definition(content):
    """Parse dihedral angle definition from ModRedundant section"""
    pattern = r'D\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+S\s+\d+\s+-?[\d.]+'
    matches = re.findall(pattern, content)

    if matches:
        # Gaussian numbering starts from 1, convert to integers
        atoms = [int(x) for x in matches[0]]
        print(
            f"Found dihedral definition: atoms {atoms[0]}-{atoms[1]}-{atoms[2]}-{atoms[3]}"
        )
        return atoms
    else:
        print("No dihedral definition found, please check input file")
        return None


def calculate_dihedral_angle(coords, atoms):
    """
    Calculate dihedral angle
    atoms: 4 atom indices (1-based)
    coords: coordinate array
    """
    # Convert to 0-based indices
    i, j, k, l = [x - 1 for x in atoms]

    # Get coordinates of four atoms
    r1 = np.array(coords[i])
    r2 = np.array(coords[j])
    r3 = np.array(coords[k])
    r4 = np.array(coords[l])

    # Calculate vectors
    v1 = r2 - r1
    v2 = r3 - r2
    v3 = r4 - r3

    # Calculate normal vectors
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    # Normalize
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # Calculate dihedral angle
    cos_angle = np.dot(n1, n2)
    # Ensure cos_angle is within [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle = math.acos(cos_angle)

    # Determine sign
    cross_product = np.cross(n1, n2)
    if np.dot(cross_product, v2) < 0:
        angle = -angle

    return math.degrees(angle)


def parse_optimization_steps(content, dihedral_atoms):
    """Parse final optimized structures and energies for each scan point"""
    results = []

    # Find all "Optimization completed" or "Stationary point found" positions
    completion_patterns = [r'Optimization completed\.']

    completion_positions = []
    for pattern in completion_patterns:
        matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            completion_positions.append(match.start())

    completion_positions = sorted(set(completion_positions))
    print(f"Found {len(completion_positions)} completed optimizations")

    for i, pos in enumerate(completion_positions):
        # Search for the nearest coordinates and energy before each completion position
        content_before = content[:pos]

        # Find the last Standard orientation coordinates
        coord_pattern = r'(?:Standard|Input) orientation:.*?\n.*?\n.*?\n.*?\n.*?\n((?:\s*\d+\s+\d+\s+\d+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n)+)'

        coord_matches = list(
            re.finditer(coord_pattern, content_before,
                        re.MULTILINE | re.DOTALL))

        if coord_matches:
            last_coord_match = coord_matches[-1]
            coord_block = last_coord_match.group(1).strip()
            coords = []
            for line in coord_block.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        # Extract x, y, z coordinates
                        x, y, z = float(parts[3]), float(parts[4]), float(
                            parts[5])
                        coords.append([x, y, z])

            # Find corresponding energy (after coordinates)
            content_after_coord = content[
                last_coord_match.end():pos +
                1000]  # Search in a range after coordinates
            energy_pattern = r'SCF Done:\s+E\([^)]+\)\s*=\s*([-\d.]+)\s+A\.U\.'
            energy_match = re.search(energy_pattern, content_after_coord)

            if not energy_match:
                # If energy not found after coordinates, search for the nearest energy before coordinates
                content_around_coord = content[
                    max(0,
                        last_coord_match.start() -
                        5000):last_coord_match.end()]
                energy_matches = list(
                    re.finditer(energy_pattern, content_around_coord))
                if energy_matches:
                    energy_match = energy_matches[-1]

            if energy_match and coords:
                energy_au = float(energy_match.group(1))
                energy_kj = energy_au * 2625.5

                # Ensure sufficient atoms to calculate dihedral angle
                if len(coords) >= max(dihedral_atoms):
                    try:
                        dihedral = calculate_dihedral_angle(
                            coords, dihedral_atoms)
                        results.append((dihedral, energy_kj))
                        print(
                            f"Scan point {i+1}: dihedral = {dihedral:.2f}°, energy = {energy_kj:.6f} kJ/mol"
                        )
                    except Exception as e:
                        print(
                            f"Error calculating dihedral for scan point {i+1}: {e}"
                        )
                else:
                    print(f"Insufficient atoms for scan point {i+1}")
            else:
                print(
                    f"No corresponding coordinates or energy found for scan point {i+1}"
                )

    return results


def write_results(results, output_filename):
    """Write results to txt file"""
    # Remove duplicates before writing

    # Calculate relative energies
    min_energy = results[0][1]
    relative_results = [(dihedral, energy - min_energy)
                        for dihedral, energy in results]

    with open(output_filename, 'w', encoding='utf-8') as f:
        for dihedral, rel_energy in relative_results:
            f.write(f"{dihedral:.2f}  {rel_energy:.8f}\n")

    print(f"\nResults saved to: {output_filename}")
    print(f"Total unique data points extracted: {len(results)}")
    print(f"Minimum energy (reference): {min_energy:.6f} kJ/mol")
    return results


def process_relaxed_dihedral_scan(input_file):
    """
    Orchestrates the processing of a single Gaussian output file.
    """
    print("-" * 60)
    print(f"Processing file: {input_file}")

    content = read_gaussian_output(input_file)
    if not content:
        return  # Stop if file could not be read
    dihedral_atoms = parse_modredundant_definition(content)
    # Parse optimization steps to get (dihedral, energy) pairs
    print("Parsing for scan summary...")
    results = parse_optimization_steps(content, dihedral_atoms)

    if not results:
        print(f"--> No valid scan summary data found in {input_file}")
        return

    # Define output file name and write the results
    output_file = input_file.rsplit('.', 1)[0] + '.txt'
    write_results(results, output_file)

    # Display statistics for the processed file
    dihedrals = [r[0] for r in results]
    energies = [r[1] for r in results]
    print("\nStatistics:")
    print(f"  - Points found: {len(dihedrals)}")
    print(
        f"  - Dihedral range: {min(dihedrals):.2f}° to {max(dihedrals):.2f}°")
    print(
        f"  - Energy range: {min(energies):.4f} to {max(energies):.4f} Hartrees"
    )
    print(f"--> Finished processing {input_file}")


class GaussianLogParser:
    """
    A parser for extracting geometries from Gaussian log files and exporting multi-frame XYZ files.
    """

    ELEMENT_MAP = {
        1: 'H',
        2: 'He',
        3: 'Li',
        4: 'Be',
        5: 'B',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
        10: 'Ne',
        11: 'Na',
        12: 'Mg',
        13: 'Al',
        14: 'Si',
        15: 'P',
        16: 'S',
        17: 'Cl',
        18: 'Ar',
        19: 'K',
        20: 'Ca',
        21: 'Sc',
        22: 'Ti',
        23: 'V',
        24: 'Cr',
        25: 'Mn',
        26: 'Fe',
        27: 'Co',
        28: 'Ni',
        29: 'Cu',
        30: 'Zn',
        31: 'Ga',
        32: 'Ge',
        33: 'As',
        34: 'Se',
        35: 'Br',
        36: 'Kr',
        37: 'Rb',
        38: 'Sr',
        39: 'Y',
        40: 'Zr',
        41: 'Nb',
        42: 'Mo',
        43: 'Tc',
        44: 'Ru',
        45: 'Rh',
        46: 'Pd',
        47: 'Ag',
        48: 'Cd',
        49: 'In',
        50: 'Sn',
        51: 'Sb',
        52: 'Te',
        53: 'I'
    }

    def __init__(self, filename):
        self.filename = Path(filename)
        self.component_name = self.filename.stem
        self.search_words = ['Stationary point found', 'orientation:']
        self.scan_dir = Path.cwd() / f"{self.component_name}-scan"
        self.xyz_filename = self.scan_dir / f"{self.component_name}.xyz"
        self.data = []

    def run(self):
        """Run the whole pipeline: parse → find markers → extract → write XYZ."""
        self.data = self.parse_log_file()
        keyword_lines = self.find_keyword_lines()

        stationary_lines = keyword_lines[self.search_words[0]]
        orientation_lines = keyword_lines[self.search_words[1]]

        if not stationary_lines:
            raise RuntimeError(f"No '{self.search_words[0]}' markers found.")
        if not orientation_lines:
            raise RuntimeError(f"No '{self.search_words[1]}' markers found.")

        markers = self.find_orientation_markers(stationary_lines,
                                                orientation_lines)
        if not markers:
            raise RuntimeError(
                "No orientation markers found for stationary points.")

        self.generate_xyz_file(markers)
        print(f"XYZ successfully generated: {self.xyz_filename}")

    def parse_log_file(self):
        """Reads the log file and returns its content as a list of lines."""
        try:
            with open(self.filename, 'r') as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file '{self.filename}' not found")

    def find_keyword_lines(self):
        """Finds line numbers containing specific keywords."""
        results = {kw: [] for kw in self.search_words}

        for idx, line in enumerate(self.data, 1):
            for kw in self.search_words:
                if kw in line:
                    results[kw].append(idx)

        return results

    @staticmethod
    def find_orientation_markers(stationary_lines, orientation_lines):
        """Match each stationary point to the preceding orientation block."""
        orientation_sorted = sorted(orientation_lines)
        markers = []

        for s_line in stationary_lines:
            pos = bisect.bisect_right(orientation_sorted, s_line) - 1
            if pos >= 0:
                markers.append(orientation_sorted[pos])
            else:
                print(
                    f"Warning: No 'orientation:' before stationary point at line {s_line}"
                )

        return markers

    def extract_geometry(self, start_line):
        """
        Extracts atomic numbers + XYZ coords starting ~4 lines after 'orientation:'.
        """
        pattern = re.compile(
            r'^\s*\d+\s+(\d+)\s+\d+\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)'
        )

        geometry = []
        line = start_line + 4

        while line < len(self.data):
            match = pattern.match(self.data[line])
            if not match:
                break

            atomic_num = int(match.group(1))
            element = self.ELEMENT_MAP.get(atomic_num, "X")
            coords = tuple(float(match.group(i)) for i in range(2, 5))
            geometry.append((element, coords))

            line += 1

        return geometry

    def generate_xyz_file(self, markers):
        """Write all geometries into a multi-frame XYZ file."""
        self.scan_dir.mkdir(parents=True, exist_ok=True)

        with open(self.xyz_filename, 'w') as f:
            for i, marker in enumerate(markers, 1):
                geom = self.extract_geometry(marker)

                if not geom:
                    print(
                        f"Warning: No geometry extracted for marker at line {marker}"
                    )
                    continue

                f.write(f"{len(geom)}\n")
                f.write(f"Frame {i} from log line {marker}\n")

                for element, (x, y, z) in geom:
                    f.write(f"{element} {x:.6f} {y:.6f} {z:.6f}\n")


class XYZDeflectionAngleCalculator:
    """
    Read multi-frame XYZ file and calculate deflection angles
    for a specified chain of atoms across all frames.
    """

    def __init__(self, xyz_file_path: str, atom_indices: List[int]):
        """
        Args:
            xyz_file_path: path to XYZ file
            atom_indices: atom index list (1-based)
        """
        self.xyz_file_path = xyz_file_path
        self.atom_indices = atom_indices
        self.frames = []

    # ======================== Reading XYZ ========================

    def read_xyz_file(self) -> List[Dict]:
        """Read multi-frame XYZ and store internally."""

        frames = []
        with open(self.xyz_file_path, 'r') as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            try:
                n_atoms = int(lines[i].strip())
                i += 1

                comment = lines[i].strip()
                i += 1

                symbols = []
                coords = []

                for j in range(n_atoms):
                    line = lines[i + j].strip().split()
                    if len(line) < 4:
                        continue

                    symbols.append(line[0])
                    coords.append(
                        [float(line[1]),
                         float(line[2]),
                         float(line[3])])

                frames.append({
                    "symbols": symbols,
                    "coordinates": np.array(coords)
                })

                i += n_atoms

            except Exception as e:
                print(f"Error reading XYZ: {e}")
                break

        self.frames = frames
        return frames

    @staticmethod
    def calculate_deflection_angle(p1, p2, p3) -> float:
        """Calculate one deflection angle."""
        v1 = p1 - p2
        v2 = p3 - p2

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        dot = np.dot(v1, v2)
        dot = max(min(dot, 1.0), -1.0)

        bond_angle = np.degrees(np.arccos(dot))
        return 180.0 - bond_angle

    def get_deflection_angles(self, frame: Dict) -> List[float]:
        """
        Calculate deflection angles for one frame.
        """
        indices = [i - 1 for i in self.atom_indices]

        for idx in indices:
            if idx < 0 or idx >= len(frame["coordinates"]):
                raise ValueError(f"Atom index {idx+1} out of range.")

        angles = []
        if len(indices) >= 3:
            for i in range(len(indices) - 2):
                p1 = frame["coordinates"][indices[i]]
                p2 = frame["coordinates"][indices[i + 1]]
                p3 = frame["coordinates"][indices[i + 2]]

                ang = self.calculate_deflection_angle(p1, p2, p3)
                angles.append(ang)

        return angles

    def calculate_all_frames(self) -> pd.DataFrame:
        """Calculate deflection angles for all frames and return a DataFrame."""

        if not self.frames:
            self.read_xyz_file()

        results = []

        for i, frame in enumerate(self.frames):
            try:
                angles = self.get_deflection_angles(frame)

                angle_names = []
                for j in range(len(angles)):
                    atoms = self.atom_indices[j:j + 3]
                    angle_names.append(
                        f"deflection_angle_{'-'.join(map(str, atoms))}")

                row = {"frame": i + 1}
                for name, ang in zip(angle_names, angles):
                    row[name] = ang

                results.append(row)

            except Exception as e:
                print(f"Error processing frame {i+1}: {e}")

        df = pd.DataFrame(results)
        return df

    def save_csv(self, output_path: str = None) -> str:
        """Compute deflection angles and save to CSV. Return the file path."""
        df = self.calculate_all_frames()

        if output_path is None:
            output_path = os.path.splitext(
                self.xyz_file_path)[0] + '_deflection_angles.csv'

        df.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

        return output_path


def build_connectivity(coords, atoms, covalent_radii=None, scale_factor=1.2):
    """
    Automatically determine bonding connectivity based on interatomic distances.

    :param coords: Atomic coordinates array (N, 3), in Ångstroms
    :param atoms: List of atomic symbols (length N)
    :param covalent_radii: Optional dictionary of covalent radii (in Å)
    :param scale_factor: Scaling factor for bond distance threshold (default 1.2)
    :return: Adjacency list where adj[i] contains indices of atoms bonded to i
    """
    if covalent_radii is None:
        # Default covalent radii (Å) from Cordero et al. (2008), Dalton Trans.
        # Noble gases are estimated/interpolated
        covalent_radii = {
            'H': 0.37,
            'He': 0.32,
            'Li': 1.34,
            'Be': 0.90,
            'B': 0.82,
            'C': 0.77,
            'N': 0.75,
            'O': 0.73,
            'F': 0.71,
            'Ne': 0.69,
            'Na': 1.54,
            'Mg': 1.30,
            'Al': 1.18,
            'Si': 1.11,
            'P': 1.10,
            'S': 1.03,
            'Cl': 0.99,
            'Ar': 0.96,
            'K': 1.93,
            'Ca': 1.71,
            'Sc': 1.48,
            'Ti': 1.36,
            'V': 1.34,
            'Cr': 1.22,
            'Mn': 1.19,
            'Fe': 1.16,
            'Co': 1.11,
            'Ni': 1.10,
            'Cu': 1.12,
            'Zn': 1.18,
            'Ga': 1.24,
            'Ge': 1.21,
            'As': 1.21,
            'Se': 1.16,
            'Br': 1.14,
            'Kr': 1.17,
            'Rb': 2.06,
            'Sr': 1.85,
            'Y': 1.63,
            'Zr': 1.54,
            'Nb': 1.47,
            'Mo': 1.38,
            'Tc': 1.28,
            'Ru': 1.25,
            'Rh': 1.25,
            'Pd': 1.20,
            'Ag': 1.28,
            'Cd': 1.36,
            'In': 1.42,
            'Sn': 1.40,
            'Sb': 1.40,
            'Te': 1.36,
            'I': 1.33,
        }

    n_atoms = len(coords)

    # Vector of covalent radii for each atom (fallback to 0.8 Å for unknown elements)
    radii = np.array([covalent_radii.get(symbol, 0.8) for symbol in atoms])

    # Compute all pairwise distance matrices efficiently
    # Shape: (n_atoms, n_atoms)
    diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)

    # Matrix of summed covalent radii
    sum_radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]

    # Bond threshold matrix
    threshold_matrix = sum_radii_matrix * scale_factor

    # Bond mask: distance <= threshold and not self (dist > 0)
    bond_mask = (dist_matrix <= threshold_matrix) & (dist_matrix > 0)

    # Extract bonded pairs (i, j) with i < j to avoid duplicates
    i_indices, j_indices = np.where(np.triu(bond_mask, k=1))

    # Build adjacency list
    adj_list = [[] for _ in range(n_atoms)]
    for i, j in zip(i_indices, j_indices):
        adj_list[i].append(j)
        adj_list[j].append(i)

    return adj_list


def bfs_rotating_group(adj_list, axis_atom1, axis_atom2):
    """
    Find atoms that need to be rotated around a given axis
    :param adj_list: connectivity list (adjacency list)
    :param axis_atom1: rotation axis atom 1 index (0-based)
    :param axis_atom2: rotation axis atom 2 index (0-based)
    :return: rotating group atom indices (0-based)
    """
    n_atoms = len(adj_list)
    visited = [False] * n_atoms
    queue = deque([axis_atom2])
    visited[axis_atom2] = True
    visited[axis_atom1] = True  # 锁定轴原子1侧

    rotating_group = []

    while queue:
        current = queue.popleft()
        rotating_group.append(current)
        for neighbor in adj_list[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    return rotating_group


def calculate_dihedral(coords, a, b, c, d):
    """
    Calculate dihedral a-b-c-d in degrees
    :param coords: coordinates of atoms (numpy array)
    :param a,b,c,d: atom indices (0-based)
    :return: dihedral angle in degrees
    """
    v1 = coords[b] - coords[a]
    v2 = coords[c] - coords[b]
    v3 = coords[d] - coords[c]

    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)

    n1_norm = n1 / np.linalg.norm(n1)
    n2_norm = n2 / np.linalg.norm(n2)
    v2_norm = v2 / np.linalg.norm(v2)
    x = np.dot(n1_norm, n2_norm)
    y = np.dot(np.cross(n1_norm, n2_norm), v2_norm)

    angle = atan2(y, x)
    return degrees(angle)


def rotate_atoms(coords, atom_indices, axis_point1, axis_point2, angle_deg):
    """
    Rotate atoms around a given axis
    :param coords: initial coordinates (numpy array)
    :param atom_indices: atom indices to rotate (0-based)
    :param axis_point1: axis point 1 (numpy array)
    :param axis_point2: axis point 2 (numpy array)
    :param angle_deg: rotation angle in degrees
    :return: rotated coordinates
    """
    axis_vec = axis_point2 - axis_point1
    axis_len = np.linalg.norm(axis_vec)
    if axis_len < 1e-6:
        raise ValueError("Invalid rotation axis: points are too close")
    axis_vec = axis_vec / axis_len

    center = axis_point1

    translated = coords.copy()
    translated[atom_indices] -= center

    theta = radians(angle_deg)
    c = cos(theta)
    s = sin(theta)
    t = 1 - c

    x, y, z = axis_vec
    rot_matrix = np.array(
        [[t * x * x + c, t * x * y - z * s, t * x * z + y * s],
         [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
         [t * x * z - y * s, t * y * z + x * s, t * z * z + c]])

    rotated = translated.copy()
    for i in atom_indices:
        rotated[i] = rot_matrix @ translated[i]

    rotated[atom_indices] += center
    return rotated


def generate_dihedral_scan_gjf(atoms,
                               coords,
                               charge,
                               spin_multiplicity,
                               method,
                               dihedral1,
                               dihedral2,
                               axis_atom1_idx,
                               axis_atom2_idx,
                               scanned_angles,
                               output_filename="dihedral_scan.gjf",
                               output_dir="."):
    """
    Generate a Gaussian input file (.gjf) for dihedral angle scan.

    :param atoms: List of atomic symbols
    :param coords: Numpy array of atomic coordinates (N, 3)
    :param charge: Molecular charge
    :param spin_multiplicity: Spin multiplicity
    :param method: Gaussian calculation method line
    :param dihedral1: First dihedral to fix (list of 4 atom indices, 1-indexed)
    :param dihedral2: Second dihedral to fix (list of 4 atom indices, 1-indexed)
    :param axis_atom1_idx: First atom of rotation axis (0-indexed)
    :param axis_atom2_idx: Second atom of rotation axis (0-indexed)
    :param scanned_angles: List of angles to scan (in degrees)
    :param output_filename: Output file name
    """
    # Convert dihedrals to 0-indexed
    dihedral1_idx = [i - 1 for i in dihedral1]
    dihedral2_idx = [i - 1 for i in dihedral2]

    # Build connectivity
    adjacency = build_connectivity(coords, atoms)

    # Find rotating atoms
    rotating_atoms = bfs_rotating_group(adjacency, axis_atom1_idx,
                                        axis_atom2_idx)
    print(f"Rotating atoms (0-based): {rotating_atoms}")

    # Calculate initial dihedral angles
    init_dih1 = calculate_dihedral(coords, *dihedral1_idx)
    init_dih2 = calculate_dihedral(coords, *dihedral2_idx)
    print(
        f"Initial Dihedral 1: {init_dih1:.2f} deg, Dihedral 2: {init_dih2:.2f} deg"
    )

    # Define rotation axis
    axis_point1 = coords[axis_atom1_idx]
    axis_point2 = coords[axis_atom2_idx]

    # Generate GJF blocks for each angle
    gjf_blocks = []

    for k, angle in enumerate(scanned_angles, 1):
        # Rotate atoms
        new_coords = rotate_atoms(coords=coords.copy(),
                                  atom_indices=rotating_atoms,
                                  axis_point1=axis_point1,
                                  axis_point2=axis_point2,
                                  angle_deg=angle)

        # Calculate current dihedrals
        current_dih1 = calculate_dihedral(new_coords, *dihedral1_idx)
        current_dih2 = calculate_dihedral(new_coords, *dihedral2_idx)
        print(
            f"{angle} deg: Dihedral 1={current_dih1:.2f} deg, Dihedral 2={current_dih2:.2f} deg"
        )

        # Create GJF block
        block_lines = []
        block_lines.append(f"%chk={output_dir}/{k}.chk")
        block_lines.append(method)
        block_lines.append("")
        block_lines.append(
            f"Dihedral Scan: {dihedral1}={angle} deg, {dihedral2}={angle} deg")
        block_lines.append("")
        block_lines.append(f"{charge} {spin_multiplicity}")

        # Add atoms and coordinates
        for i, atom in enumerate(atoms):
            x, y, z = new_coords[i]
            block_lines.append(f"{atom} {x:.12f} {y:.12f} {z:.12f}")

        # Add ModRedundant section
        block_lines.append("")
        block_lines.append(
            f"D {dihedral1[0]} {dihedral1[1]} {dihedral1[2]} {dihedral1[3]} F")
        block_lines.append(
            f"D {dihedral2[0]} {dihedral2[1]} {dihedral2[2]} {dihedral2[3]} F")
        block_lines.append("")

        gjf_blocks.append("\n".join(block_lines))

    # Join blocks with --link1--
    gjf_content = "\n\n--link1--\n".join(gjf_blocks)
    gjf_content += "\n"
    # Write to file
    output = Path(output_dir) / output_filename
    with open(output, 'w') as f:
        f.write(gjf_content)

    print(f"\nSuccessfully Generated GJF file: {output}")
    print(
        f"Contain {len(scanned_angles)} points ({scanned_angles[0]}deg to {scanned_angles[-1]}deg)"
    )


ELEMENTS = {
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
    'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
    'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm',
    'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W',
    'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'
}


def read_gjf_coords(filename):
    atoms = []
    coords = []
    lines = []
    with open(filename, 'r') as f:
        for line in f:
            lines.append(line.rstrip('\n'))

    found_charge_mult = False
    start_index = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('%') or stripped.startswith(
                '#') or not stripped:
            continue
        if re.match(r'^\s*-?\d+\s+-?\d+', stripped):
            found_charge_mult = True
            start_index = i + 1
            break

    if not found_charge_mult:
        raise ValueError("Could not find charge and multiplicity line.")

    for line in lines[start_index:]:
        s = line.strip()
        if not s:
            break
        if s.startswith('%') or s.startswith('#'):
            break
        parts = s.split()
        if len(parts) < 4:
            break
        atom_raw = parts[0]

        elem_match = re.match(r'^([A-Za-z]+)', atom_raw)
        if not elem_match:
            break
        elem = elem_match.group(1).capitalize()
        if elem not in ELEMENTS:
            break
        try:
            x, y, z = map(float, parts[1:4])
        except ValueError:
            break

        atoms.append(elem)
        coords.append([x, y, z])

    coords = np.array(coords) if coords else np.empty((0, 3))
    return atoms, coords


def gaussian_dihedral_energy_single_xyz(log_file, dihedral_atoms):

    HARTREE_TO_KJMOL = (sc.physical_constants["Hartree energy"][0] *
                        sc.Avogadro / 1000.0)
    idx = [i - 1 for i in dihedral_atoms]
    xyz_file = Path(log_file).with_suffix(".xyz")
    txt_file = Path(log_file).with_suffix(".txt")

    def dihedral(p1, p2, p3, p4):
        b0 = p1 - p2
        b1 = p3 - p2
        b2 = p4 - p3
        b1 /= np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
        return np.degrees(np.arctan2(y, x))

    def extract_last_energy(text):
        pats = [
            r"SCF Done:\s+E\(.+?\)\s+=\s+(-?\d+\.\d+)",
            r"EUMP2\s+=\s+(-?\d+\.\d+)",
        ]
        for pat in pats:
            m = re.findall(pat, text)
            if m:
                return float(m[-1])
        return None

    def extract_last_geometry(text):
        lines = text.splitlines()
        geom = []
        i = 0
        while i < len(lines):
            if any(k in lines[i] for k in (
                    "Standard orientation:",
                    "Input orientation:",
                    "Z-Matrix orientation:",
            )):
                geom = []
                i += 1
                dash = 0
                while i < len(lines):
                    if "-----" in lines[i]:
                        dash += 1
                        if dash == 2:
                            i += 1
                            continue
                        if dash == 3:
                            break
                    elif dash == 2:
                        p = lines[i].split()
                        if len(p) >= 6:
                            geom.append((
                                int(p[1]),
                                float(p[3]),
                                float(p[4]),
                                float(p[5]),
                            ))
                    i += 1
            i += 1
        return geom

    with open(log_file) as f:
        lines = f.readlines()

    frames = []
    buffer = ""

    for line in lines:
        buffer += line

        if "Stationary point found." in line:

            energy = extract_last_energy(buffer)
            geom = extract_last_geometry(buffer)

            if energy is not None and geom:
                coords = np.array([[x, y, z] for _, x, y, z in geom])
                angle = dihedral(
                    coords[idx[0]],
                    coords[idx[1]],
                    coords[idx[2]],
                    coords[idx[3]],
                )
                frames.append((angle, energy * HARTREE_TO_KJMOL, geom))

    if not frames:
        raise RuntimeError("No optimized structures found")

    energies = np.array([e for _, e, _ in frames])
    energies -= energies.min()

    with open(xyz_file, "w") as f:
        for i, ((angle, _, geom), e) in enumerate(zip(frames, energies), 1):
            f.write(f"{len(geom)}\n")
            f.write(f"step={i}  dihedral={angle:.2f} deg  "
                    f"rel_energy={e:.6f} kJ/mol\n")
            for a, x, y, z in geom:
                f.write(f"{a:2d} {x:.12f} {y:.12f} {z:.12f}\n")

    with open(txt_file, "w") as f:
        for (angle, _, _), e in zip(frames, energies):
            f.write(f"{angle:.2f} {e:.6f}\n")

    print(f"Extracted {len(frames)} optimized structures")
