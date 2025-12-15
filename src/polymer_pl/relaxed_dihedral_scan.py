import math
import re
import bisect
from pathlib import Path
import pandas as pd
import os
from typing import List, Dict
import numpy as np


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
