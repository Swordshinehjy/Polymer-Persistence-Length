import re
from typing import Dict, List, Optional, Tuple

import numpy as np


class GaussianStructureAnalyzer:
    """
    A tool for analyzing Gaussian optimization output files and calculating deflection angles between atomic vectors
    """

    def __init__(self):
        self.coordinates = []
        self.atom_types = []
        self.final_structure = None

    def read_gaussian_output(self, filename: str) -> bool:
        """
        Read Gaussian output file and extract the final optimized structure
        
        Args:
            filename: Path to Gaussian output file
            
        Returns:
            bool: Whether the file was successfully read
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            # More comprehensive pattern matching, including different separator formats
            orientation_patterns = [
                # Standard orientation with various separators
                r'Standard orientation:.*?\n.*?-+.*?\n.*?Center.*?Coordinates.*?\n.*?-+.*?\n(.*?)(?=\n\s*-+|\n\s*\n|\n\s*\w+\s*\w|\Z)',
                # Input orientation
                r'Input orientation:.*?\n.*?-+.*?\n.*?Center.*?Coordinates.*?\n.*?-+.*?\n(.*?)(?=\n\s*-+|\n\s*\n|\n\s*\w+\s*\w|\Z)',
            ]

            all_coordinates = []

            for i, pattern in enumerate(orientation_patterns):
                matches = re.findall(pattern, content,
                                     re.DOTALL | re.MULTILINE | re.IGNORECASE)
                print(f"Pattern {i+1} found {len(matches)} matches")

                if matches:
                    for j, match in enumerate(matches):
                        coords, atoms = self._parse_coordinates_section(match)
                        if coords and len(coords) > 2:  # Need at least 3 atoms to be valid
                            all_coordinates.append((coords, atoms))

                    if all_coordinates:
                        break

            if not all_coordinates:
                print("Coordinate information not found, trying broader search...")
                # Try broader search
                lines = content.split('\n')
                coords, atoms = self._parse_raw_lines(lines)
                if coords:
                    all_coordinates.append((coords, atoms))

            if not all_coordinates:
                print("Error: Unable to extract coordinate information from file")
                print("Please confirm this is a valid Gaussian output file")
                return False

            # Take the last coordinate set as the final optimized structure
            self.final_structure, self.atom_types = all_coordinates[-1]
            print(f"Successfully read final optimized structure with {len(self.final_structure)} atoms")
            print(f"Atom position number range: 1 to {len(self.final_structure)}")
            return True

        except FileNotFoundError:
            print(f"File {filename} does not exist")
            return False
        except Exception as e:
            print(f"Error reading file: {e}")
            return False

    def _parse_coordinates_section(
            self, section: str) -> Tuple[List[np.ndarray], List[str]]:
        """
        Parse coordinate section
        
        Args:
            section: Text of the coordinate section
            
        Returns:
            coordinates: List of coordinates
            atom_types: List of atom types
        """
        coordinates = []  # List to store coordinates
        atom_types = []

        lines = section.strip().split('\n')  # Split input text by lines

        for line in lines:  # Iterate through each line
            # Match coordinate line format: atom number atomic number atom type x y z
            match = re.match(
                r'\s*(\d+)\s+(\d+)\s+(\d+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)',
                line)
            if match:  # If match is successful
                atom_num, atomic_num, atom_type, x, y, z = match.groups()  # Extract matched groups
                coordinates.append(np.array([float(x), float(y), float(z)]))  # Convert coordinates to array and add to list
                # Determine atom type based on atomic number
                atom_symbol = self._atomic_number_to_symbol(int(atomic_num))  # Convert atomic number to atom symbol
                atom_types.append(atom_symbol)  # Add atom symbol to type list

        return coordinates, atom_types  # Return coordinate list and atom type list

    def _atomic_number_to_symbol(self, atomic_num: int) -> str:
        """
        Convert atomic number to element symbol
        """
        periodic_table = {
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
            36: 'Kr'
        }
        return periodic_table.get(atomic_num, f'X{atomic_num}')

    def _parse_raw_lines(
            self, lines: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """
        Parse coordinates from raw lines (backup method)
        """
        coordinates = []
        atom_types = []
        coord_blocks = []
        current_block = []

        for line in lines:
            match = re.match(
                r'\s*(\d+)\s+(\d+)\s+(\d+)\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)',
                line)
            if match:
                current_block.append(match)
            else:
                if len(current_block) > 2:  # At least 3 atoms
                    coord_blocks.append(current_block)
                current_block = []

        if len(current_block) > 2:
            coord_blocks.append(current_block)

        if not coord_blocks:
            return [], []
        final_block = coord_blocks[-1]

        for match in final_block:
            atom_num, atomic_num, atom_type, x, y, z = match.groups()
            coordinates.append(np.array([float(x), float(y), float(z)]))
            atom_symbol = self._atomic_number_to_symbol(int(atomic_num))
            atom_types.append(atom_symbol)

        return coordinates, atom_types

    def calculate_bond_vectors(self,
                               atom_sequence: List[int]) -> List[np.ndarray]:
        if not self.final_structure:
            raise ValueError("Please read Gaussian output file first")

        bond_vectors = []

        for i in range(len(atom_sequence) - 1):
            atom1_idx = atom_sequence[i] - 1  # Convert to 0-based index
            atom2_idx = atom_sequence[i + 1] - 1

            if atom1_idx >= len(self.final_structure) or atom2_idx >= len(
                    self.final_structure):
                raise ValueError(
                    f"Atom number out of range: {atom_sequence[i]} or {atom_sequence[i+1]}")

            vector = self.final_structure[atom2_idx] - self.final_structure[
                atom1_idx]
            # Normalize
            vector_normalized = vector / np.linalg.norm(vector)
            bond_vectors.append(vector_normalized)

        return bond_vectors

    def calculate_deflection_angles(self,
                                    atom_sequence: List[int]) -> List[float]:
        """
        1. Use bond_vectors[1] × bond_vectors[0] to define the Z-axis direction of the reference coordinate system.
        2. For each deflection angle, calculate the cross product of the current two bond vectors.
        3. Dot product the cross product vector with the reference Z-axis.
        4. If the dot product is positive, the angle is positive; if negative, the angle is negative.
        """
        if len(atom_sequence) < 3:
            raise ValueError("At least 3 atoms are required to calculate deflection angles")
        bond_vectors = self.calculate_bond_vectors(atom_sequence)
        if len(bond_vectors) < 2:
            print("Warning: Less than 2 bond vectors, cannot calculate any deflection angles.")
            return []
        v0 = bond_vectors[0]
        v1 = bond_vectors[1]
        z_ref = np.cross(v0, v1)
        z_ref_norm = np.linalg.norm(z_ref)
        if z_ref_norm < 1e-6:  # Use a small tolerance
            print("Warning: The first two bond vectors are collinear and cannot define a unique reference plane. All angles will be 0 or 180 degrees, unsigned.")
            z_ref_normalized = np.array([0, 0, 0])
        else:
            z_ref_normalized = z_ref / z_ref_norm

        deflection_angles = []
        for i in range(len(bond_vectors) - 1):
            ri = bond_vectors[i]
            ri_plus_1 = bond_vectors[i + 1]
            dot_product = np.dot(ri, ri_plus_1)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.degrees(angle_rad)
            current_cross = np.cross(ri, ri_plus_1)
            sign_determinant = np.dot(current_cross, z_ref_normalized)
            signed_angle = angle_deg * np.sign(sign_determinant)
            deflection_angles.append(signed_angle)

        return deflection_angles
    def calculate_bond_lengths(self, atom_sequence: List[int]) -> List[float]:
        if not self.final_structure:
            raise ValueError("Please read Gaussian output file first")
        bond_lengths = []
        for i in range(len(atom_sequence) - 1):
            atom1_idx = atom_sequence[i] - 1
            atom2_idx = atom_sequence[i + 1] - 1
            if atom1_idx >= len(self.final_structure) or atom2_idx >= len(self.final_structure):
                raise ValueError(f"Atom number out of range: {atom_sequence[i]} or {atom_sequence[i+1]}")
            coord1 = self.final_structure[atom1_idx]
            coord2 = self.final_structure[atom2_idx]
            distance = np.linalg.norm(coord2 - coord1)
            bond_lengths.append(distance)

        return bond_lengths