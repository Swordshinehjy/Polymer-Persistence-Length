import re
from typing import List, Tuple

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
                        if coords and len(
                                coords
                        ) > 2:  # Need at least 3 atoms to be valid
                            all_coordinates.append((coords, atoms))

                    if all_coordinates:
                        break

            if not all_coordinates:
                print(
                    "Coordinate information not found, trying broader search..."
                )
                # Try broader search
                lines = content.split('\n')
                coords, atoms = self._parse_raw_lines(lines)
                if coords:
                    all_coordinates.append((coords, atoms))

            if not all_coordinates:
                print(
                    "Error: Unable to extract coordinate information from file"
                )
                print("Please confirm this is a valid Gaussian output file")
                return False

            # Take the last coordinate set as the final optimized structure
            self.final_structure, self.atom_types = all_coordinates[-1]
            print(
                f"Successfully read final optimized structure with {len(self.final_structure)} atoms"
            )
            print(
                f"Atom position number range: 1 to {len(self.final_structure)}"
            )
            self.final_structure = np.array(self.final_structure)
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
                atom_num, atomic_num, atom_type, x, y, z = match.groups(
                )  # Extract matched groups
                coordinates.append(np.array([
                    float(x), float(y), float(z)
                ]))  # Convert coordinates to array and add to list
                # Determine atom type based on atomic number
                atom_symbol = self._atomic_number_to_symbol(
                    int(atomic_num))  # Convert atomic number to atom symbol
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

    def _validate_atom_sequence(self, atom_sequence: List[int]) -> np.ndarray:
        """
        Validate input atom indices and convert to 0-based numpy array.
        """
        if self.final_structure is None:
            raise ValueError("Please read the Gaussian output file first.")

        # Ensure all indices are integers
        if not all(
                isinstance(atom, (int, np.integer)) for atom in atom_sequence):
            raise TypeError("All atom indices must be integers.")

        # Convert to 0-based index
        indices = np.asarray(atom_sequence, dtype=int) - 1

        # Check range
        n_atoms = self.final_structure.shape[0]
        if np.any(indices < 0) or np.any(indices >= n_atoms):
            raise ValueError(f"Atom number out of range: {atom_sequence}")

        return indices

    def calculate_bond_vectors(self, atom_sequence: List[int]) -> np.ndarray:
        """
        Calculate normalized bond vectors for a sequence of atoms.
        """
        indices = self._validate_atom_sequence(atom_sequence)

        # Compute bond vectors
        bond_vectors = self.final_structure[
            indices[1:]] - self.final_structure[indices[:-1]]

        # Normalize (avoid division by zero)
        norms = np.linalg.norm(bond_vectors, axis=1, keepdims=True)
        bond_vectors_normalized = bond_vectors / np.where(norms == 0, 1, norms)

        return bond_vectors_normalized

    def calculate_deflection_angles(self,
                                    atom_sequence: List[int]) -> np.ndarray:
        """
        Calculate signed deflection angles between sequential bond vectors.

        Steps:
        1. Use cross(bond1, bond0) to define the reference Z-axis.
        2. For each pair of adjacent bond vectors, compute cross products.
        3. Sign of angle determined by dot(cross_current, z_ref).
        """
        if len(atom_sequence) < 3:
            raise ValueError(
                "At least 3 atoms are required to calculate deflection angles."
            )

        bond_vectors = self.calculate_bond_vectors(atom_sequence)

        if len(bond_vectors) < 2:
            return np.array([])

        # Reference Z direction
        z_ref = np.cross(bond_vectors[0], bond_vectors[1])
        norm_z = np.linalg.norm(z_ref)
        z_ref = z_ref / norm_z if norm_z > 1e-6 else np.zeros(3)

        # Adjacent pairs
        v1 = bond_vectors[:-1]
        v2 = bond_vectors[1:]

        # Angle magnitudes (via dot product)
        dots = np.einsum("ij,ij->i", v1, v2)
        dots = np.clip(dots, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(dots))

        # Signs (via reference Z axis)
        crosses = np.cross(v1, v2)
        signs = np.sign(np.einsum("ij,j->i", crosses, z_ref))

        return angles_deg * signs

    def calculate_bond_lengths(self, atom_sequence: List[int]) -> np.ndarray:
        """
        Compute bond lengths between sequential atoms in the sequence.
        """
        indices = self._validate_atom_sequence(atom_sequence)
        diffs = self.final_structure[indices[:-1]] - self.final_structure[
            indices[1:]]
        return np.linalg.norm(diffs, axis=1)
