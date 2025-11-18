#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import re

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
        print(f"Found dihedral definition: atoms {atoms[0]}-{atoms[1]}-{atoms[2]}-{atoms[3]}")
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
    i, j, k, l = [x-1 for x in atoms]

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
    completion_patterns = [
        r'Optimization completed\.'
    ]

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

        coord_matches = list(re.finditer(coord_pattern, content_before, re.MULTILINE | re.DOTALL))

        if coord_matches:
            last_coord_match = coord_matches[-1]
            coord_block = last_coord_match.group(1).strip()
            coords = []
            for line in coord_block.split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 6:
                        # Extract x, y, z coordinates
                        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        coords.append([x, y, z])

            # Find corresponding energy (after coordinates)
            content_after_coord = content[last_coord_match.end():pos + 1000]  # Search in a range after coordinates
            energy_pattern = r'SCF Done:\s+E\([^)]+\)\s*=\s*([-\d.]+)\s+A\.U\.'
            energy_match = re.search(energy_pattern, content_after_coord)

            if not energy_match:
                # If energy not found after coordinates, search for the nearest energy before coordinates
                content_around_coord = content[max(0, last_coord_match.start()-5000):last_coord_match.end()]
                energy_matches = list(re.finditer(energy_pattern, content_around_coord))
                if energy_matches:
                    energy_match = energy_matches[-1]

            if energy_match and coords:
                energy_au = float(energy_match.group(1))
                energy_kj = energy_au * 2625.5

                # Ensure sufficient atoms to calculate dihedral angle
                if len(coords) >= max(dihedral_atoms):
                    try:
                        dihedral = calculate_dihedral_angle(coords, dihedral_atoms)
                        results.append((dihedral, energy_kj))
                        print(f"Scan point {i+1}: dihedral = {dihedral:.2f}°, energy = {energy_kj:.6f} kJ/mol")
                    except Exception as e:
                        print(f"Error calculating dihedral for scan point {i+1}: {e}")
                else:
                    print(f"Insufficient atoms for scan point {i+1}")
            else:
                print(f"No corresponding coordinates or energy found for scan point {i+1}")

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
        return # Stop if file could not be read
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
    print(f"  - Dihedral range: {min(dihedrals):.2f}° to {max(dihedrals):.2f}°")
    print(f"  - Energy range: {min(energies):.4f} to {max(energies):.4f} Hartrees")
    print(f"--> Finished processing {input_file}")