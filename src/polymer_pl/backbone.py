from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from . import tool

class PolymerBackbone:
    """
    Calculates the backbone information of a polymer chain based on its
    molecular structure and dihedral angle potentials.

    This class encapsulates the calculations for determining the persistence
    length from bond lengths, bond angles, and rotational potentials
    using the matrix transformation method or the Monte Carlo method.
    """

    def __init__(self,
                 bond_lengths: Union[List[float], np.ndarray],
                 bond_angles_deg: Union[List[float], np.ndarray],
                 n_repeat_units: int = 1):
        """
        Initializes the PolymerPersistence model.

        Args:
            bond_lengths (list or np.ndarray): The lengths of the bonds in the repeat unit.
            bond_angles_deg (list or np.ndarray): The deflection angles between bonds in degrees.
            n_repeat_units (int, optional): The number of repeat units in the polymer chain. Defaults to 1.
        """
        if bond_lengths is None or bond_angles_deg is None:
            raise ValueError("Bond lengths and angles must be provided.")
        self.bond_lengths = np.asarray(bond_lengths)
        self.bond_angles_rad = np.deg2rad(np.array(bond_angles_deg))
        self.repeat_units = n_repeat_units

        self._chain = None
        self._contour_length = None
        self._ree = None
        self._bbox_corners = None
        self._bbox_dimensions = None
        self._turn_curvature = None
        self._circ_curvature = None

    def generate_chain(self):
        """Generate a polymer chain with n_repeat_units."""
        bonds = self.bond_lengths
        l_array = np.tile(bonds, self.repeat_units)
        all_l = np.vstack((l_array, np.zeros((1, l_array.shape[0])))).T
        all_angle = np.tile(self.bond_angles_rad, self.repeat_units)
        angles = np.cumsum(all_angle[1:])
        vectors = all_l[1:]
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        rotated_x = vectors[:, 0] * cos_angles - vectors[:, 1] * sin_angles
        rotated_y = vectors[:, 0] * sin_angles + vectors[:, 1] * cos_angles
        segments = np.column_stack((rotated_x, rotated_y))
        return np.cumsum(np.vstack((np.array([[0, 0], [bonds[0],
                                                       0]]), segments)),
                         axis=0)

    def calculate_curvature(self):
        """
        Calculate the curvature of the polymer chain.
        """
        l_prev = np.roll(self.bond_lengths, 1)
        l_next = self.bond_lengths
        self._turn_curvature = 4 * np.sin(
            self.bond_angles_rad / 2) / (l_prev + l_next)
        self._circ_curvature = 2 * np.sin(self.bond_angles_rad) / np.sqrt(
            l_prev**2 + l_next**2 +
            2 * l_prev * l_next * np.cos(self.bond_angles_rad))

    def calculate_minimum_bounding_box(self):
        """
        Calculate the minimum area bounding box using rotating calipers method.
        
        Returns:
            tuple: (corners, length, width, aspect_ratio)
        """
        if self._chain is None:
            self.run_calculation()

        points = self._chain
        # Get convex hull
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        # Initialize minimum area
        min_area = float('inf')
        best_box = None
        # For each edge of the convex hull
        n = len(hull_points)
        for i in range(n):
            # Get edge vector
            edge = hull_points[(i + 1) % n] - hull_points[i]
            edge_length = np.linalg.norm(edge)
            if edge_length < 1e-10:
                continue
            edge = edge / edge_length
            # Get perpendicular vector
            perp = np.array([-edge[1], edge[0]])
            # Project all points onto edge and perpendicular
            proj_edge = np.dot(hull_points, edge)
            proj_perp = np.dot(hull_points, perp)
            # Get bounding box dimensions
            length = proj_edge.max() - proj_edge.min()
            width = proj_perp.max() - proj_perp.min()
            area = length * width
            # Update minimum if this is smaller
            if area < min_area:
                min_area = area
                # Calculate corner points
                min_edge = proj_edge.min()
                max_edge = proj_edge.max()
                min_perp = proj_perp.min()
                max_perp = proj_perp.max()
                corners = np.array([
                    min_edge * edge + min_perp * perp,
                    max_edge * edge + min_perp * perp,
                    max_edge * edge + max_perp * perp,
                    min_edge * edge + max_perp * perp
                ])

                best_box = (corners, length, width,
                            length / width if width > 0 else np.inf)

        self._bbox_corners, self._bbox_length, self._bbox_width, self._bbox_aspect_ratio = best_box
        return best_box

    def run_calculation(self):
        self._chain = self.generate_chain()
        self._ree = np.linalg.norm(self._chain[-1] - self._chain[0])
        diffs = np.diff(self._chain, axis=0)
        self._contour_length = np.sum(np.linalg.norm(diffs, axis=1))
        self.calculate_minimum_bounding_box()
        self.calculate_curvature()

    @property
    def ree_lc_ratio(self):
        if self._chain is None:
            self.run_calculation()
        return self._ree / self._contour_length

    @property
    def bbox_aspect_ratio(self):
        if self._bbox_corners is None:
            self.run_calculation()
        return self._bbox_aspect_ratio

    @property
    def bbox_dimensions(self):
        if self._bbox_corners is None:
            self.run_calculation()
        return self._bbox_length, self._bbox_width

    @property
    def average_cosine_angle(self):
        if self._chain is None:
            self.run_calculation()
        cos_theta = np.cos(self.bond_angles_rad)
        return np.mean(cos_theta), np.std(cos_theta)

    def calculate_angles(self, segments: Union[List, np.ndarray]):
        """
        Calculate the angles between consecutive segments.
        Args:
            segments (List[np.ndarray] or np.ndarray): List of segments or array of segments.
            example: [[1, 2], [4, 5]] or np.array([[1, 2], [4, 5]])
        """
        if self._chain is None:
            self.run_calculation()
        orientation = self._chain[-1] - self._chain[0]
        angles = []
        for start_idx, end_idx in segments:
            segment_vector = self._chain[end_idx] - self._chain[start_idx]

            dot_product = np.dot(orientation, segment_vector)
            cross_product = np.cross(orientation, segment_vector)
            norm_orientation = np.linalg.norm(orientation)
            norm_segment = np.linalg.norm(segment_vector)
            if norm_orientation == 0 or norm_segment == 0:
                angles.append(0.0)
                continue
            cos_theta = dot_product / (norm_orientation * norm_segment)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            unsigned_angle = np.rad2deg(np.arccos(cos_theta))

            if cross_product >= 0:
                angle = unsigned_angle
            else:
                angle = -unsigned_angle
            angles.append(angle)
        return angles

    def report_angles_segments(self, segments: Union[List, np.ndarray]):
        angles = self.calculate_angles(segments)
        print("-- Angles Between Segments and Backbone Orientation --")
        for idx, angle in enumerate(angles, 1):
            print(f"Segment {idx}: {angle:.2f} degrees")
        print("------------------------------------------------------")

    def draw_chain(self, show_bbox=True):
        """
        Draw the polymer chain with optional minimum bounding box.
        
        Args:
            show_bbox (bool): Whether to show the minimum bounding box
        """
        if self._chain is None:
            self.run_calculation()

        plt.plot(self._chain[:, 0], self._chain[:, 1], 'co-', linewidth=2)

        if show_bbox and self._bbox_corners is not None:
            # Close the box by adding the first corner at the end
            bbox_closed = np.vstack(
                [self._bbox_corners, self._bbox_corners[0]])
            plt.plot(bbox_closed[:, 0],
                     bbox_closed[:, 1],
                     'r--',
                     linewidth=2,
                     alpha=0.5,
                     label=f'AR={self._bbox_aspect_ratio:.2f}')

        plt.gca().set_aspect('equal', adjustable='box')
        title = "Polymer Chain with Minimum Bounding Box" if show_bbox else "Polymer Chain"
        tool.format_subplot("x (Å)", "y (Å)", title)

    def report(self):
        """Prints a summary of the calculation results."""
        ratio = self.ree_lc_ratio
        cos_mean, cos_std = self.average_cosine_angle
        theta_mean = np.rad2deg(np.mean(self.bond_angles_rad))
        theta_std = np.rad2deg(np.std(self.bond_angles_rad))
        theta_abs_mean = np.rad2deg(np.mean(np.abs(self.bond_angles_rad)))
        theta_abs_std = np.rad2deg(np.std(np.abs(self.bond_angles_rad)))
        bbox_length, bbox_width = self.bbox_dimensions
        bbox_ar = self.bbox_aspect_ratio
        turn_curvature_mean = np.mean(self._turn_curvature)
        turn_curvature_std = np.std(self._turn_curvature)
        circ_curvature_mean = np.mean(self._circ_curvature)
        circ_curvature_std = np.std(self._circ_curvature)

        print("---- Backbone Calculation Report ----")
        print(f"Straightness Ratio (Ree/Lc): {ratio:.6f}")
        print(f"<cos(θ)>: {cos_mean:.6f} ± {cos_std:.6f}")
        print(f"<θ>: {theta_mean:.6f} ± {theta_std:.6f} (deg)")
        print(f"<|θ|>: {theta_abs_mean:.6f} ± {theta_abs_std:.6f} (deg)")
        print(
            f"Turn Curvature: {turn_curvature_mean:.6f} ± {turn_curvature_std:.6f}"
        )
        print(
            f"Circular Curvature: {circ_curvature_mean:.6f} ± {circ_curvature_std:.6f}"
        )
        print("-------- Minimum Bounding Box Analysis --------")
        print(f"Box Length/Width: {bbox_length:.4f} Å / {bbox_width:.4f} Å")
        print(f"Box Aspect Ratio (L/W): {bbox_ar:.6f}")
        print("-----------------------------------------------")
