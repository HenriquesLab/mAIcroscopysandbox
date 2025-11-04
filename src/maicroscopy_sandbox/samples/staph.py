import math
import numpy as np
from dataclasses import dataclass
from skimage.draw import ellipse, line, ellipse_perimeter
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import gaussian
from scipy.ndimage import binary_fill_holes


class StaphMembrane(object):
    """
    Simulates bacterial cell membrane dynamics with cell division.

    This class models the growth, septum formation, and division of
    Staphylococcus-like bacterial cells. Cells progress through three
    phases (p1: growth, p2: septum formation, p3: pre-division) before
    dividing into two daughter cells.

    Parameters
    ----------
    sample_size : np.array, default=[1000, 1000]
        Size of the sample area in pixels [height, width].
    bleaching_rate : float, default=0.001
        Rate of fluorophore bleaching per frame.
    n_objects : int, default=1
        Initial number of bacterial cells to create.
    pixel_size : int, default=100
        Size of one pixel in nanometers.
    cell_size_std : float, default=0.05
        Standard deviation of cell size as fraction of mean size.
    p1_rate : int, default=42
        Mean percentage of cell cycle spent in growth phase.
    p2_rate : int, default=29
        Mean percentage of cell cycle spent in septum formation.
    progression_rate : int, default=2
        Progression percentage points per frame (2 ≈ 1 minute real time).

    Attributes
    ----------
    cells : dict
        Dictionary of Cell objects indexed by cell ID.
    max_label : int
        Maximum cell ID used for tracking.
    """

    def __init__(
        self,
        sample_size: np.array = [1000, 1000],
        bleaching_rate: float = 0.001,
        n_objects: int = 1,
        pixel_size: int = 100,
        cell_size_std: float = 0.05,
        p1_rate: int = 42,
        p2_rate: int = 29,
        p3_rate: int = 29,
        progression_rate: int = 2,
        cyto_fluor: float = 0.4,
        axis_ratio: float = 1.4,
    ):
        self.sample_size = sample_size
        self.bleaching_rate = bleaching_rate
        self.cell_size = 1000 // pixel_size
        self.cell_size_std = self.cell_size * cell_size_std
        self.p1_rate = p1_rate
        self.p2_rate = p2_rate
        self.p3_rate = p3_rate
        self.axis_ratio = axis_ratio
        self.progression_rate = progression_rate  # percentage points per frame, 2 equals roughly 1 minute of real time
        self.cells = self.create_cells(n_objects)
        self.max_label = n_objects - 1
        self.cyto_fluor = cyto_fluor

    def create_cells(self, n_objects):
        """
        Create multiple bacterial cells at random non-overlapping positions.

        Parameters
        ----------
        n_objects : int
            Number of cells to create.

        Returns
        -------
        dict
            Dictionary of Cell objects indexed by integer IDs.
        """
        cells = {}
        coordinates = self.generate_random_coordinates(
            (self.sample_size[0], self.sample_size[1]),
            self.cell_size * 2,
            n_objects,
        )
        for i in range(n_objects):
            cells[i] = self.create_cell(coordinates[i])

        return cells

    def create_cell(self, coordinates, progression: int = None):
        """
        Create a single bacterial cell with random properties.

        Parameters
        ----------
        coordinates : tuple
            (row, col) position for cell center.

        Returns
        -------
        Cell
            New cell object with randomized size, orientation, and phase rates.
        """
        length = np.random.normal(self.cell_size, self.cell_size_std)
        cell_max_axis_ratio = np.random.normal(self.axis_ratio, 0.05)
        p1 = np.random.randint(self.p1_rate - 5, self.p1_rate + 5)
        p2 = np.random.randint(self.p2_rate - 5, self.p2_rate + 5)
        p3 = np.random.randint(self.p3_rate - 5, self.p3_rate + 5)

        if progression is None:
            progression = np.random.randint(0, 100)

        cell = Cell(
            coordinates[0],
            coordinates[1],
            length,
            length,
            cell_max_axis_ratio / (p1 + p3),
            np.random.randint(-math.pi, math.pi),
            p1,
            p2,
            p3,
            0,
        )
        self.calculate_dynamics(cell, rate=progression)
        return cell

    def generate_mask(self):
        """
        Generate fluorescence mask showing all cells with membranes and septa.

        Updates cell dynamics (growth, septum formation, division), resolves
        collisions, and renders all cells including membranes and septa based
        on their current progression phase.

        Returns
        -------
        np.ndarray
            2D array of fluorescence intensities for the sample.
        """
        mask = np.zeros(self.sample_size).astype(np.float32)

        # First, update dynamics for all cells
        # Create a list of cell_ids to avoid dictionary change during iteration
        cell_ids = list(self.cells.keys())
        for cell_id in cell_ids:
            # Skip if cell was deleted (e.g., due to division)
            if cell_id not in self.cells:
                continue
            self.calculate_dynamics(cell_id, rate=self.progression_rate)

        # Resolve all collisions after dynamics updates
        self.resolve_all_collisions()

        # Now render ALL cells (including newly created daughter cells)
        for cell_id, cell in self.cells.items():
            membrane_mask = np.zeros(self.sample_size).astype(np.float32)
            cyto_mask = np.zeros(self.sample_size).astype(np.float32)
            septum_mask = np.zeros(self.sample_size).astype(np.float32)
            sep_completion = (cell.progression - cell.p1) / cell.p2

            membrane_mask, cyto_mask, septum_mask = draw_ellipse_with_axes(
                membrane_mask,
                cyto_mask,
                septum_mask,
                cell.center_row,
                cell.center_col,
                cell.major_axis,
                cell.minor_axis,
                angle_deg=np.rad2deg(cell.orientation),
                septum_completion=sep_completion,
                value_membrane=1,
                value_cyto=self.cyto_fluor,
                value_septum=1,
            )

            mask += membrane_mask + cyto_mask + septum_mask

        return mask

    def resolve_all_collisions(self, max_iterations=10):
        """
        Iteratively resolve all cell collisions by pushing overlapping cells
        apart along the vector between their centers.
        """
        for iteration in range(max_iterations):
            collisions_found = False
            cell_ids = list(self.cells.keys())

            for i, cell_id_a in enumerate(cell_ids):
                if cell_id_a not in self.cells:
                    continue
                cell_a = self.cells[cell_id_a]

                for cell_id_b in cell_ids[i + 1 :]:
                    if cell_id_b not in self.cells:
                        continue
                    cell_b = self.cells[cell_id_b]

                    # Calculate distance between cell centers
                    dx = cell_b.center_col - cell_a.center_col
                    dy = cell_b.center_row - cell_a.center_row
                    distance = np.sqrt(dx**2 + dy**2)

                    # Minimum safe distance (sum of major axes)
                    min_safe_distance = (
                        cell_a.major_axis + cell_b.major_axis + 1
                    )

                    # Check for collision
                    if distance < min_safe_distance and distance > 0:
                        collisions_found = True

                        # Calculate overlap and push amount
                        overlap = min_safe_distance - distance
                        push_distance = (overlap / 2) + 1  # Add 1px buffer

                        # Normalize direction vector
                        dx_norm = dx / distance
                        dy_norm = dy / distance

                        # Push both cells apart along the direction vector
                        cell_a.center_row -= int(dy_norm * push_distance)
                        cell_a.center_col -= int(dx_norm * push_distance)
                        cell_b.center_row += int(dy_norm * push_distance)
                        cell_b.center_col += int(dx_norm * push_distance)

                        # Keep cells within sample boundaries
                        cell_a.center_row = np.clip(
                            cell_a.center_row, 0, self.sample_size[0] - 1
                        )
                        cell_a.center_col = np.clip(
                            cell_a.center_col, 0, self.sample_size[1] - 1
                        )
                        cell_b.center_row = np.clip(
                            cell_b.center_row, 0, self.sample_size[0] - 1
                        )
                        cell_b.center_col = np.clip(
                            cell_b.center_col, 0, self.sample_size[1] - 1
                        )

            # If no collisions found, we're done
            if not collisions_found:
                break

    def calculate_dynamics(self, cell_or_id, rate=2):
        """
        Update cell progression and trigger division if needed.

        Advances cell through growth phases (p1, p2, p3) and triggers
        cell division when progression reaches 100.

        Parameters
        ----------
        cell_or_id : Cell or int
            Either a Cell object or cell ID from self.cells dictionary.
        """
        # Accept either a Cell object or a cell_id
        if isinstance(cell_or_id, Cell):
            cell = cell_or_id
            cell_id = None
        else:
            cell_id = cell_or_id
            cell = self.cells[cell_id]

        for i in range(rate):
            cell.progression += 1
            if cell.progression < cell.p1 or (
                100 > cell.progression > (cell.p1 + cell.p2)
            ):
                # Growth phase
                cell.major_axis += cell.max_axis_increase
            elif cell.progression >= 100 and cell_id is not None:
                self.divide_cell(cell_id)
                break

    def divide_cell(self, cell_id):
        """
        Split a cell into two daughter cells along its major axis.

        Creates two daughter cells positioned along the parent's major axis,
        rotated 90 degrees from parent orientation. Handles collision
        detection and pushes neighbors away if needed.

        Parameters
        ----------
        cell_id : int
            ID of the parent cell to divide.

        Notes
        -----
        The parent cell is deleted and replaced with two daughters at
        cell IDs (max_id + 1) and (max_id + 2).
        """
        parent_cell = self.cells[cell_id]

        # Calculate positions for two daughter cells along major axis
        # Centers should be one full major axis away from each other
        # So each is offset by half the parent's major axis
        offset_distance = parent_cell.major_axis / 2

        # Calculate offset in row and col based on orientation
        # Major axis is along the orientation angle
        row_offset = offset_distance * np.sin(parent_cell.orientation)
        col_offset = offset_distance * np.cos(parent_cell.orientation)

        # Positions for the two daughter cells along the major axis
        daughter1_pos = (
            int(parent_cell.center_row - row_offset),
            int(parent_cell.center_col - col_offset),
        )
        daughter2_pos = (
            int(parent_cell.center_row + row_offset),
            int(parent_cell.center_col + col_offset),
        )

        # Create daughter cells using create_cell method
        # This ensures they have proper random variations
        daughter1 = self.create_cell(
            daughter1_pos, progression=np.random.randint(0, parent_cell.p1)
        )
        daughter2 = self.create_cell(
            daughter2_pos, progression=np.random.randint(0, parent_cell.p2)
        )

        # Rotate each daughter cell by 90 degrees from parent
        daughter1.orientation = parent_cell.orientation + np.random.normal(
            np.pi / 2, 0.2
        )
        daughter2.orientation = parent_cell.orientation + np.random.normal(
            np.pi / 2, 0.2
        )

        # Normalize orientations to [-pi, pi]
        daughter1.orientation = np.arctan2(
            np.sin(daughter1.orientation), np.cos(daughter1.orientation)
        )
        daughter2.orientation = np.arctan2(
            np.sin(daughter2.orientation), np.cos(daughter2.orientation)
        )

        # Get next available cell IDs and add daughters first
        max_id = max(self.cells.keys())
        daughter1_id = max_id + 1
        daughter2_id = max_id + 2

        # Delete parent cell first
        del self.cells[cell_id]

        # Add daughter cells temporarily to check for collisions
        self.cells[daughter1_id] = daughter1
        self.cells[daughter2_id] = daughter2

        # Check and resolve collisions for both daughters
        for daughter_id in [daughter1_id, daughter2_id]:
            daughter = self.cells[daughter_id]

            # Check collision with all other cells
            for neighbor_id, neighbor_cell in list(self.cells.items()):
                if neighbor_id == daughter_id:
                    continue

                # Calculate distance between daughter and neighbor
                dx = neighbor_cell.center_col - daughter.center_col
                dy = neighbor_cell.center_row - daughter.center_row
                distance = np.sqrt(dx**2 + dy**2)

                # Minimum safe distance (sum of major axes)
                min_safe_distance = (
                    daughter.major_axis + neighbor_cell.major_axis
                )

                # If cells overlap, push them apart
                if distance < min_safe_distance and distance > 0:
                    # Calculate push amount
                    overlap = min_safe_distance - distance
                    push_amount = overlap / 2 + 2  # Add 2px buffer

                    # Normalize direction vector
                    dx_norm = dx / distance
                    dy_norm = dy / distance

                    # Push both cells apart
                    daughter.center_row -= int(dy_norm * push_amount)
                    daughter.center_col -= int(dx_norm * push_amount)
                    neighbor_cell.center_row += int(dy_norm * push_amount)
                    neighbor_cell.center_col += int(dx_norm * push_amount)

                    # Keep within bounds
                    daughter.center_row = np.clip(
                        daughter.center_row, 0, self.sample_size[0] - 1
                    )
                    daughter.center_col = np.clip(
                        daughter.center_col, 0, self.sample_size[1] - 1
                    )
                    neighbor_cell.center_row = np.clip(
                        neighbor_cell.center_row, 0, self.sample_size[0] - 1
                    )
                    neighbor_cell.center_col = np.clip(
                        neighbor_cell.center_col, 0, self.sample_size[1] - 1
                    )

        # Update max_label
        self.max_label = max_id

    def generate_random_coordinates(self, shape, spacing, num_points):
        """
        Generate random non-overlapping coordinates for cell placement.

        Parameters
        ----------
        shape : tuple
            (height, width) of the sample area.
        spacing : float
            Minimum distance between cell centers.
        num_points : int
            Number of coordinates to generate.

        Returns
        -------
        list
            List of (row, col) tuples for cell positions.
        """
        coordinates = [(self.sample_size[0] // 2, self.sample_size[1] // 2)]
        while len(coordinates) < num_points:
            # Generate a random point
            point = (
                np.random.randint(0, shape[0]),
                np.random.randint(0, shape[1]),
            )

            # Check if the point satisfies the spacing constraint
            if all(
                np.linalg.norm(np.array(point) - np.array(coord)) >= spacing
                for coord in coordinates
            ):
                coordinates.append(point)

        return coordinates


def draw_ellipse_with_axes(
    membrane,
    cyto,
    septum,
    cy,
    cx,
    major_axis_length,
    minor_axis_length,
    angle_deg=0,
    septum_completion=1.0,
    value_membrane=1,
    value_cyto=0.5,
    value_septum=2,
):
    """
    Draws an ellipse and its axes on an existing NumPy array.
    The septum (minor axis) appears from the edges inward based on septum_completion.

    Parameters
    ----------
    image : np.ndarray
        2D NumPy array on which to draw (modified in place).
    major_axis_length : float
        Total length of the major axis.
    minor_axis_length : float
        Total length of the minor axis.
    angle_deg : float, optional
        Rotation angle in degrees (counterclockwise).
    septum_completion : float, optional
        Fraction (0.0–1.0) controlling septum closure.
        - 0.0: open (no septum drawn)
        - 1.0: closed (full minor axis drawn)
    value_ellipse : int or float, optional
        Pixel value for the ellipse perimeter.
    value_axis : int or float, optional
        Pixel value for the axes (major & septum).

    Returns
    -------
    image : np.ndarray
        The modified image array with the ellipse and axes drawn.
    """
    # Clamp the septum fraction
    septum_completion = np.clip(septum_completion, 0.0, 1.0)

    # Semi-axis lengths
    a = major_axis_length
    b = minor_axis_length

    # Draw ellipse perimeter
    rr, cc = ellipse_perimeter(
        cy,
        cx,
        int(b),
        int(a),
        orientation=np.deg2rad(angle_deg),
        shape=membrane.shape,
    )
    tmp = np.zeros((membrane.shape), dtype=np.float32)
    tmp[rr, cc] = 1
    for i in range(4):
        tmp = binary_dilation(tmp)
    cyto[rr, cc] = 1
    cyto = binary_fill_holes(cyto).astype(np.float32)
    eroded = binary_erosion(cyto)
    for i in range(4):
        eroded = binary_erosion(eroded)
    membrane = cyto - eroded
    membrane = membrane * value_membrane

    # Rotation matrix
    theta = np.deg2rad(angle_deg)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )

    # Define axis endpoints in ellipse coordinates
    minor_axis_endpoints = np.array([[0, -b], [0, b]])

    # Apply rotation and translation to image coordinates
    minor_rot = minor_axis_endpoints @ R.T + np.array([cx, cy])

    # Draw septum (minor axis) — from edges inward
    if septum_completion > 0:
        inward_fraction = 1.0 - septum_completion
        half_length = (minor_rot[1] - minor_rot[0]) / 2.0
        midpoint = (minor_rot[1] + minor_rot[0]) / 2.0

        inner_offset = half_length * inward_fraction
        top_inner = midpoint - inner_offset
        bottom_inner = midpoint + inner_offset

        # Two line segments from edges to inner tips
        segments = [(minor_rot[0], top_inner), (bottom_inner, minor_rot[1])]
        for p0, p1 in segments:
            rr, cc = line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
            septum[rr, cc] = 1

    septum = septum * (
        ((septum > 0).astype(np.float32) - (tmp > 0).astype(np.float32)) > 0
    )
    for i in range(2):
        septum = binary_dilation(septum)
    cyto = (eroded > 0).astype(np.float32) - (septum > 0).astype(np.float32)
    septum = septum * value_septum
    cyto = cyto * value_cyto
    return membrane, cyto, septum


@dataclass
class Cell:
    """
    Data class representing a single bacterial cell.

    Attributes
    ----------
    center_row : int
        Row position of cell center in pixels.
    center_col : int
        Column position of cell center in pixels.
    major_axis : int
        Length of cell major axis in pixels.
    minor_axis : int
        Length of cell minor axis in pixels.
    max_axis_increase : float
        Growth rate of major axis per frame.
    orientation : int
        Cell orientation angle in radians.
    p1 : int
        Percentage of cycle for growth phase.
    p2 : int
        Percentage of cycle for septum formation.
    p3 : int
        Percentage of cycle for pre-division phase.
    progression : int
        Current progression through cell cycle (0-100).
    """

    center_row: int
    center_col: int
    major_axis: int
    minor_axis: int
    max_axis_increase: float
    orientation: int
    p1: int
    p2: int
    p3: int
    progression: int
