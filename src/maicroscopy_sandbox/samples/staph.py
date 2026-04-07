import math
import numpy as np

from .sample import Sample
from dataclasses import dataclass
from skimage.draw import ellipse_perimeter
from skimage.morphology import binary_erosion, binary_dilation
from skimage.filters import gaussian
from scipy.ndimage import binary_fill_holes


class StaphMembrane(Sample):
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

    @staticmethod
    def sample_septum_tilt_deg():
        """
        Sample a septum tilt relative to the image plane.

        The distribution is biased toward near-perpendicular septa while
        still allowing shallower orientations:
        - 0-15 degrees: 5%
        - 15-60 degrees: 55%
        - 60-90 degrees: 40%
        """
        tilt_bucket = np.random.choice(
            [0, 1, 2], p=[0.05, 0.55, 0.40]
        )
        if tilt_bucket == 0:
            return np.random.uniform(0.0, 15.0)
        if tilt_bucket == 1:
            return np.random.uniform(15.0, 60.0)
        return np.random.uniform(60.0, 90.0)

    @staticmethod
    def sample_septum_rotation_deg():
        """Sample the in-plane orientation of the septum projection."""
        return np.random.uniform(0.0, 180.0)

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
            self.sample_septum_tilt_deg(),
            self.sample_septum_rotation_deg(),
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
            septum_phase = "none"
            sep_completion = 0.0
            if cell.progression >= cell.p1:
                if cell.progression < (cell.p1 + cell.p2):
                    septum_phase = "ring"
                    sep_completion = (cell.progression - cell.p1) / cell.p2
                elif cell.progression < 100:
                    septum_phase = "closed"
                    sep_completion = 1.0

            membrane_mask, cyto_mask, septum_mask = draw_ellipse_with_axes(
                membrane_mask,
                cyto_mask,
                septum_mask,
                cell.center_row,
                cell.center_col,
                cell.major_axis,
                cell.minor_axis,
                angle_deg=np.rad2deg(cell.orientation),
                septum_tilt_deg=cell.septum_tilt_deg,
                septum_rotation_deg=cell.septum_rotation_deg,
                septum_phase=septum_phase,
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
    septum_tilt_deg=90,
    septum_rotation_deg=None,
    septum_phase="ring",
    septum_completion=1.0,
    value_membrane=1,
    value_cyto=0.5,
    value_septum=2,
):
    """
    Draws an ellipse and a modeled septum on an existing NumPy array.

    During phase 2, the septum is rendered as the projection of a closing
    ring centered at midcell. During phase 3, the septum remains the
    projected ring midcut, but in its fully closed state.

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
    septum_tilt_deg : float, optional
        Tilt of the septal ring relative to the image plane.
        90 degrees gives the narrowest projected ring and 0 degrees the
        broadest circular projection.
    septum_rotation_deg : float, optional
        In-plane rotation of the septum projection.
    septum_phase : {"none", "ring", "closed"}, optional
        Septum rendering mode based on the cell cycle phase.
    septum_completion : float, optional
        Fraction (0.0–1.0) controlling ring closure during phase 2.
        - 0.0: wide open ring
        - 1.0: fully closed thin ring
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

    # Draw the septum at midcell aligned with the cell minor axis.
    septum = draw_projected_septum(
        septum,
        cy,
        cx,
        minor_axis_length,
        septum_tilt_deg,
        septum_rotation_deg,
        septum_phase,
        septum_completion,
    )

    septum = septum * (
        ((septum > 0).astype(np.float32) - (tmp > 0).astype(np.float32)) > 0
    )
    cyto = (eroded > 0).astype(np.float32) - (septum > 0).astype(np.float32)
    septum = septum * value_septum
    cyto = cyto * value_cyto
    return membrane, cyto, septum


def draw_projected_septum(
    septum,
    cy,
    cx,
    minor_axis_length,
    septum_tilt_deg,
    septum_rotation_deg,
    septum_phase,
    septum_completion,
):
    """
    Draw the projected septum at midcell.

    Phase 2 uses a projected outer circle minus a smaller projected inner
    circle that shrinks as the cell progresses. Phase 3 keeps only the outer
    circle. The rendered septum is the midsection cut through that shape.
    """
    if septum_phase == "none":
        return septum

    if septum_rotation_deg is None:
        septum_rotation_deg = angle_deg + 90.0

    theta = np.deg2rad(septum_rotation_deg)
    tilt_rad = np.deg2rad(np.clip(septum_tilt_deg, 0.0, 90.0))

    ring_radius = max(1.5, float(minor_axis_length) * 0.98)
    projected_radius = max(0.6, ring_radius * np.cos(tilt_rad))
    ring_band_thickness = max(0.85, ring_radius * 0.22)
    max_inner_radius = max(0.5, ring_radius - ring_band_thickness)
    max_inner_projected = max(
        0.35, projected_radius - max(0.35, ring_band_thickness)
    )
    min_phase2_inner_radius = 0.55
    min_phase2_inner_projected = 0.36
    # Keep near-perpendicular septa visibly thick and taper the slice so
    # the ends preserve the curvature of the projected ring.
    cut_half_thickness = max(
        ring_band_thickness * (0.55 + 0.28 * np.sin(tilt_rad)),
        projected_radius * 0.24,
        0.9,
    )

    pad = int(np.ceil(max(ring_radius, projected_radius))) + 4
    row_min = max(0, int(cy) - pad)
    row_max = min(septum.shape[0], int(cy) + pad + 1)
    col_min = max(0, int(cx) - pad)
    col_max = min(septum.shape[1], int(cx) + pad + 1)

    rows, cols = np.mgrid[row_min:row_max, col_min:col_max]
    dy = rows - cy
    dx = cols - cx

    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)

    outer = (u / ring_radius) ** 2 + (v / projected_radius) ** 2 <= 1.0
    if septum_phase == "ring":
        septum_completion = np.clip(septum_completion, 0.0, 1.0)
        inner_radius = max(
            min_phase2_inner_radius,
            max_inner_radius * (1.0 - septum_completion),
        )
        inner_projected = max(
            min_phase2_inner_projected,
            max_inner_projected * (1.0 - septum_completion),
        )
    else:
        inner_radius = 0.0
        inner_projected = 0.0

    if inner_radius > 0.5 and inner_projected > 0.35:
        inner = (
            (u / inner_radius) ** 2 + (v / inner_projected) ** 2 <= 1.0
        )
        septum_shape = outer & ~inner
    else:
        septum_shape = outer

    taper = np.sqrt(np.clip(1.0 - (u / max(ring_radius, 1e-6)) ** 2, 0.0, 1.0))
    local_cut_half_thickness = np.maximum(
        ring_band_thickness * 0.35, cut_half_thickness * taper
    )
    midcut = np.abs(v) <= local_cut_half_thickness
    local_mask = septum_shape & midcut

    septum[row_min:row_max, col_min:col_max][local_mask] = 1
    return septum


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
    septum_tilt_deg : float
        Tilt of the septal ring relative to the image plane.
    septum_rotation_deg : float
        In-plane rotation of the septum projection.
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
    septum_tilt_deg: float
    septum_rotation_deg: float
    p1: int
    p2: int
    p3: int
    progression: int
