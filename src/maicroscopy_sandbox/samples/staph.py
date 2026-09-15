import math
import numpy as np

from .sample import Sample
from dataclasses import dataclass
from typing import Callable

ELLIPSE_MORPHOLOGY_PADDING = 8
SEPTUM_VISIBILITY_EPSILON = 0.02
MEMBRANE_SHELL_THICKNESS = 0.2
SEPTUM_PLANE_HALF_WIDTH = 0.75


class StaphMembrane(Sample):
    """Staphylococcus-like sample with growth, septation, and division.

    Args:
        sample_size: Sample dimensions in pixels as ``[height, width]``.
        bleaching_rate: Per-frame bleaching rate.
        n_objects: Number of initial cells.
        pixel_size: Pixel size in nanometers.
        cell_size: Mean cell size in nanometers, used for scaling other parameters and generating initial coordinates.
        cell_size_std: Variation of cell size as a fraction of the mean cell size (0-1). #TODO - change this var name to reflect that it's for a uniform distribution.
        p1_rate: Mean percentage spent in the growth phase.
        p2_rate: Mean percentage spent in septum formation.
        p3_rate: Mean percentage spent in the pre-division phase.
        progression_rate: Cell-cycle progression per frame in percentage points.
        cyto_fluor: Fluorescence intensity used for the cytoplasm.
        axis_ratio: Mean ratio between major and minor axes.

    Attributes:
        cells: Dictionary of ``Cell`` instances keyed by cell ID.
        max_label: Highest cell ID currently assigned.
    """

    def __init__(
        self,
        sample_size: np.array = [1000, 1000],
        bleaching_rate: float = 0.001,
        n_objects: int = 1,
        pixel_size: int = 100,
        cell_size: int = 1200,
        cell_size_std: float = 0.05,
        p1_rate: int = 42,
        p2_rate: int = 29,
        p3_rate: int = 29,
        progression_rate: int = 2,
        cyto_fluor: float = 0.4,
        axis_ratio: float = 1.4,
        z_size: int | None = None,
        voxel_size: tuple[float, float] | None = None,
        septum_tilt_distribution: str
        | tuple[float, float]
        | Callable[[], float] = "biased_high",
        septum_rotation_distribution: str
        | tuple[float, float]
        | Callable[[], float] = "uniform",
    ):
        self.sample_size = sample_size
        self.bleaching_rate = bleaching_rate
        self.voxel_size = (
            (float(pixel_size), float(pixel_size))
            if voxel_size is None
            else (float(voxel_size[0]), float(voxel_size[1]))
        )
        self.z_size = (
            int(z_size)
            if z_size is not None
            else int(np.ceil((cell_size / self.voxel_size[1]) * 2.5))
        )
        self.volume_shape = (self.z_size, self.sample_size[0], self.sample_size[1])
        self.septum_tilt_distribution = septum_tilt_distribution
        self.septum_rotation_distribution = septum_rotation_distribution
        self._cell_size = cell_size  # in nanometers, used for scaling other parameters and generating initial coordinates
        self.cell_size = self._cell_size / self.voxel_size[0]
        self.cell_size_std = (self._cell_size * cell_size_std) / self.voxel_size[0]
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
        """Return a tilt angle biased toward edge-on septa.

        Returns:
            A tilt angle in degrees.
        """
        bins = np.array([[0, 15], [15, 30], [30, 45], [45, 60], [60, 90]])
        probabilities = np.array([0.05, 0.10, 0.20, 0.25, 0.40])
        selected = np.random.choice(len(bins), p=probabilities)
        return float(np.random.uniform(bins[selected, 0], bins[selected, 1]))

    @staticmethod
    def sample_septum_rotation_deg():
        """Return a random septum azimuth in the acquisition plane.

        Returns:
            A rotation offset in degrees.
        """
        return float(np.random.uniform(0.0, 180.0))

    def create_cells(self, n_objects):
        """Create the initial population of bacterial cells.

        Args:
            n_objects: Number of cells to generate.

        Returns:
            A dictionary of cell IDs mapped to ``Cell`` instances.
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
        """Create a single bacterial cell at ``coordinates``.

        Args:
            coordinates: ``(row, col)`` location for the new cell.
            progression: Optional initial cell-cycle progression.

        Returns:
            A configured ``Cell`` instance.
        """
        length = np.random.uniform(
            self.cell_size - self.cell_size_std,
            self.cell_size + self.cell_size_std,
        )
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
            np.random.uniform(-math.pi, math.pi),
            self._sample_septum_tilt_deg(),
            self._sample_septum_rotation_deg(),
            p1,
            p2,
            p3,
            0,
        )
        self.calculate_dynamics(cell, rate=progression)
        return cell

    def _sample_septum_tilt_deg(self):
        return _sample_angle_distribution(
            self.septum_tilt_distribution,
            default_sampler=self.sample_septum_tilt_deg,
            low=0.0,
            high=90.0,
        )

    def _sample_septum_rotation_deg(self):
        return _sample_angle_distribution(
            self.septum_rotation_distribution,
            default_sampler=self.sample_septum_rotation_deg,
            low=0.0,
            high=180.0,
        )

    def generate_mask(self):
        """Generate the current fluorescence mask for the sample.

        Returns:
            A ``float32`` central z slice containing the rendered cells.
        """
        volume = self.generate_volume()
        return volume[volume.shape[0] // 2]

    def generate_volume(self):
        """Generate the current 3D fluorescence mask for the sample.

        Returns:
            A ``float32`` volume with shape ``(z, y, x)``.
        """
        mask = np.zeros(self.volume_shape, dtype=np.float32)

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
            septum_phase, sep_completion = calculate_septum_state(cell)

            self._render_cell_into_volume(
                mask,
                cell,
                septum_phase=septum_phase,
                septum_completion=sep_completion,
            )

        return mask

    def _render_cell_into_volume(
        self,
        volume: np.ndarray,
        cell,
        septum_phase: str,
        septum_completion: float,
    ) -> None:
        """Render a single 3D cell into the target volume."""
        z_slice, row_slice, col_slice, local_z, local_row, local_col = (
            _cell_render_roi_3d(
                self.volume_shape,
                self.z_size // 2,
                cell.center_row,
                cell.center_col,
                cell.major_axis,
                cell.minor_axis,
                self.voxel_size,
            )
        )
        local_shape = (
            z_slice.stop - z_slice.start,
            row_slice.stop - row_slice.start,
            col_slice.stop - col_slice.start,
        )
        membrane, cyto, septum = draw_staph_cell_volume(
            local_shape,
            local_z,
            local_row,
            local_col,
            cell.major_axis,
            cell.minor_axis,
            angle_deg=np.rad2deg(cell.orientation),
            septum_tilt_deg=cell.septum_tilt_deg,
            septum_rotation_deg=cell.septum_rotation_deg,
            septum_phase=septum_phase,
            septum_completion=septum_completion,
            xy_to_z_scale=self.voxel_size[0] / self.voxel_size[1],
            value_membrane=1,
            value_cyto=self.cyto_fluor,
            value_septum=2,
        )
        volume[z_slice, row_slice, col_slice] += membrane + cyto + septum

    def _render_cell_into_mask(
        self,
        mask: np.ndarray,
        cell,
        septum_phase: str,
        septum_completion: float,
    ) -> None:
        """Render a single cell into the target mask.

        Args:
            mask: Full-size output mask to update.
            cell: Cell instance to render.
            septum_phase: Septum state for the current cell-cycle phase.
            septum_completion: Fraction of septum completion in the ring phase.
        """
        row_slice, col_slice, local_row, local_col = _cell_render_roi_2d(
            self.sample_size,
            cell.center_row,
            cell.center_col,
            cell.major_axis,
            cell.minor_axis,
        )
        local_shape = (
            row_slice.stop - row_slice.start,
            col_slice.stop - col_slice.start,
        )
        membrane_mask = np.zeros(local_shape, dtype=np.float32)
        cyto_mask = np.zeros(local_shape, dtype=np.float32)
        septum_mask = np.zeros(local_shape, dtype=np.float32)

        membrane_mask, cyto_mask, septum_mask = draw_ellipse_with_axes(
            membrane_mask,
            cyto_mask,
            septum_mask,
            local_row,
            local_col,
            cell.major_axis,
            cell.minor_axis,
            angle_deg=np.rad2deg(cell.orientation),
            septum_tilt_deg=cell.septum_tilt_deg,
            septum_rotation_deg=cell.septum_rotation_deg,
            septum_phase=septum_phase,
            septum_completion=septum_completion,
            value_membrane=1,
            value_cyto=self.cyto_fluor,
            value_septum=2,
        )

        mask[row_slice, col_slice] += membrane_mask + cyto_mask + septum_mask

    def resolve_all_collisions(self, max_iterations=10):
        """Resolve overlaps by pushing cells apart along the center vector.

        Args:
            max_iterations: Maximum number of collision-resolution passes.
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
                    min_safe_distance = (cell_a.major_axis + cell_b.major_axis) / 2 + 1

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
        """Advance cell progression and divide when the cycle completes.

        Args:
            cell_or_id: ``Cell`` instance or integer cell ID.
            rate: Number of progression steps to advance.
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
        """Split one cell into two daughter cells.

        Args:
            cell_id: ID of the parent cell to divide.
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
                min_safe_distance = (daughter.major_axis + neighbor_cell.major_axis) / 2

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
        """Generate non-overlapping cell center coordinates.

        Args:
            shape: Output shape used for sampling.
            spacing: Minimum allowed distance between coordinates.
            num_points: Number of coordinates to generate.

        Returns:
            A list of ``(row, col)`` coordinates.
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
    """Draw the middle slice of a 3D cell into image arrays.

    Returns:
        Three arrays representing the membrane, cytoplasm, and septum.
    """
    local_shape = (membrane.shape[0], membrane.shape[1])
    z_planes = max(9, int(np.ceil(minor_axis_length)) + 2 * ELLIPSE_MORPHOLOGY_PADDING)
    center_z = z_planes // 2
    membrane_volume, cyto_volume, septum_volume = draw_staph_cell_volume(
        (z_planes, local_shape[0], local_shape[1]),
        center_z,
        cy,
        cx,
        major_axis_length,
        minor_axis_length,
        angle_deg=angle_deg,
        septum_tilt_deg=septum_tilt_deg,
        septum_rotation_deg=septum_rotation_deg,
        septum_phase=septum_phase,
        septum_completion=septum_completion,
        value_membrane=value_membrane,
        value_cyto=value_cyto,
        value_septum=value_septum,
    )
    membrane = membrane_volume[center_z]
    cyto = cyto_volume[center_z]
    septum = septum_volume[center_z]
    return membrane, cyto, septum


def draw_staph_cell_volume(
    shape,
    cz,
    cy,
    cx,
    major_axis_length,
    minor_axis_length,
    angle_deg=0,
    septum_tilt_deg=90,
    septum_rotation_deg=None,
    septum_phase="ring",
    septum_completion=1.0,
    xy_to_z_scale=1.0,
    value_membrane=1,
    value_cyto=0.5,
    value_septum=2,
):
    """Voxelize a spheroidal ``S. aureus`` cell and arbitrarily oriented septum.

    The coordinate system is ``(z, y, x)``. ``septum_tilt_deg`` is measured
    from face-on to edge-on relative to the acquisition plane, and
    ``septum_rotation_deg`` is the azimuth of that tilt in the image plane.
    """
    septum_completion = float(np.clip(septum_completion, 0.0, 1.0))
    septum_rotation_deg = 0.0 if septum_rotation_deg is None else septum_rotation_deg
    a = _axis_length_to_semi_axis(major_axis_length)
    b = _axis_length_to_semi_axis(minor_axis_length)
    z_scale = 1.0 / max(float(xy_to_z_scale), np.finfo(float).eps)

    zz, rr, cc = np.indices(shape, dtype=np.float32)
    rel_z = (zz - float(cz)) * z_scale
    rel_row = rr - float(cy)
    rel_col = cc - float(cx)

    angle_rad = np.deg2rad(angle_deg)
    major_coord = rel_col * np.cos(angle_rad) + rel_row * np.sin(angle_rad)
    minor_coord = -rel_col * np.sin(angle_rad) + rel_row * np.cos(angle_rad)
    normalized = (major_coord / a) ** 2 + (minor_coord / b) ** 2 + (rel_z / b) ** 2
    cell_mask = normalized <= 1.0
    inner = normalized <= max(0.0, 1.0 - MEMBRANE_SHELL_THICKNESS) ** 2

    septum_mask = np.zeros(shape, dtype=bool)
    if septum_phase != "none":
        axis_angle_rad = np.deg2rad(angle_deg + 90.0 + septum_rotation_deg)
        tilt_rad = np.deg2rad(np.clip(septum_tilt_deg, 0.0, 90.0))
        normal = np.array(
            [
                np.cos(tilt_rad),
                -np.sin(tilt_rad) * np.cos(axis_angle_rad),
                np.sin(tilt_rad) * np.sin(axis_angle_rad),
            ],
            dtype=np.float32,
        )
        plane_distance = rel_z * normal[0] + rel_row * normal[1] + rel_col * normal[2]
        radial_squared = np.maximum(
            0.0,
            rel_z**2 + rel_row**2 + rel_col**2 - plane_distance**2,
        )
        radial = np.sqrt(radial_squared)
        septum_base = cell_mask & (np.abs(plane_distance) <= SEPTUM_PLANE_HALF_WIDTH)
        if np.any(septum_base):
            outer_radius = max(1.0, float(np.max(radial[septum_base])))
        else:
            outer_radius = max(1.0, b)

        if septum_phase == "closed":
            septum_mask = septum_base
        else:
            min_visible_completion = max(
                SEPTUM_VISIBILITY_EPSILON,
                2.0 / max(2.0 * outer_radius, 1.0),
            )
            visible_completion = max(septum_completion, min_visible_completion)
            inner_radius = outer_radius * (1.0 - visible_completion)
            septum_mask = septum_base & (radial >= inner_radius)

    septum_mask &= cell_mask
    membrane_mask = cell_mask & ~inner & ~septum_mask
    cyto_mask = inner & ~septum_mask

    return (
        membrane_mask.astype(np.float32) * value_membrane,
        cyto_mask.astype(np.float32) * value_cyto,
        septum_mask.astype(np.float32) * value_septum,
    )


def _cell_render_roi_2d(
    sample_size,
    center_row,
    center_col,
    major_axis_length,
    minor_axis_length,
):
    pad = (
        int(
            np.ceil(
                max(
                    _axis_length_to_semi_axis(major_axis_length),
                    _axis_length_to_semi_axis(minor_axis_length),
                )
            )
        )
        + ELLIPSE_MORPHOLOGY_PADDING
    )
    row_min = max(0, int(center_row) - pad)
    row_max = min(sample_size[0], int(center_row) + pad + 1)
    col_min = max(0, int(center_col) - pad)
    col_max = min(sample_size[1], int(center_col) + pad + 1)

    row_slice = slice(row_min, row_max)
    col_slice = slice(col_min, col_max)
    local_row = center_row - row_min
    local_col = center_col - col_min
    return row_slice, col_slice, local_row, local_col


def _cell_render_roi_3d(
    volume_shape,
    center_z,
    center_row,
    center_col,
    major_axis_length,
    minor_axis_length,
    voxel_size,
):
    xy_to_z_scale = float(voxel_size[0]) / float(voxel_size[1])
    xy_pad = (
        int(
            np.ceil(
                max(
                    _axis_length_to_semi_axis(major_axis_length),
                    _axis_length_to_semi_axis(minor_axis_length),
                )
            )
        )
        + ELLIPSE_MORPHOLOGY_PADDING
    )
    z_pad = (
        int(np.ceil(_axis_length_to_semi_axis(minor_axis_length) * xy_to_z_scale))
        + ELLIPSE_MORPHOLOGY_PADDING
    )

    z_min = max(0, int(center_z) - z_pad)
    z_max = min(volume_shape[0], int(center_z) + z_pad + 1)
    row_min = max(0, int(center_row) - xy_pad)
    row_max = min(volume_shape[1], int(center_row) + xy_pad + 1)
    col_min = max(0, int(center_col) - xy_pad)
    col_max = min(volume_shape[2], int(center_col) + xy_pad + 1)

    z_slice = slice(z_min, z_max)
    row_slice = slice(row_min, row_max)
    col_slice = slice(col_min, col_max)
    local_z = center_z - z_min
    local_row = center_row - row_min
    local_col = center_col - col_min
    return z_slice, row_slice, col_slice, local_z, local_row, local_col


def _sample_angle_distribution(distribution, default_sampler, low, high):
    if callable(distribution):
        return float(np.clip(distribution(), low, high))

    if isinstance(distribution, tuple):
        return float(np.random.uniform(distribution[0], distribution[1]))

    if distribution == "biased_high":
        return float(default_sampler())
    if distribution == "uniform":
        return float(np.random.uniform(low, high))
    if distribution == "perpendicular":
        return float(high)
    if distribution == "parallel":
        return float(low)

    raise ValueError(
        "Angle distribution must be a callable, a (low, high) tuple, or one of "
        "'biased_high', 'uniform', 'perpendicular', 'parallel'."
    )


def calculate_septum_state(cell):
    """Return the septum rendering state for the current cell progression."""
    progression = int(np.clip(cell.progression, 0, 100))
    p1_end = max(0, int(cell.p1))
    p2_duration = max(1, int(cell.p2))
    p2_end = p1_end + p2_duration

    if progression < p1_end:
        return "none", 0.0
    if progression < p2_end:
        completion = (progression - p1_end) / p2_duration
        return "ring", float(np.clip(completion, 0.0, 1.0))
    if progression < 100:
        return "closed", 1.0
    return "none", 0.0


def _axis_length_to_semi_axis(axis_length):
    """Convert a full axis length in pixels to the semi-axis used by skimage."""
    return max(1.0, float(axis_length) / 2.0)


@dataclass
class Cell:
    """State for a single simulated bacterial cell."""

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
