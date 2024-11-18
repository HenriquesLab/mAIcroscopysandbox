import math
import numpy as np
from dataclasses import dataclass
from skimage.draw import ellipse
from skimage.morphology import binary_erosion

from .sample import Sample


class Ellipsoid(object):
    def __init__(
        self,
        sample_size: np.array = [1000, 1000],
        bleaching_rate: float = 0.001,
        movement_rate: float = 10.0,
        movement_probability: float = 0.1,
        rotation: int = math.pi*0.1,
        rotation_probability: float = 0.1,
        axis_deformation_rate: float = 0.2,
        axis_deformation_probability: float = 0.1,
        n_objects: int = 1,
        cell_size: int = 100,
        cell_size_std: float = 5,
        mode: str = "Full",
    ):
        """
        Parameters
        ----------
        sample_size : tuple, optional
            The size of the sample as a tuple of two integers. The default is (1000, 1000).
        bleaching_rate : float, optional
            The rate of bleaching per frame as a float. The default is 0.001.
        movement_rate : float, optional
            The rate of movement per frame as a float. The default is 10.0.
        movement_probability : float, optional
            The probability of movement per frame as a float. The default is 0.1.
        rotation : int, optional
            The rotation of the ellipsoid in degrees as an integer. The default is 0.
        n_ojects : int, optional
            The number of objects in the sample as an integer. The default is 1.
        mode : str, optional
            The mode of the sample, either "Full" or "Edges". The default is "Full".
        """
        self.sample_size = sample_size
        self.movement_probability = movement_probability
        self.bleaching_rate = bleaching_rate
        self.movement_rate = movement_rate
        self.movement_probability = movement_probability
        self.rotation = rotation
        self.rotation_probability = rotation_probability
        self.axis_deformation_rate = axis_deformation_rate
        self.axis_deformation_probability = axis_deformation_probability
        self.cell_size = cell_size
        self.cell_size_std = cell_size_std
        self.mode = mode
        self.cells = self.create_cells(n_objects)

    def create_cells(self, n_objects):
        cells = []
        coordinates = self.generate_random_coordinates(
            (self.sample_size[0], self.sample_size[1]), self.cell_size * 2, n_objects
        )
        for i in range(n_objects):
            cells.append(
                Cell(
                    coordinates[i][0],
                    coordinates[i][1],
                    np.random.normal(self.cell_size, self.cell_size_std),
                    np.random.normal(self.cell_size // 2, self.cell_size_std),
                    np.random.randint(-math.pi, math.pi),
                )
            )

        return cells

    def generate_mask(self):
        self.calculate_dynamics()
        mask = np.zeros(self.sample_size).astype(np.float32)
        for cell in self.cells:
            rr, cc = ellipse(
                cell.center_row,
                cell.center_col,
                cell.major_axis,
                cell.minor_axis,
                shape=(self.sample_size[0], self.sample_size[1]),
                rotation=cell.orientation,
            )
            mask[rr, cc] = 1
        if self.mode == "Edges":
            eroded = binary_erosion(mask)
            for i in range(2):
                eroded = binary_erosion(eroded)
            return mask - eroded
        else:
            return mask

    def calculate_dynamics(self):
        for cell in self.cells:
            if np.random.rand() < self.movement_probability:
                cell.center_row += np.random.normal(0, self.movement_rate)
                cell.center_col += np.random.normal(0, self.movement_rate)
            if np.random.rand() < self.rotation_probability:
                cell.orientation += np.random.normal(0, self.rotation)
            if np.random.rand() < self.axis_deformation_probability:
                cell.major_axis -= np.random.normal(0, self.axis_deformation_rate)
                cell.minor_axis += np.random.normal(0, self.axis_deformation_rate)

    def generate_random_coordinates(self, shape, spacing, num_points):
        coordinates = [(self.sample_size[0] // 2, self.sample_size[1] // 2)]
        while len(coordinates) < num_points:
            # Generate a random point
            point = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))

            # Check if the point satisfies the spacing constraint
            if all(
                np.linalg.norm(np.array(point) - np.array(coord)) >= spacing
                for coord in coordinates
            ):
                coordinates.append(point)

        return coordinates


@dataclass
class Cell:
    center_row: int
    center_col: int
    major_axis: int
    minor_axis: int
    orientation: int
