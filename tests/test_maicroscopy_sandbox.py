#!/usr/bin/env python

"""Tests for `maicroscopy_sandbox` package."""

import os
import pathlib
import subprocess
import tempfile
import types

import numpy as np
import warnings
from tifffile import imwrite

from maicroscopy_sandbox import maicroscopy_sandbox as microscope_module
from maicroscopy_sandbox.fluorescence_sim import (
    FromLoc2Image_MultiThreaded,
    binary2locs,
    generate_image,
)
from maicroscopy_sandbox.maicroscopy_sandbox import mAIcroscopySandbox
from maicroscopy_sandbox.samples.binary import BinarySample
from maicroscopy_sandbox.samples.ellipsoid import Ellipsoid
from maicroscopy_sandbox.samples.sample import Sample
from maicroscopy_sandbox.samples.staph import (
    StaphMembrane,
    draw_ellipse_with_axes,
)


def test_septum_tilt_sampling_is_biased_toward_high_angles():
    rng = np.random.default_rng(1234)
    original_choice = np.random.choice
    original_uniform = np.random.uniform

    np.random.choice = rng.choice
    np.random.uniform = rng.uniform
    try:
        tilts = np.array(
            [StaphMembrane.sample_septum_tilt_deg() for _ in range(20000)]
        )
    finally:
        np.random.choice = original_choice
        np.random.uniform = original_uniform

    low_fraction = np.mean((tilts >= 0) & (tilts < 15))
    high_fraction = np.mean((tilts >= 60) & (tilts <= 90))

    assert 0.03 <= low_fraction <= 0.07
    assert 0.36 <= high_fraction <= 0.44


def test_phase_2_ring_closes_until_phase_3():
    sample_shape = (64, 64)

    membrane = np.zeros(sample_shape, dtype=np.float32)
    cyto = np.zeros(sample_shape, dtype=np.float32)
    early_ring = np.zeros(sample_shape, dtype=np.float32)
    _, _, early_ring = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        early_ring,
        32,
        32,
        12,
        10,
        angle_deg=0,
        septum_tilt_deg=30,
        septum_rotation_deg=0,
        septum_phase="ring",
        septum_completion=0.1,
    )

    late_ring = np.zeros(sample_shape, dtype=np.float32)
    _, _, late_ring = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        late_ring,
        32,
        32,
        12,
        10,
        angle_deg=0,
        septum_tilt_deg=30,
        septum_rotation_deg=0,
        septum_phase="ring",
        septum_completion=0.9,
    )

    septum_closed = np.zeros(sample_shape, dtype=np.float32)
    _, _, septum_closed = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        septum_closed,
        32,
        32,
        12,
        10,
        angle_deg=0,
        septum_tilt_deg=30,
        septum_rotation_deg=0,
        septum_phase="closed",
        septum_completion=1.0,
    )

    early_rows, early_cols = np.where(early_ring > 0)
    late_rows, late_cols = np.where(late_ring > 0)
    closed_rows, closed_cols = np.where(septum_closed > 0)

    early_width = early_cols.max() - early_cols.min()
    early_height = early_rows.max() - early_rows.min()
    late_width = late_cols.max() - late_cols.min()
    late_height = late_rows.max() - late_rows.min()
    closed_width = closed_cols.max() - closed_cols.min()
    closed_height = closed_rows.max() - closed_rows.min()

    assert early_ring.sum() > 0
    assert late_ring.sum() > early_ring.sum()
    assert septum_closed.sum() >= late_ring.sum()
    assert septum_closed.sum() > 0
    assert closed_width >= early_width
    assert closed_width >= late_width
    assert closed_height >= early_height
    assert closed_height >= late_height
    assert early_ring[32, 32] == 0
    assert late_ring[32, 32] == 0
    assert septum_closed[32, 32] > 0
    assert closed_width > 0
    assert closed_height > 0


def test_septum_orientation_varies_in_plane_and_reaches_membrane():
    sample_shape = (64, 64)
    membrane = np.zeros(sample_shape, dtype=np.float32)
    cyto = np.zeros(sample_shape, dtype=np.float32)

    septum_a = np.zeros(sample_shape, dtype=np.float32)
    _, _, septum_a = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        septum_a,
        32,
        32,
        12,
        10,
        angle_deg=0,
        septum_tilt_deg=35,
        septum_rotation_deg=0,
        septum_phase="closed",
        septum_completion=1.0,
    )

    septum_b = np.zeros(sample_shape, dtype=np.float32)
    _, _, septum_b = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        septum_b,
        32,
        32,
        12,
        10,
        angle_deg=0,
        septum_tilt_deg=35,
        septum_rotation_deg=90,
        septum_phase="closed",
        septum_completion=1.0,
    )

    rows_a, cols_a = np.where(septum_a > 0)
    rows_b, cols_b = np.where(septum_b > 0)

    height_a = rows_a.max() - rows_a.min()
    width_a = cols_a.max() - cols_a.min()
    height_b = rows_b.max() - rows_b.min()
    width_b = cols_b.max() - cols_b.min()

    assert septum_a.sum() > 0
    assert septum_b.sum() > 0
    assert septum_a.sum() != septum_b.sum() or not np.array_equal(
        septum_a, septum_b
    )
    assert width_a != width_b or height_a != height_b
    assert max(width_a, height_a) >= 14


def test_near_perpendicular_septum_keeps_visible_thickness():
    sample_shape = (64, 64)
    membrane = np.zeros(sample_shape, dtype=np.float32)
    cyto = np.zeros(sample_shape, dtype=np.float32)
    septum = np.zeros(sample_shape, dtype=np.float32)

    _, _, septum = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        septum,
        32,
        32,
        12,
        10,
        angle_deg=0,
        septum_tilt_deg=85,
        septum_rotation_deg=45,
        septum_phase="closed",
        septum_completion=1.0,
    )

    rows, cols = np.where(septum > 0)
    height = rows.max() - rows.min()
    width = cols.max() - cols.min()

    assert septum.sum() > 0
    assert min(height, width) >= 2


def test_phase_1_cell_major_axis_matches_pixel_size_scaling():
    np.random.seed(11)
    pixel_size = 50
    sample = StaphMembrane(
        sample_size=[128, 128],
        n_objects=1,
        pixel_size=pixel_size,
        cell_size_std=0,
        progression_rate=0,
        axis_ratio=1.0,
    )
    cell = next(iter(sample.cells.values()))
    cell.progression = 0
    cell.p1 = 100
    cell.p2 = 0
    cell.p3 = 0
    cell.major_axis = sample.cell_size
    cell.minor_axis = sample.cell_size
    cell.orientation = 0
    cell.center_row = 64
    cell.center_col = 64

    mask = sample.generate_mask()
    _, cols = np.where(mask > 0)
    rendered_major_axis = cols.max() - cols.min() + 1
    expected_major_axis = 1000 / pixel_size

    assert abs(rendered_major_axis - expected_major_axis) <= 2


def test_package_import_works_without_optional_sr_feature():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            "conda",
            "run",
            "-n",
            "aiobio",
            "python",
            "-c",
            (
                "import maicroscopy_sandbox; "
                "print(maicroscopy_sandbox.mAIcroscopySandbox.__name__)"
            ),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": "src",
            "MPLCONFIGDIR": "/tmp/mpl",
            "XDG_CACHE_HOME": "/tmp/xdg",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "mAIcroscopySandbox" in completed.stdout


def test_package_has_no_sr_exports():
    import maicroscopy_sandbox

    assert not any(name.lower().endswith("srrf") for name in dir(maicroscopy_sandbox))


def test_sample_base_generates_reproducible_mask():
    sample = Sample(sample_size=[16, 16])
    mask = sample.generate_mask()

    assert mask.shape == (16, 16)
    assert mask.dtype == np.bool_
    assert mask.any()


def test_binary_sample_reads_mask_from_tiff():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = pathlib.Path(tmpdir) / "binary_sample.tif"
        img = np.zeros((8, 8), dtype=np.uint8)
        img[2:5, 3:6] = 1
        imwrite(img_path, img)

        sample = BinarySample(str(img_path), sample_size=[8, 8])
        mask = sample.generate_mask()

    assert mask.shape == (8, 8)
    assert mask.dtype == np.bool_
    assert mask.sum() == 9


def test_ellipsoid_generates_full_and_edge_masks():
    np.random.seed(42)
    full = Ellipsoid(
        sample_size=[64, 64],
        n_objects=2,
        cell_size=10,
        cell_size_std=0,
        movement_probability=0,
        rotation_probability=0,
        axis_deformation_probability=0,
        mode="Full",
    )
    edges = Ellipsoid(
        sample_size=[64, 64],
        n_objects=2,
        cell_size=10,
        cell_size_std=0,
        movement_probability=0,
        rotation_probability=0,
        axis_deformation_probability=0,
        mode="Edges",
    )

    full_mask = full.generate_mask()
    edge_mask = edges.generate_mask()
    coords = edges.generate_random_coordinates((64, 64), 8, 3)

    assert full_mask.shape == (64, 64)
    assert edge_mask.shape == (64, 64)
    assert full_mask.sum() > edge_mask.sum() > 0
    assert len(coords) == 3


def test_ellipsoid_dynamics_update_cells():
    np.random.seed(7)
    sample = Ellipsoid(
        sample_size=[64, 64],
        n_objects=1,
        cell_size=10,
        cell_size_std=0,
        movement_probability=1,
        rotation_probability=1,
        axis_deformation_probability=1,
        movement_rate=1,
        rotation=0.5,
        axis_deformation_rate=0.1,
    )
    cell = sample.cells[0]
    initial = (
        cell.center_row,
        cell.center_col,
        cell.orientation,
        cell.major_axis,
        cell.minor_axis,
    )

    sample.calculate_dynamics()

    updated = (
        cell.center_row,
        cell.center_col,
        cell.orientation,
        cell.major_axis,
        cell.minor_axis,
    )
    assert updated != initial


def test_generate_image_and_binary2locs_smoke():
    np.random.seed(3)
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:8, 5:9] = 1
    bleaching = np.ones((16, 16), dtype=np.float32)

    image = generate_image(
        mask,
        bleaching,
        laser_intensity=100,
        gaussian_sigma=0.5,
        readout_noise=0,
        ADC_offset=0,
    )
    locs = binary2locs(mask, density=0.5)

    assert image.shape == (16, 16)
    assert image.dtype == np.float64
    assert image.max() > 0
    assert len(locs) == 2
    assert len(locs[0]) == len(locs[1]) == 8


def test_microscope_setters_and_stage_bounds():
    scope = mAIcroscopySandbox(stage_size=[20, 20], fov_size=[10, 10])
    scope.sample = types.SimpleNamespace(bleaching_rate=0.0, generate_mask=lambda: np.ones((20, 20), dtype=np.float32))

    scope.set_wavelenght(488)
    scope.set_wavelenght_std(20)
    scope.set_NA(1.4)
    scope.set_sigma(0.4)
    scope.set_sigma_std(0.2)
    scope.set_ADC_per_photon_conversion(1.5)
    scope.set_ADC_offset(42)
    scope.set_readout_noise(3)
    scope.set_gaussian_sigma(0.5)
    scope.set_laser_power(120)
    assert scope.laser_power == 100.0
    scope.set_laser_power(-5)
    assert scope.laser_power == 0

    scope.current_position = [18, 18]
    movement = [5, 5]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        corrected = scope._check_move_stage(movement)

    assert corrected[0] == 2
    assert corrected[1] == 2
    assert len(caught) == 2


def _load_head_fluorescence_module():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    source = subprocess.check_output(
        ["git", "show", "HEAD:src/maicroscopy_sandbox/fluorescence_sim.py"],
        cwd=repo_root,
        text=True,
    )
    module = types.ModuleType("head_fluorescence_sim")
    module.__dict__["__file__"] = (
        "HEAD:src/maicroscopy_sandbox/fluorescence_sim.py"
    )
    exec(compile(source, module.__dict__["__file__"], "exec"), module.__dict__)
    return module


def test_acquisition_updates_bleaching_and_returns_frame(monkeypatch):
    sample = StaphMembrane(
        sample_size=[256, 256],
        bleaching_rate=0.05,
        n_objects=8,
        pixel_size=64,
        progression_rate=5,
        cyto_fluor=0.5,
        axis_ratio=1.6,
    )

    microscope = mAIcroscopySandbox(
        stage_size=[256, 256],
        fov_size=[128, 128],
        laser_intensity=1000,
        gaussian_sigma=1,
        random_seed=99,
    )
    microscope.set_readout_noise(200)
    microscope.set_laser_power(70)
    microscope.set_sigma(1)
    microscope.set_sigma_std(0.1)
    microscope.set_ADC_offset(100)
    microscope.load_sample(sample, acquire=False)

    initial_bleaching = microscope.bleaching.copy()

    def deterministic_generate_image(mask, bleaching, **kwargs):
        return (mask * 10 + bleaching).astype(np.float32)

    monkeypatch.setattr(
        microscope_module, "generate_image", deterministic_generate_image
    )

    frame = microscope.acquire_image()

    assert frame.shape == (128, 128)
    assert frame.dtype == np.int16
    assert np.any(microscope.bleaching < initial_bleaching)
    assert np.all(microscope.bleaching >= 0)


def test_refactored_fluorescence_kernel_matches_head_implementation():
    head_fluorescence = _load_head_fluorescence_module()

    xc_array = np.array([5, 17, 30, 45], dtype=np.int64)
    yc_array = np.array([6, 20, 31, 48], dtype=np.int64)
    photon_array = np.array([1000.0, 500.0, 750.0, 300.0], dtype=np.float64)
    sigma_array = np.array([1.1, 0.8, 1.6, 1.2], dtype=np.float64)
    mask = np.ones((64, 64), dtype=np.float64)
    bleaching = np.linspace(0.5, 1.0, 64 * 64, dtype=np.float64).reshape(64, 64)

    baseline = head_fluorescence.FromLoc2Image_MultiThreaded(
        xc_array,
        yc_array,
        photon_array,
        sigma_array,
        64,
        64,
        1,
        bleaching,
    )
    current = FromLoc2Image_MultiThreaded(
        xc_array,
        yc_array,
        photon_array,
        sigma_array,
        64,
        64,
        1,
        mask,
        bleaching,
    )

    np.testing.assert_allclose(current, baseline, rtol=0, atol=1e-12)


def test_mask_gated_fluorescence_kernel_matches_head_masked_result():
    head_fluorescence = _load_head_fluorescence_module()

    xc_array = np.array([10, 18, 33], dtype=np.int64)
    yc_array = np.array([12, 25, 40], dtype=np.int64)
    photon_array = np.array([1200.0, 800.0, 650.0], dtype=np.float64)
    sigma_array = np.array([1.3, 0.9, 1.7], dtype=np.float64)
    mask = np.zeros((64, 64), dtype=np.float64)
    mask[8:20, 8:20] = 1.0
    mask[20:35, 15:30] = 0.5
    mask[36:48, 28:45] = 1.0
    bleaching = np.linspace(0.7, 1.1, 64 * 64, dtype=np.float64).reshape(64, 64)

    baseline = head_fluorescence.FromLoc2Image_MultiThreaded(
        xc_array,
        yc_array,
        photon_array,
        sigma_array,
        64,
        64,
        1,
        bleaching,
    ) * mask
    current = (
        FromLoc2Image_MultiThreaded(
            xc_array,
            yc_array,
            photon_array,
            sigma_array,
            64,
            64,
            1,
            mask,
            bleaching,
        )
        * mask
    )

    np.testing.assert_allclose(current, baseline, rtol=0, atol=1e-12)


def test_septum_aligns_with_cell_minor_axis():
    sample_shape = (96, 96)
    membrane = np.zeros(sample_shape, dtype=np.float32)
    cyto = np.zeros(sample_shape, dtype=np.float32)

    vertical_septum = np.zeros(sample_shape, dtype=np.float32)
    _, _, vertical_septum = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        vertical_septum,
        48,
        48,
        18,
        10,
        angle_deg=0,
        septum_tilt_deg=85,
        septum_rotation_deg=0,
        septum_phase="closed",
        septum_completion=1.0,
    )

    horizontal_septum = np.zeros(sample_shape, dtype=np.float32)
    _, _, horizontal_septum = draw_ellipse_with_axes(
        membrane.copy(),
        cyto.copy(),
        horizontal_septum,
        48,
        48,
        18,
        10,
        angle_deg=90,
        septum_tilt_deg=85,
        septum_rotation_deg=0,
        septum_phase="closed",
        septum_completion=1.0,
    )

    vertical_rows, vertical_cols = np.where(vertical_septum > 0)
    horizontal_rows, horizontal_cols = np.where(horizontal_septum > 0)

    vertical_height = vertical_rows.max() - vertical_rows.min()
    vertical_width = vertical_cols.max() - vertical_cols.min()
    horizontal_height = horizontal_rows.max() - horizontal_rows.min()
    horizontal_width = horizontal_cols.max() - horizontal_cols.min()

    assert vertical_height > vertical_width
    assert horizontal_width > horizontal_height
