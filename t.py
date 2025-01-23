import math
from matplotlib import pyplot as plt
from maicroscopy_sandbox import mAIcroscopySandbox
from maicroscopy_sandbox.samples.ellipsoid import Ellipsoid
from maicroscopy_sandbox.smartSRRF import smartSRRF

microscope = mAIcroscopySandbox(fov_size=[500, 500])
sample = Ellipsoid(
    sample_size=microscope.stage_size,
    bleaching_rate=0.05,
    n_objects=20,
    movement_rate=5,
    movement_probability=0.0,
    axis_deformation_probability=0.0,
    axis_deformation_rate=0.1,
    rotation=math.pi * 0.5,
    rotation_probability=0.0,
    mode="Edges",
)
microscope.set_laser_power(100)
frame = microscope.load_sample(sample, acquire=True)

from maicroscopy_sandbox import smartSRRF

output = smartSRRF(microscope, plot=True)
