# Usage

## Quick Start

```python
from maicroscopy_sandbox import mAIcroscopySandbox
from maicroscopy_sandbox.samples.staph import StaphMembrane

microscope = mAIcroscopySandbox(fov_size=[500, 500], laser_intensity=1000)
sample = StaphMembrane(sample_size=microscope.stage_size, n_objects=5, pixel_size=30)

microscope.set_laser_power(100)
frame = microscope.load_sample(sample, acquire=True)
```

## Common Patterns

### Acquire a Single Frame

```python
frame = microscope.acquire_image()
```

### Move the Stage

```python
new_frame = microscope.move_stage([100, 50], acquire=True)
```

### Adjust Optical Parameters

```python
microscope.set_readout_noise(50)
microscope.set_ADC_offset(100)
microscope.set_gaussian_sigma(2.0)
```

## Sample Types

- `StaphMembrane` for membrane and septum dynamics
- `Ellipsoid` for simple moving objects
- `BinarySample` for loading masks from TIFF files

## Tips

- Use smaller `fov_size` and `n_objects` for faster runs.
- Keep plotting outside the acquisition loop when benchmarking.
- Use `t.py` as a performance smoke test after changes.

