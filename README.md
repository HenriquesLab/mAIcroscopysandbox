# mAIcroscopy Sandbox

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python package for simulating realistic fluorescence microscopy experiments with biological samples. Perfect for testing super-resolution algorithms, training AI models, and teaching microscopy concepts.

## Features

- 🔬 **Realistic Microscopy Simulation**: Accurate modeling of photon noise, bleaching, and detector characteristics
- 🦠 **Bacterial Cell Dynamics**: Simulate *Staphylococcus*-like cells with growth, septum formation, and division
- 📸 **Stage Control**: Move and scan across large samples with precise positioning
- 🔆 **Laser Control**: Adjustable laser power with realistic photobleaching effects
- 🎯 **Super-Resolution Ready**: Built-in smartSRRF implementation for super-resolution imaging
- 🧪 **Extensible**: Easy to create custom sample types and imaging modalities

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Basic Microscopy Simulation](#basic-microscopy-simulation)
  - [Bacterial Cell Simulation](#bacterial-cell-simulation)
  - [Time-Lapse Imaging](#time-lapse-imaging)
  - [Super-Resolution Imaging](#super-resolution-imaging)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements

- Python 3.10 or higher
- pip package manager

### Install from source

1. Clone the repository:
```bash
git clone https://github.com/HenriquesLab/mAIcroscopysandbox.git
cd mAIcroscopysandbox
```

2. Install the package:
```bash
pip install -e .
```

3. For development (includes testing and linting tools):
```bash
pip install -e ".[dev,test]"
```

### Dependencies

The package automatically installs the following dependencies:
- `numpy<2` - Numerical computing
- `scikit-image` - Image processing
- `numba` - JIT compilation for performance
- `nanopyx` - Advanced microscopy algorithms
- `matplotlib` - Visualization (for examples)

## Quick Start

Here's a minimal example to get you started:

```python
import numpy as np
from matplotlib import pyplot as plt
from maicroscopy_sandbox import mAIcroscopySandbox
from maicroscopy_sandbox.samples.staph import StaphMembrane

# Create a virtual microscope
microscope = mAIcroscopySandbox(
    fov_size=[500, 500],  # Field of view in pixels
    laser_intensity=1000   # Photons per pixel
)

# Create a bacterial cell sample
sample = StaphMembrane(
    sample_size=microscope.stage_size,
    n_objects=5,           # Number of cells
    pixel_size=30          # nm per pixel
)

# Load sample and acquire image
microscope.set_laser_power(100)
frame = microscope.load_sample(sample, acquire=True)

# Display the image
plt.imshow(frame, cmap="gray")
plt.title("Simulated Fluorescence Microscopy")
plt.colorbar(label="Intensity (ADU)")
plt.show()
```

## Usage Examples

### Basic Microscopy Simulation

Control the microscope stage and laser settings:

```python
from maicroscopy_sandbox import mAIcroscopySandbox
from maicroscopy_sandbox.samples.ellipsoid import Ellipsoid

# Initialize microscope
microscope = mAIcroscopySandbox(
    stage_size=[5000, 5000],
    fov_size=[300, 300],
    laser_intensity=100000
)

# Configure detector parameters
microscope.set_readout_noise(50)      # Camera noise
microscope.set_ADC_offset(100)        # Baseline offset
microscope.set_gaussian_sigma(2.0)    # PSF blur

# Create and load a simple ellipsoid sample
sample = Ellipsoid(
    sample_size=microscope.stage_size,
    n_objects=10
)

# Acquire image
microscope.set_laser_power(50)  # 50% laser power
frame = microscope.load_sample(sample, acquire=True)

# Move stage and acquire another image
new_frame = microscope.move_stage([100, 50], acquire=True)
```

### Bacterial Cell Simulation

Simulate realistic bacterial cell growth and division:

```python
from maicroscopy_sandbox import mAIcroscopySandbox
from maicroscopy_sandbox.samples.staph import StaphMembrane
import matplotlib.pyplot as plt

# Create microscope
microscope = mAIcroscopySandbox(fov_size=[500, 500], laser_intensity=1000)

# Create bacterial sample with specific parameters
sample = StaphMembrane(
    sample_size=microscope.stage_size,
    n_objects=3,              # Initial number of cells
    pixel_size=30,            # 30 nm/pixel
    bleaching_rate=0.05,      # 5% bleaching per frame
    progression_rate=10,      # Cell cycle progression speed
    p1_rate=42,              # Growth phase percentage
    p2_rate=29               # Septum formation percentage
)

# Load sample
microscope.set_laser_power(100)
frame = microscope.load_sample(sample, acquire=True)

# Visualize
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.imshow(frame, cmap="gray")
ax.set_title("Bacterial Membrane Simulation")
ax.axis('off')
plt.tight_layout()
plt.show()

# Access cell information
print(f"Number of cells: {len(sample.cells)}")
for cell_id, cell in sample.cells.items():
    print(f"Cell {cell_id}: progression={cell.progression}%")
```

### Time-Lapse Imaging

Capture a time series showing cell dynamics:

```python
from maicroscopy_sandbox import mAIcroscopySandbox
from maicroscopy_sandbox.samples.staph import StaphMembrane
import matplotlib.pyplot as plt

# Setup
microscope = mAIcroscopySandbox(fov_size=[500, 500], laser_intensity=1000)
sample = StaphMembrane(
    sample_size=microscope.stage_size,
    n_objects=1,
    pixel_size=30,
    progression_rate=20  # Faster progression for demo
)

microscope.set_laser_power(100)
microscope.load_sample(sample)

# Acquire time series
n_frames = 10
frames = []

for i in range(n_frames):
    frame = microscope.acquire_image()
    frames.append(frame)
    print(f"Frame {i+1}/{n_frames}: {len(sample.cells)} cells")

# Display montage
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (ax, frame) in enumerate(zip(axes.flat, frames)):
    ax.imshow(frame, cmap="gray")
    ax.set_title(f"t={i}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### Super-Resolution Imaging

Apply super-resolution reconstruction using smartSRRF:

```python
from maicroscopy_sandbox import mAIcroscopySandbox, smartSRRF
from maicroscopy_sandbox.samples.staph import StaphMembrane

# Create microscope and sample
microscope = mAIcroscopySandbox(fov_size=[500, 500], laser_intensity=1000)
sample = StaphMembrane(
    sample_size=microscope.stage_size,
    n_objects=5,
    pixel_size=30
)

microscope.set_laser_power(100)
microscope.load_sample(sample)

# Perform super-resolution reconstruction
sr_image = smartSRRF(microscope, plot=True)

# The smartSRRF function will:
# 1. Acquire multiple frames
# 2. Perform SRRF reconstruction
# 3. Display comparison plot
```

## API Reference

### mAIcroscopySandbox

Main microscope simulation class.

#### Constructor Parameters
- `stage_size` (array): Stage size in pixels [height, width]. Default: [5000, 5000]
- `fov_size` (array): Field of view in pixels [height, width]. Default: [300, 300]
- `laser_intensity` (float): Maximum laser intensity in photons/pixel. Default: 100000
- `pixel_size` (float): Physical pixel size in nanometers. Default: 100

#### Key Methods
- `load_sample(sample, acquire=False)`: Load a sample onto the stage
- `acquire_image()`: Capture an image at the current position
- `move_stage(movement, acquire=False)`: Move stage by offset [row, col]
- `set_laser_power(power)`: Set laser power (0-100%)
- `set_readout_noise(noise)`: Set camera readout noise
- `set_gaussian_sigma(sigma)`: Set PSF blur amount

### StaphMembrane

Bacterial cell sample with growth and division.

#### Constructor Parameters
- `sample_size` (array): Sample area in pixels. Default: [1000, 1000]
- `n_objects` (int): Initial number of cells. Default: 1
- `pixel_size` (int): Pixel size in nanometers. Default: 100
- `bleaching_rate` (float): Bleaching rate per frame. Default: 0.001
- `progression_rate` (int): Cell cycle progression speed. Default: 2
- `p1_rate` (int): Growth phase percentage. Default: 42
- `p2_rate` (int): Septum formation percentage. Default: 29

#### Cell Attributes
Each cell in `sample.cells` has:
- `center_row`, `center_col`: Position
- `major_axis`, `minor_axis`: Size
- `orientation`: Angle in radians
- `progression`: Cell cycle progress (0-100%)
- `p1`, `p2`, `p3`: Phase boundaries

## Advanced Usage

### Creating Custom Samples

Extend the `Sample` base class to create custom samples:

```python
from maicroscopy_sandbox.samples.sample import Sample
import numpy as np

class CustomSample(Sample):
    def __init__(self, sample_size=[1000, 1000], **kwargs):
        super().__init__(sample_size=sample_size, **kwargs)
        # Your initialization code
        
    def generate_mask(self):
        # Return 2D array with fluorophore distribution
        mask = np.zeros(self.sample_size, dtype=np.float32)
        # ... your custom mask generation ...
        return mask
    
    def calculate_dynamics(self):
        # Update sample state between frames
        pass
```

### Adjusting Optical Parameters

Fine-tune the microscope optics:

```python
microscope.set_wavelenght(488)      # Excitation wavelength (nm)
microscope.set_NA(1.4)              # Numerical aperture
microscope.set_sigma(0.21)          # Fluorophore emission spread
microscope.set_ADC_per_photon_conversion(1.5)  # Detector gain
```

### Photobleaching Control

Monitor and control photobleaching:

```python
# Create sample with high bleaching
sample = StaphMembrane(
    sample_size=microscope.stage_size,
    bleaching_rate=0.1  # 10% per frame
)

microscope.load_sample(sample)

# Reduce laser power to minimize bleaching
microscope.set_laser_power(30)  # 30% power

# Acquire multiple frames
for i in range(10):
    frame = microscope.acquire_image()
    # Bleaching is automatically applied
```

### Saving Time-Lapse Data

Export image sequences:

```python
import tifffile
import numpy as np

frames = []
for i in range(100):
    frame = microscope.acquire_image()
    frames.append(frame)

# Save as multi-page TIFF
stack = np.stack(frames, axis=0)
tifffile.imwrite("timelapse.tif", stack)
```

## Project Structure

```
mAIcroscopysandbox/
├── src/
│   └── maicroscopy_sandbox/
│       ├── __init__.py
│       ├── maicroscopy_sandbox.py    # Main microscope class
│       ├── fluorescence_sim.py       # Image generation
│       ├── smartSRRF.py             # Super-resolution
│       └── samples/
│           ├── __init__.py
│           ├── sample.py            # Base sample class
│           ├── staph.py             # Bacterial cells
│           ├── ellipsoid.py         # Simple ellipsoids
│           └── binary.py            # Binary structures
├── notebooks/
│   ├── example_usage.ipynb          # Usage examples
│   └── example_from_binary.ipynb    # Binary sample demo
├── tests/
│   └── test_maicroscopy_sandbox.py
├── README.md
├── setup.cfg
├── pyproject.toml
└── LICENSE.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone the repo
git clone https://github.com/HenriquesLab/mAIcroscopysandbox.git
cd mAIcroscopysandbox

# Install in development mode with all extras
pip install -e ".[dev,test]"

# Run tests
pytest

# Run linter
ruff check src/

# Format code
ruff format src/
```

## Citation

If you use this software in your research, please cite:

```bibtex
@software{maicroscopy_sandbox,
  author = {Saraiva, Bruno},
  title = {mAIcroscopy Sandbox: Realistic Fluorescence Microscopy Simulation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/HenriquesLab/mAIcroscopysandbox}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## Acknowledgments

- Developed at the Henriques Lab
- Built on scikit-image, numpy, and nanopyx
- Inspired by real microscopy workflows and challenges

## Contact

- **Author**: Bruno Saraiva
- **Email**: bruno.msaraiva2@gmail.com
- **Lab**: [Henriques Lab](https://henriqueslab.github.io/)

## Troubleshooting

### Common Issues

**ImportError: No module named 'maicroscopy_sandbox'**
- Make sure you installed the package: `pip install -e .`
- Check you're in the correct virtual environment

**Cells not appearing/disappearing**
- Restart the Jupyter kernel after code changes
- The module caches the old version until restart

**Slow performance**
- Reduce `fov_size` for faster rendering
- Lower `n_objects` to simulate fewer cells
- Disable plotting during time-lapse acquisition

**Images too dark/bright**
- Adjust `laser_intensity` and `laser_power`
- Check `ADC_per_photon_conversion` and `ADC_offset`
- Verify sample mask has appropriate fluorophore density

For more help, please [open an issue](https://github.com/HenriquesLab/mAIcroscopysandbox/issues) on GitHub.
