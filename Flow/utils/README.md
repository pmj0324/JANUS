# Utils Documentation

## 📚 Overview

The `utils` package provides comprehensive utilities for data processing, visualization, and analysis in the GENESIS IceCube neutrino event generation project. This package is organized into specialized submodules for different functionalities.

## 🗂️ Package Structure

```
utils/
├── __init__.py                    # Package initialization and main exports
├── README.md                      # This documentation
├── denormalization.py             # Data normalization/denormalization utilities
├── fast_3d_plot.py               # Legacy: Ultra-fast 3D plotting (deprecated)
├── gpu_utils.py                  # GPU optimization utilities
├── event_visualization/           # Event visualization modules
│   ├── __init__.py
│   ├── _helpers.py               # Common visualization helpers
│   ├── event_show.py             # NPZ file event visualization
│   ├── event_grid.py             # Grid-based event visualization
│   ├── event_array.py            # Direct array visualization
│   ├── event_dataloader.py       # Dataloader integration
│   ├── event_dataloader_viz.py   # Advanced dataloader visualization
│   ├── event_fast.py             # Ultra-fast event plotting
│   └── event_script_legacy.py    # Legacy visualization script
└── h5/                           # HDF5 data utilities
    ├── __init__.py
    ├── h5_hist.py                # Histogram generation
    ├── h5_reader.py              # HDF5 file reading
    ├── h5_stats.py               # Statistical analysis
    ├── h5_time_align.py          # Time alignment utilities
    └── h5_distribution.py        # Data distribution visualization
```

## 🚀 Quick Start

### Basic Imports

```python
# Main utilities
from utils import (
    # Data processing
    denormalize_signal, denormalize_label, denormalize_full_event,
    
    # Event visualization
    show_event_from_npz, plot_event_fast, show_event_grid,
    show_event_from_array, show_event_from_dataloader,
    
    # H5 utilities
    plot_hist_pair, analyze_h5_dataset, read_h5_event
)

# Specific module imports
from utils.event_visualization.event_show import show_event_from_npz
from utils.event_visualization.event_fast import plot_event_fast
from utils.h5.h5_hist import plot_hist_pair
from utils.denormalization import denormalize_from_config
```

## 📊 Core Modules

### 1. Data Normalization (`denormalization.py`)

Handles conversion between normalized and original data scales for training and visualization.

#### Key Functions

| Function | Purpose | Usage |
|----------|---------|-------|
| `denormalize_signal()` | Convert normalized signals back to original scale | Training, sampling, visualization |
| `denormalize_label()` | Convert normalized labels back to original scale | Event analysis, validation |
| `denormalize_full_event()` | Convert entire event data | Complete event processing |
| `denormalize_from_config()` | Denormalize using config parameters | Automated processing |

#### Example Usage

```python
from utils.denormalization import denormalize_signal, denormalize_from_config

# Manual denormalization
original_signal = denormalize_signal(
    normalized_signal,
    charge_offset=0.0,
    charge_scale=1.0,
    time_offset=0.0,
    time_scale=1.0,
    time_transform="ln"
)

# Using config
original_data = denormalize_from_config(
    normalized_data,
    config_file="configs/default.yaml"
)
```

#### Parameters

- **Signal Normalization**: `charge_offset`, `charge_scale`, `time_offset`, `time_scale`
- **Time Transform**: `"ln"`, `"log10"`, or `None`
- **Label Normalization**: Energy, zenith, azimuth, position offsets and scales

### 2. Event Visualization (`event_visualization/`)

Comprehensive event visualization system with multiple specialized modules.

#### 2.1 Basic Event Visualization (`event_show.py`)

Visualize events from NPZ files with full 3D PMT sphere rendering.

```python
from utils.event_visualization.event_show import show_event_from_npz

fig, ax = show_event_from_npz(
    npz_path="outputs/sample_0000.npz",
    detector_csv="configs/detector_geometry.csv",
    output_path="event_visualization.png",
    show=True,
    figure_size=(16, 12),
    sphere_resolution=(50, 25),
    base_radius=5.0,
    radius_scale=0.2,
    alpha=0.8,
    show_detector_hull=True,
    show_background=True
)
```

**Features:**
- 3D PMT sphere rendering with NPE → size, Time → color mapping
- Detector hull outline visualization
- Background PMT display
- Event information overlay (Energy, Zenith, Azimuth, Position)
- Configurable styling and resolution

#### 2.2 Ultra-Fast Visualization (`event_fast.py`)

Optimized for speed - perfect for training loops and real-time visualization.

```python
from utils.event_visualization.event_fast import plot_event_fast, plot_event_comparison_fast

# Single event
fig, ax = plot_event_fast(
    charge_data=event_charges,      # (5160,) numpy array
    time_data=event_times,          # (5160,) numpy array
    geometry=detector_geometry,     # (5160, 3) numpy array
    labels=event_labels,            # (6,) numpy array
    output_path="fast_event.png",
    plot_type="both",               # "npe", "time", or "both"
    figure_size=(20, 8),
    sphere_size=10.0,
    alpha=0.8
)

# Comparison visualization
fig = plot_event_comparison_fast(
    real_data={'charge': real_charges, 'time': real_times, 'geometry': real_geom, 'labels': real_labels},
    generated_data={'charge': gen_charges, 'time': gen_times, 'geometry': gen_geom, 'labels': gen_labels},
    output_path="comparison.png",
    max_events=4
)
```

**Features:**
- Ultra-fast rendering optimized for speed
- Direct array input (no NPZ files needed)
- Side-by-side NPE and time plots
- Batch comparison visualization
- Minimal memory footprint

#### 2.3 Grid Visualization (`event_grid.py`)

Display NPE and time distributions in separate grid layouts.

```python
from utils.event_visualization.event_grid import show_event_grid

fig, axes = show_event_grid(
    npz_path="sample.npz",
    detector_csv="configs/detector_geometry.csv",
    output_path="grid_visualization.png",
    grid_layout="2x1",              # "2x1", "1x2", "2x2"
    figure_size=(16, 12),
    show_separate=True,             # Separate NPE and time plots
    show_combined=True              # Combined plot
)
```

**Features:**
- Grid-based layout options
- Separate NPE and time visualizations
- Combined view with dual encoding
- Statistical overlays
- Customizable grid arrangements

#### 2.4 Direct Array Visualization (`event_array.py`)

Visualize events directly from NumPy arrays without NPZ files.

```python
from utils.event_visualization.event_array import show_event_from_array

fig, ax = show_event_from_array(
    charge_data=charges,            # (5160,) or (B, 5160)
    time_data=times,                # (5160,) or (B, 5160)
    geometry=geometry,              # (5160, 3) or (B, 5160, 3)
    labels=labels,                  # (6,) or (B, 6)
    output_path="array_viz.png",
    plot_type="both",
    show_individual=True,           # Show individual PMTs
    show_statistics=True            # Show statistical overlays
)
```

**Features:**
- Direct NumPy array input
- Batch processing support
- Training loop integration
- Real-time visualization
- No file I/O overhead

#### 2.5 Dataloader Integration (`event_dataloader.py`)

Seamlessly integrate with PMTSignalsH5 dataloader for training monitoring.

```python
from utils.event_visualization.event_dataloader import show_event_from_dataloader

# Visualize specific event from dataloader
result = show_event_from_dataloader(
    dataloader=my_dataloader,
    event_index=42,
    output_path="dataloader_event.png",
    show_normalized=True,
    show_denormalized=True,
    save_npz=True
)
```

**Features:**
- Direct dataloader integration
- Automatic normalization handling
- Training monitoring capabilities
- Batch event visualization
- Real-time training feedback

#### 2.6 Advanced Dataloader Visualization (`event_dataloader_viz.py`)

Comprehensive dataloader analysis with automatic denormalization.

```python
from utils.event_visualization.event_dataloader_viz import visualize_event_from_dataloader

result = visualize_event_from_dataloader(
    config_path="configs/default.yaml",
    event_index=42,
    output_dir="outputs/event_analysis",
    detector_csv="configs/detector_geometry.csv",
    save_npz=True,
    save_png=True,
    plot_type="both",
    show_stats=True
)

# Access results
print(f"Event info: {result['labels_denormalized']}")
print(f"NPZ saved: {result['npz_path']}")
print(f"PNG saved: {result['png_path']}")
```

**Features:**
- Automatic config-based denormalization
- Comprehensive event statistics
- Multiple output formats
- CLI interface support
- Detailed event analysis

### 3. HDF5 Utilities (`h5/`)

Specialized tools for HDF5 data processing and analysis.

#### 3.1 Histogram Generation (`h5_hist.py`)

Create beautiful histograms for HDF5 data analysis.

```python
from utils.h5.h5_hist import plot_hist_pair

plot_hist_pair(
    h5_path="data/events.h5",
    bins=200,
    out_prefix="hist_analysis",
    log_time_transform="ln",
    exclude_zero=True,
    min_time_threshold=1.0,
    style="modern",
    logy=True,
    figsize=(16, 8)
)
```

**Features:**
- Charge (NPE) and time distribution histograms
- Multiple styling options (modern, elegant, classic)
- Log transformation support
- Zero value exclusion
- Customizable binning and thresholds

#### 3.2 HDF5 File Reading (`h5_reader.py`)

Efficient HDF5 file reading utilities.

```python
from utils.h5.h5_reader import read_h5_event, read_h5_batch, analyze_h5_dataset

# Read single event
event_data = read_h5_event(
    h5_path="data/events.h5",
    event_index=42,
    channels=["charge", "time"],
    normalize=True
)

# Read batch of events
batch_data = read_h5_batch(
    h5_path="data/events.h5",
    event_indices=[0, 1, 2, 3, 4],
    batch_size=5
)

# Analyze dataset structure
dataset_info = analyze_h5_dataset("data/events.h5")
print(f"Dataset shape: {dataset_info['shape']}")
print(f"Available keys: {dataset_info['keys']}")
```

**Features:**
- Efficient single event reading
- Batch processing support
- Dataset structure analysis
- Automatic normalization
- Memory-efficient loading

#### 3.3 Statistical Analysis (`h5_stats.py`)

Comprehensive statistical analysis tools for HDF5 datasets.

```python
from utils.h5.h5_stats import get_dataset_stats, compare_datasets

# Get dataset statistics
stats = get_dataset_stats(
    h5_path="data/events.h5",
    sample_size=1000,
    channels=["charge", "time"]
)

print(f"Charge stats: {stats['charge']}")
print(f"Time stats: {stats['time']}")

# Compare two datasets
comparison = compare_datasets(
    h5_path1="data/real_events.h5",
    h5_path2="data/generated_events.h5",
    sample_size=500
)
```

**Features:**
- Comprehensive statistical analysis
- Dataset comparison tools
- Sampling-based analysis
- Distribution analysis
- Quality assessment metrics

#### 3.4 Time Alignment (`h5_time_align.py`)

Utilities for time series data alignment and processing.

```python
from utils.h5.h5_time_align import align_time_data, process_time_series

# Align time data
aligned_data = align_time_data(
    time_data=raw_times,
    reference_time=0.0,
    max_time=1000.0,
    align_method="first_hit"
)

# Process time series
processed_data = process_time_series(
    time_data=aligned_data,
    transform="ln",
    exclude_zeros=True,
    normalize=True
)
```

**Features:**
- Time alignment algorithms
- Time series preprocessing
- Reference time alignment
- Outlier detection and handling
- Time transformation utilities

#### 3.5 Distribution Visualization (`h5_distribution.py`)

Command-line tool for H5 data distribution visualization.

```bash
# Basic usage
python3 utils/h5/h5_distribution.py -p data/events.h5

# Advanced options
python3 utils/h5/h5_distribution.py \
    -p data/events.h5 \
    --bins 300 \
    --log-time ln \
    --exclude-zero \
    --min-time 1.0 \
    --style modern \
    --logy \
    --figsize 20 12 \
    --out my_analysis
```

**Features:**
- Command-line interface
- Multiple visualization styles
- Configurable parameters
- Batch processing support
- High-quality output generation

### 4. GPU Utilities (`gpu_utils.py`)

GPU optimization and monitoring utilities.

```python
from utils.gpu_utils import optimize_gpu_settings, monitor_gpu_usage

# Optimize GPU settings
optimal_settings = optimize_gpu_settings(
    model_size="large",
    available_memory="24GB"
)

# Monitor GPU usage
gpu_stats = monitor_gpu_usage(
    duration=60,  # seconds
    interval=1    # seconds
)
```

## 🎨 Visualization Styles and Options

### Style Presets

| Style | Description | Use Case |
|-------|-------------|----------|
| `modern` | Clean, contemporary design | Publications, presentations |
| `elegant` | Sophisticated, refined look | Reports, documentation |
| `classic` | Traditional scientific style | Legacy compatibility |

### Color Schemes

- **NPE Visualization**: Viridis, Plasma, Inferno
- **Time Visualization**: Plasma, Magma, Jet
- **Combined Views**: Dual encoding with size + color

### Resolution Options

- **Sphere Resolution**: `(u_steps, v_steps)` for PMT spheres
- **Figure Size**: `(width, height)` in inches
- **Output DPI**: 150 (default), 300 (publication quality)

## 🔧 Configuration and Customization

### Default Parameters

```python
# Event visualization defaults
DEFAULT_FIGURE_SIZE = (16, 12)
DEFAULT_SPHERE_RESOLUTION = (50, 25)
DEFAULT_BASE_RADIUS = 5.0
DEFAULT_RADIUS_SCALE = 0.2
DEFAULT_ALPHA = 0.8

# H5 analysis defaults
DEFAULT_BINS = 200
DEFAULT_LOG_TIME = "ln"
DEFAULT_STYLE = "modern"
DEFAULT_EXCLUDE_ZERO = True
```

### Custom Styling

```python
# Custom color scheme
custom_colors = {
    'npe': 'viridis',
    'time': 'plasma',
    'background': 'lightgray',
    'hull': 'darkgray'
}

# Custom figure styling
fig_style = {
    'figsize': (20, 12),
    'dpi': 300,
    'facecolor': 'white',
    'edgecolor': 'black',
    'linewidth': 2
}
```

## 📈 Performance Optimization

### Speed Optimizations

1. **Fast Visualization**: Use `event_fast.py` for training loops
2. **Batch Processing**: Process multiple events simultaneously
3. **Memory Management**: Use sampling for large datasets
4. **GPU Acceleration**: Enable CUDA when available

### Memory Management

```python
# Efficient batch processing
batch_size = 32
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    visualize_batch(batch, output_dir=f"batch_{i//batch_size}")
    del batch  # Explicit cleanup
```

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```python
# Fix: Ensure proper path setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### Memory Issues
```python
# Fix: Reduce resolution or batch size
plot_event_fast(
    charge_data, time_data, geometry, labels,
    sphere_resolution=(20, 10),  # Reduced resolution
    figure_size=(12, 8)          # Smaller figure
)
```

#### Visualization Quality
```python
# Fix: Increase resolution and DPI
show_event_from_npz(
    npz_path, output_path,
    sphere_resolution=(100, 50),  # Higher resolution
    figure_size=(24, 16),         # Larger figure
    dpi=300                       # High DPI
)
```

### Debug Mode

```python
# Enable debug output
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug parameters
show_event_from_npz(
    npz_path, output_path,
    debug=True,                   # Enable debug output
    verbose=True                  # Detailed logging
)
```

## 📝 Examples and Tutorials

### Example 1: Basic Event Visualization

```python
from utils.event_visualization.event_show import show_event_from_npz

# Visualize a single event
fig, ax = show_event_from_npz(
    npz_path="outputs/sample_0000.npz",
    output_path="event_visualization.png",
    show=True
)
```

### Example 2: Training Loop Integration

```python
from utils.event_visualization.event_fast import plot_event_fast

# During training
for epoch in range(num_epochs):
    for batch_idx, (signals, geometry, labels) in enumerate(dataloader):
        # Training code...
        
        # Visualize every 100 batches
        if batch_idx % 100 == 0:
            plot_event_fast(
                charge_data=signals[0, 0].cpu().numpy(),
                time_data=signals[0, 1].cpu().numpy(),
                geometry=geometry[0].cpu().numpy(),
                labels=labels[0].cpu().numpy(),
                output_path=f"training_epoch_{epoch}_batch_{batch_idx}.png"
            )
```

### Example 3: Data Analysis Pipeline

```python
from utils.h5.h5_reader import read_h5_event
from utils.h5.h5_stats import get_dataset_stats
from utils.event_visualization.event_array import show_event_from_array

# Analyze dataset
stats = get_dataset_stats("data/events.h5", sample_size=1000)
print(f"Dataset statistics: {stats}")

# Visualize sample events
for i in range(5):
    event = read_h5_event("data/events.h5", event_index=i)
    show_event_from_array(
        charge_data=event['charge'],
        time_data=event['time'],
        geometry=event['geometry'],
        labels=event['labels'],
        output_path=f"analysis_event_{i}.png"
    )
```

### Example 4: Batch Comparison

```python
from utils.event_visualization.event_fast import plot_event_comparison_fast

# Compare real vs generated events
real_data = {
    'charge': real_charges,
    'time': real_times,
    'geometry': real_geometry,
    'labels': real_labels
}

generated_data = {
    'charge': gen_charges,
    'time': gen_times,
    'geometry': gen_geometry,
    'labels': gen_labels
}

fig = plot_event_comparison_fast(
    real_data=real_data,
    generated_data=generated_data,
    output_path="real_vs_generated.png",
    max_events=8
)
```

## 🔄 Migration Guide

### From Legacy Code

#### Old Import Pattern
```python
# ❌ Old way (deprecated)
from utils.visualization import create_3d_event_plot, EventVisualizer
from utils.fast_3d_plot import plot_event_3d
```

#### New Import Pattern
```python
# ✅ New way (recommended)
from utils.event_visualization.event_show import show_event_from_npz
from utils.event_visualization.event_fast import plot_event_fast
from utils.event_visualization.event_grid import show_event_grid
```

#### Function Mapping

| Old Function | New Function | Notes |
|--------------|--------------|-------|
| `create_3d_event_plot()` | `show_event_from_npz()` | Same functionality, better API |
| `plot_event_3d()` | `plot_event_fast()` | Enhanced performance |
| `EventVisualizer` | `show_event_from_array()` | Simplified interface |

## 📚 API Reference

### Core Functions

#### `denormalize_signal(signal, offsets, scales, time_transform)`
Convert normalized signal data back to original scale.

**Parameters:**
- `signal`: Normalized signal tensor/array
- `offsets`: Offset values for denormalization
- `scales`: Scale values for denormalization
- `time_transform`: Time transformation type

**Returns:** Denormalized signal data

#### `show_event_from_npz(npz_path, output_path, detector_csv, **kwargs)`
Visualize event from NPZ file with full 3D rendering.

**Parameters:**
- `npz_path`: Path to NPZ file
- `output_path`: Output image path
- `detector_csv`: Detector geometry CSV path
- `**kwargs`: Visualization options

**Returns:** `(fig, ax)` matplotlib objects

#### `plot_event_fast(charge_data, time_data, geometry, labels, **kwargs)`
Ultra-fast event visualization from arrays.

**Parameters:**
- `charge_data`: Charge values array (5160,)
- `time_data`: Time values array (5160,)
- `geometry`: PMT geometry array (5160, 3)
- `labels`: Event labels array (6,)

**Returns:** `(fig, axes)` matplotlib objects

#### `plot_hist_pair(h5_path, bins, out_prefix, **kwargs)`
Generate histogram pair for H5 data.

**Parameters:**
- `h5_path`: Path to HDF5 file
- `bins`: Number of histogram bins
- `out_prefix`: Output file prefix
- `**kwargs`: Histogram options

**Returns:** None (saves files)

## 🤝 Contributing

### Adding New Visualization Modules

1. Create new module in `event_visualization/`
2. Follow naming convention: `event_[functionality].py`
3. Include comprehensive docstrings
4. Add CLI support with `argparse`
5. Update `__init__.py` exports
6. Add tests and examples

### Code Style Guidelines

- Use type hints for all function parameters
- Include comprehensive docstrings
- Follow PEP 8 style guidelines
- Use descriptive variable names
- Include error handling and validation

## 📄 License

This utilities package is part of the GENESIS project. See the main project license for details.

## 🆘 Support

For issues and questions:
1. Check this documentation
2. Review example code
3. Check the main project issues
4. Contact the development team

---

*Last updated: October 2024*
*Version: 2.0.0*