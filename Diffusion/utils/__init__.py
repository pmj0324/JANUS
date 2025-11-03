# Legacy imports for backward compatibility
from .denormalization import (
    denormalize_signal,
    denormalize_label,
    denormalize_full_event,
    denormalize_from_config
)
# Legacy fast_3d_plot imports removed - use event_visualization.event_fast instead
# visualization.py removed - use event_visualization modules instead

# New organized modules
from .event_visualization import (
    show_event_from_npz,
    show_event_grid,
    show_event_from_array,
    show_event_from_dataloader,
    plot_event_fast,
    plot_event_comparison_fast,
    visualize_event_from_dataloader
)
from .h5 import (
    plot_hist_pair,
    analyze_h5_dataset,
    get_dataset_stats,
    align_time_data,
    read_h5_event,
    read_h5_batch
)

__all__ = [
    # Legacy functions
    "denormalize_signal",
    "denormalize_label",
    "denormalize_full_event",
    "denormalize_from_config",
    
    # New event visualization
    "show_event_from_npz",
    "show_event_grid",
    "show_event_from_array", 
    "show_event_from_dataloader",
    "plot_event_fast",
    "plot_event_comparison_fast",
    "visualize_event_from_dataloader",
    
    # H5 utilities
    "plot_hist_pair",
    "analyze_h5_dataset",
    "get_dataset_stats",
    "align_time_data",
    "read_h5_event",
    "read_h5_batch",
]