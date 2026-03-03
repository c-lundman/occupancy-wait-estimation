"""kff-v2 package."""

from kff_v2.episodes import EpisodeDetectConfig, detect_queue_episodes, reconcile_by_episodes
from kff_v2.fifo import add_fifo_wait_columns
from kff_v2.interface import EstimateQueueOptions, estimate_queue_from_timestamps
from kff_v2.metrics import (
    correction_size_metrics,
    occupancy_error_metrics,
    occupancy_physical_metrics,
    wait_time_metrics,
)
from kff_v2.presets import ReconcilePreset, make_reconcile_config
from kff_v2.reconcile import ReconcileConfig, reconcile_minute_flows

__all__ = [
    "__version__",
    "EpisodeDetectConfig",
    "EstimateQueueOptions",
    "ReconcileConfig",
    "ReconcilePreset",
    "add_fifo_wait_columns",
    "correction_size_metrics",
    "detect_queue_episodes",
    "estimate_queue_from_timestamps",
    "make_reconcile_config",
    "occupancy_error_metrics",
    "occupancy_physical_metrics",
    "reconcile_by_episodes",
    "reconcile_minute_flows",
    "wait_time_metrics",
]
__version__ = "0.1.1"
