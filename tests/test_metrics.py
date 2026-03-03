import pandas as pd

from occupancy_wait_estimation import (
    correction_size_metrics,
    occupancy_error_metrics,
    occupancy_physical_metrics,
    wait_time_metrics,
)


def test_occupancy_physical_metrics_counts_negatives() -> None:
    s = pd.Series([3.0, -0.5, 2.0, -0.0000001])
    m = occupancy_physical_metrics(s, negative_eps=1e-6)
    assert m["negative_minutes"] == 1
    assert m["min"] == -0.5
    assert m["max"] == 3.0


def test_occupancy_error_metrics_basic() -> None:
    corrected = pd.Series([1.0, 3.0, 5.0])
    truth = pd.Series([1.0, 2.0, 7.0])
    m = occupancy_error_metrics(corrected, truth)
    assert abs(m["mae"] - 1.0) < 1e-12
    assert m["max_abs_err"] == 2.0


def test_wait_time_metrics_handles_nans() -> None:
    waits = pd.Series([None, 1.0, 2.0, 10.0])
    m = wait_time_metrics(waits)
    assert m["mean"] > 0.0
    assert m["p95"] >= m["p90"] >= m["p50"]


def test_correction_size_metrics() -> None:
    m = correction_size_metrics(
        in_measured=pd.Series([1.0, 2.0]),
        out_measured=pd.Series([1.0, 1.0]),
        in_corrected=pd.Series([1.5, 1.5]),
        out_corrected=pd.Series([0.0, 2.0]),
    )
    assert abs(m["in_abs_adjust_sum"] - 1.0) < 1e-12
    assert abs(m["out_abs_adjust_sum"] - 2.0) < 1e-12

