import pandas as pd
import pytest

from occupancy_wait_estimation import ReconcileConfig, estimate_queue_from_timestamps, reconcile_minute_flows


def test_estimate_queue_only_inflow_events_stays_physical() -> None:
    in_df = pd.DataFrame(
        {"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:00:25Z", "2026-01-20T06:01:05Z"]}
    )
    out_df = pd.DataFrame({"timestamp": []})
    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert len(queue) >= 2
    assert float(queue["Pax i kö"].min()) >= -1e-9
    assert float(queue["Pax ur kö"].sum()) >= -1e-9


def test_estimate_queue_only_outflow_events_stays_physical() -> None:
    in_df = pd.DataFrame({"timestamp": []})
    out_df = pd.DataFrame(
        {"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:00:25Z", "2026-01-20T06:01:05Z"]}
    )
    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert len(queue) >= 2
    assert float(queue["Pax i kö"].min()) >= -1e-9
    assert float(queue["Pax ur kö"].min()) >= -1e-9
    assert float(queue["Pax in i kö"].min()) >= -1e-9


def test_estimate_queue_drops_unparseable_timestamps() -> None:
    in_df = pd.DataFrame(
        {"timestamp": ["bad", "2026-01-20T06:00:05Z", "2026-01-20 06:00:35+00:00", "not_a_ts"]}
    )
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})
    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert len(queue) >= 2
    assert float(queue["Pax in i kö"].sum()) > 0.0


def test_estimate_queue_handles_unsorted_duplicate_timestamps() -> None:
    in_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:30Z",
                "2026-01-20T06:00:10Z",
                "2026-01-20T06:00:30Z",
            ]
        }
    )
    out_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:03:20Z",
                "2026-01-20T06:03:05Z",
                "2026-01-20T06:03:55Z",
            ]
        }
    )
    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert float(queue["Pax in i kö"].max()) >= 3.0 - 1e-6
    assert float(queue["Pax i kö"].max()) >= 0.0


def test_reconcile_rejects_negative_weights() -> None:
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min"),
            "in_count": [0.0, 1.0, 0.0],
            "out_count": [0.0, 0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="non-negative"):
        reconcile_minute_flows(df, ReconcileConfig(w_in=-1.0))


def test_reconcile_rejects_invalid_activity_source() -> None:
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="1min"),
            "in_count": [0.0, 1.0, 0.0],
            "out_count": [0.0, 0.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="activity_source"):
        reconcile_minute_flows(df, ReconcileConfig(activity_source="invalid"))
