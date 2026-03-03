import pandas as pd

from occupancy_wait_estimation import (
    EpisodeDetectConfig,
    EstimateQueueOptions,
    ReconcileConfig,
    detect_queue_episodes,
    estimate_queue_from_timestamps,
    reconcile_minute_flows,
)


def test_detect_queue_episodes_empty_when_no_activity() -> None:
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=60, freq="1min"),
            "in_count": [0.0] * 60,
            "out_count": [0.0] * 60,
        }
    )
    episodes = detect_queue_episodes(df)
    assert episodes.empty


def test_detect_queue_episodes_filters_short_spikes() -> None:
    in_count = [0.0] * 30
    out_count = [0.0] * 30
    in_count[10] = 1.0
    out_count[11] = 1.0
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=30, freq="1min"),
            "in_count": in_count,
            "out_count": out_count,
        }
    )
    cfg = EpisodeDetectConfig(min_active_minutes=5, min_episode_minutes=20, buffer_minutes=2)
    episodes = detect_queue_episodes(df, config=cfg)
    assert episodes.empty


def test_reconcile_handles_outflow_before_inflow_measurements() -> None:
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=5, freq="1min"),
            "in_count": [0.0, 0.0, 3.0, 0.0, 0.0],
            "out_count": [2.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    out = reconcile_minute_flows(df, ReconcileConfig(q0=0.0, w_in=1.0, w_out=1.0))
    assert (out["occupancy_corrected_end"] >= -1e-6).all()


def test_estimate_queue_empty_inputs_returns_expected_schema() -> None:
    in_df = pd.DataFrame({"timestamp": []})
    out_df = pd.DataFrame({"timestamp": []})
    queue = estimate_queue_from_timestamps(
        in_df,
        out_df,
        options=EstimateQueueOptions(include_fifo_wait=True),
    )
    assert queue.index.name == "Tid"
    assert list(queue.columns) == ["Pax i kö", "Pax ur kö", "Pax in i kö", "Väntetid"]
    assert len(queue) == 0

