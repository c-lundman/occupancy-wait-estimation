import pandas as pd

from kff_v2 import EpisodeDetectConfig, EstimateQueueOptions, estimate_queue_from_timestamps


def test_estimate_queue_output_schema() -> None:
    in_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:05Z",
                "2026-01-20T06:00:35Z",
                "2026-01-20T06:01:12Z",
            ]
        }
    )
    out_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:45Z",
                "2026-01-20T06:01:40Z",
            ]
        }
    )

    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert queue.index.name == "Tid"
    assert list(queue.columns) == ["Pax i kö", "Pax ur kö", "Pax in i kö", "Väntetid"]
    assert queue.index.tz is None


def test_estimate_queue_debug_mode_returns_tuple() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:02:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})

    queue, debug = estimate_queue_from_timestamps(in_df, out_df, return_debug=True)
    assert queue.index.name == "Tid"
    assert "in_count_measured" in debug.columns
    assert "out_count_measured" in debug.columns
    assert "in_episode" in debug.columns
    assert "episode_start" in debug.columns
    assert "episode_end" in debug.columns
    assert "episode_start_tid" in debug.columns
    assert "episode_end_tid" in debug.columns
    assert "episode_duration_minutes" in debug.columns
    assert "Väntetid" in debug.columns


def test_estimate_queue_debug_mode_has_episode_boundaries() -> None:
    in_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:05Z",
                "2026-01-20T06:01:05Z",
                "2026-01-20T06:02:05Z",
            ]
        }
    )
    out_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:03:05Z",
                "2026-01-20T06:04:05Z",
                "2026-01-20T06:05:05Z",
            ]
        }
    )
    opts = EstimateQueueOptions(
        episodes=EpisodeDetectConfig(
            active_threshold=1.0,
            min_active_minutes=1,
            max_gap_minutes=0,
            min_episode_minutes=1,
            buffer_minutes=0,
        )
    )

    _, debug = estimate_queue_from_timestamps(in_df, out_df, options=opts, return_debug=True)

    assert bool(debug["episode_start"].any())
    assert bool(debug["episode_end"].any())
    assert debug.loc[debug["episode_start"], "episode_start_tid"].notna().all()
    assert debug.loc[debug["episode_end"], "episode_end_tid"].notna().all()


def test_estimate_queue_can_include_fifo_wait_column() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:02:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})
    opts = EstimateQueueOptions(include_fifo_wait=True)
    queue = estimate_queue_from_timestamps(in_df, out_df, options=opts)
    assert "Väntetid" in queue.columns


def test_estimate_queue_can_disable_fifo_wait_column() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:02:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})
    opts = EstimateQueueOptions(include_fifo_wait=False)
    queue = estimate_queue_from_timestamps(in_df, out_df, options=opts)
    assert "Väntetid" not in queue.columns


def test_estimate_queue_small_case_has_consistent_occupancy() -> None:
    in_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:00:01Z",
                "2026-01-20T06:00:10Z",
                "2026-01-20T06:00:40Z",
            ]
        }
    )
    out_df = pd.DataFrame(
        {
            "timestamp": [
                "2026-01-20T06:03:02Z",
                "2026-01-20T06:03:11Z",
                "2026-01-20T06:03:59Z",
            ]
        }
    )
    queue = estimate_queue_from_timestamps(in_df, out_df)
    assert abs(queue["Pax in i kö"].to_numpy() - [3.0, 0.0, 0.0, 0.0]).max() < 1e-9
    assert abs(queue["Pax ur kö"].to_numpy() - [0.0, 0.0, 0.0, 3.0]).max() < 1e-9
    assert abs(queue["Pax i kö"].to_numpy() - [3.0, 3.0, 3.0, 0.0]).max() < 1e-9
    waits = queue["Väntetid"].tolist()
    assert pd.isna(waits[0]) and pd.isna(waits[1]) and pd.isna(waits[2])
    assert waits[3] == 3.0


def test_estimate_queue_compact_interface_supports_weight_overrides() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:01:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:02:05Z"]})
    queue = estimate_queue_from_timestamps(
        in_df,
        out_df,
        w_in=1.0,
        w_out=100.0,
        multiplicative_strength=2.0,
        use_episode_splitting=True,
        include_fifo_wait=True,
    )
    assert queue.index.name == "Tid"
    assert list(queue.columns) == ["Pax i kö", "Pax ur kö", "Pax in i kö", "Väntetid"]


def test_estimate_queue_rejects_mixing_options_and_compact_args() -> None:
    in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z"]})
    out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:05Z"]})
    opts = EstimateQueueOptions()
    try:
        estimate_queue_from_timestamps(in_df, out_df, options=opts, w_out=100.0)
    except ValueError as exc:
        assert "either `options` or compact arguments" in str(exc)
        return
    assert False, "Expected ValueError when mixing options and compact args."
