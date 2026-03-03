import pandas as pd

from occupancy_wait_estimation import add_fifo_wait_columns


def test_fifo_wait_single_segment_simple_shift() -> None:
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1min"),
            "in_count_corrected": [1.0, 1.0, 0.0, 0.0],
            "out_count_corrected": [0.0, 0.0, 1.0, 1.0],
        }
    )
    out = add_fifo_wait_columns(df)
    waits = out["Väntetid"].tolist()
    assert waits[0] == 0.0
    assert waits[1] == 0.0
    assert abs(waits[2] - 2.0) < 1e-9
    assert abs(waits[3] - 2.0) < 1e-9


def test_fifo_wait_resets_per_episode() -> None:
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1min"),
            "in_count_corrected": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "out_count_corrected": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "episode_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "in_episode": [True, True, True, True, True, True, True, True],
        }
    )
    out = add_fifo_wait_columns(df, use_episode_boundaries=True)
    waits = out["Väntetid"].tolist()
    assert abs(waits[2] - 2.0) < 1e-9
    assert abs(waits[3] - 2.0) < 1e-9
    assert abs(waits[6] - 2.0) < 1e-9
    assert abs(waits[7] - 2.0) < 1e-9


def test_fifo_wait_default_ignores_episode_boundaries() -> None:
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="1min"),
            "in_count_corrected": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
            "out_count_corrected": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "episode_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "in_episode": [True, True, True, True, True, True, True, True],
        }
    )
    out = add_fifo_wait_columns(df)
    waits = out["Väntetid"].tolist()
    assert abs(waits[2] - 2.0) < 1e-9
    assert abs(waits[3] - 2.0) < 1e-9
    assert abs(waits[6] - 2.0) < 1e-9
    assert abs(waits[7] - 2.0) < 1e-9
