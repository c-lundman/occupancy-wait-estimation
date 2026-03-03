from pathlib import Path

import pandas as pd

from occupancy_wait_estimation import EpisodeDetectConfig, ReconcileConfig, detect_queue_episodes, reconcile_by_episodes


REPO_ROOT = Path(__file__).resolve().parents[1]
LOSSY_BANKED = (
    REPO_ROOT
    / "data"
    / "synthetic"
    / "lossy_banked"
    / "multi_2026-01-20_2026-01-22"
    / "banked_asymmetric_in"
    / "minute_flows.csv"
)


def test_detects_multiple_episodes_in_banked_data() -> None:
    df = pd.read_csv(LOSSY_BANKED)
    episodes = detect_queue_episodes(df, EpisodeDetectConfig())
    assert len(episodes) >= 4


def test_episode_reconcile_is_nonnegative_inside_episodes() -> None:
    df = pd.read_csv(LOSSY_BANKED)
    out = reconcile_by_episodes(
        df,
        reconcile_config=ReconcileConfig(q0=0.0, w_in=1.0, w_out=12.0),
        episode_config=EpisodeDetectConfig(),
    )
    in_ep = out["in_episode"]
    assert (out.loc[in_ep, "occupancy_corrected_end"] >= -1e-6).all()

