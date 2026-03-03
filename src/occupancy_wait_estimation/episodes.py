"""Episode detection and per-episode reconciliation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from occupancy_wait_estimation.reconcile import ReconcileConfig, reconcile_minute_flows


@dataclass(frozen=True)
class EpisodeDetectConfig:
    """Configuration for queue episode detection."""

    active_threshold: float = 1.0
    min_active_minutes: int = 5
    max_gap_minutes: int = 10
    min_episode_minutes: int = 20
    buffer_minutes: int = 10


def detect_queue_episodes(
    df: pd.DataFrame,
    config: Optional[EpisodeDetectConfig] = None,
) -> pd.DataFrame:
    """Detect active queue episodes from minute flow data.

    Parameters
    ----------
    df:
        DataFrame with `minute_start_utc`, `in_count`, `out_count`.
    config:
        Optional episode detection settings.

    Returns
    -------
    pandas.DataFrame
        One row per episode with index bounds and timestamps.
    """
    cfg = config or EpisodeDetectConfig()
    required_cols = {"minute_start_utc", "in_count", "out_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.sort_values("minute_start_utc").reset_index(drop=True)
    activity = (work["in_count"].astype(float) + work["out_count"].astype(float)).to_numpy()
    active = activity >= cfg.active_threshold

    # Bridge short inactive gaps inside active periods.
    if cfg.max_gap_minutes > 0 and len(active) > 0:
        inactive_idx = np.flatnonzero(~active)
        if len(inactive_idx) > 0:
            runs: List[tuple[int, int]] = []
            start = inactive_idx[0]
            prev = start
            for idx in inactive_idx[1:]:
                if idx == prev + 1:
                    prev = idx
                    continue
                runs.append((start, prev))
                start = idx
                prev = idx
            runs.append((start, prev))
            for s, e in runs:
                run_len = e - s + 1
                if run_len <= cfg.max_gap_minutes:
                    left_active = s > 0 and active[s - 1]
                    right_active = e < len(active) - 1 and active[e + 1]
                    if left_active and right_active:
                        active[s : e + 1] = True

    active_idx = np.flatnonzero(active)
    if len(active_idx) == 0:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "start_idx",
                "end_idx",
                "start_ts_utc",
                "end_ts_utc",
                "duration_minutes",
            ]
        )

    runs: List[tuple[int, int]] = []
    s = int(active_idx[0])
    p = s
    for idx in active_idx[1:]:
        idx = int(idx)
        if idx == p + 1:
            p = idx
            continue
        runs.append((s, p))
        s = idx
        p = idx
    runs.append((s, p))

    rows = []
    n = len(work)
    episode_id = 0
    for start_idx, end_idx in runs:
        duration = end_idx - start_idx + 1
        if duration < cfg.min_active_minutes:
            continue
        start_buf = max(0, start_idx - cfg.buffer_minutes)
        end_buf = min(n - 1, end_idx + cfg.buffer_minutes)
        full_duration = end_buf - start_buf + 1
        if full_duration < cfg.min_episode_minutes:
            continue
        episode_id += 1
        rows.append(
            {
                "episode_id": episode_id,
                "start_idx": start_buf,
                "end_idx": end_buf,
                "start_ts_utc": work.loc[start_buf, "minute_start_utc"],
                "end_ts_utc": work.loc[end_buf, "minute_start_utc"],
                "duration_minutes": full_duration,
            }
        )
    return pd.DataFrame(rows)


def reconcile_by_episodes(
    df: pd.DataFrame,
    reconcile_config: Optional[ReconcileConfig] = None,
    episode_config: Optional[EpisodeDetectConfig] = None,
    episodes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run QP reconciliation independently per detected episode."""
    work = df.sort_values("minute_start_utc").reset_index(drop=True).copy()
    required_cols = {"minute_start_utc", "in_count", "out_count"}
    missing = required_cols - set(work.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "minute_start_utc": work["minute_start_utc"],
            "in_count_measured": work["in_count"].astype(float),
            "out_count_measured": work["out_count"].astype(float),
            "in_count_corrected": work["in_count"].astype(float),
            "out_count_corrected": work["out_count"].astype(float),
            "occupancy_corrected_end": np.zeros(len(work), dtype=float),
            "episode_id": np.nan,
            "in_episode": False,
        }
    )

    episodes_df = episodes.copy() if episodes is not None else detect_queue_episodes(work, config=episode_config)
    cfg = reconcile_config or ReconcileConfig()
    if episodes_df.empty:
        # Fallback: if no episodes are detected, reconcile the full window so
        # occupancy still reflects the corrected in/out flows.
        rec = reconcile_minute_flows(
            work.loc[:, ["minute_start_utc", "in_count", "out_count"]].reset_index(drop=True),
            config=cfg,
        )
        out.loc[:, "in_count_corrected"] = rec["in_count_corrected"].to_numpy()
        out.loc[:, "out_count_corrected"] = rec["out_count_corrected"].to_numpy()
        out.loc[:, "occupancy_corrected_end"] = rec["occupancy_corrected_end"].to_numpy()
        return out

    for row in episodes_df.itertuples(index=False):
        s = int(row.start_idx)
        e = int(row.end_idx)
        episode_df = work.loc[s:e, ["minute_start_utc", "in_count", "out_count"]].reset_index(drop=True)
        rec = reconcile_minute_flows(episode_df, config=cfg)
        out.loc[s:e, "in_count_corrected"] = rec["in_count_corrected"].to_numpy()
        out.loc[s:e, "out_count_corrected"] = rec["out_count_corrected"].to_numpy()
        out.loc[s:e, "occupancy_corrected_end"] = rec["occupancy_corrected_end"].to_numpy()
        out.loc[s:e, "episode_id"] = int(row.episode_id)
        out.loc[s:e, "in_episode"] = True

    # Outside detected episodes, treat corrected flows as zeroed inactive periods.
    inactive = ~out["in_episode"]
    out.loc[inactive, "in_count_corrected"] = 0.0
    out.loc[inactive, "out_count_corrected"] = 0.0
    out.loc[inactive, "occupancy_corrected_end"] = 0.0

    return out
