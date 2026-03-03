"""FIFO wait-time reconstruction on corrected minute flows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _fifo_wait_single_segment(
    in_flow: np.ndarray,
    out_flow: np.ndarray,
    *,
    outflow_eps: float = 1e-9,
    match_tol: float = 1e-6,
) -> np.ndarray:
    """Compute minute-based FIFO waiting times for one contiguous segment."""
    n = len(in_flow)
    wait_minutes = np.full(n, np.nan, dtype=float)
    if n == 0:
        return wait_minutes

    cum_in = np.cumsum(in_flow)
    cum_out = np.cumsum(out_flow)
    t_in = 0

    for t_out in range(n):
        if out_flow[t_out] <= outflow_eps:
            continue
        target = cum_out[t_out]
        while t_in < n - 1 and cum_in[t_in] < (target - match_tol):
            t_in += 1
        if cum_in[t_in] + match_tol >= target:
            wait_minutes[t_out] = float(t_out - t_in)
    return wait_minutes


def add_fifo_wait_columns(
    df: pd.DataFrame,
    *,
    in_col: str = "in_count_corrected",
    out_col: str = "out_count_corrected",
    episode_col: str = "episode_id",
    in_episode_col: str = "in_episode",
    wait_col: str = "Väntetid",
    outflow_eps: float = 1e-9,
    match_tol: float = 1e-6,
    use_episode_boundaries: bool = False,
) -> pd.DataFrame:
    """Add FIFO wait-time columns to a corrected minute-flow DataFrame.

    Wait time at minute `t_out` is defined for minutes with positive corrected
    outflow and corresponds to the pax just exiting at that minute.
    """
    required = {in_col, out_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for FIFO reconstruction: {sorted(missing)}")

    out = df.copy().reset_index(drop=True)
    out[wait_col] = np.nan

    if use_episode_boundaries:
        has_episode = episode_col in out.columns
        has_in_episode = in_episode_col in out.columns

        if has_episode and has_in_episode:
            mask = out[in_episode_col].fillna(False).astype(bool)
            episode_ids = [x for x in out.loc[mask, episode_col].dropna().unique()]
            if episode_ids:
                for eid in episode_ids:
                    seg_mask = mask & (out[episode_col] == eid)
                    seg = out.loc[seg_mask]
                    w = _fifo_wait_single_segment(
                        seg[in_col].astype(float).to_numpy(),
                        seg[out_col].astype(float).to_numpy(),
                        outflow_eps=outflow_eps,
                        match_tol=match_tol,
                    )
                    out.loc[seg_mask, wait_col] = w
                return out

    w = _fifo_wait_single_segment(
        out[in_col].astype(float).to_numpy(),
        out[out_col].astype(float).to_numpy(),
        outflow_eps=outflow_eps,
        match_tol=match_tol,
    )
    out[wait_col] = w
    return out
