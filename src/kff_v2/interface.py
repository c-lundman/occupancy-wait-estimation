"""Public DataFrame-first interface for queue estimation from timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from kff_v2.episodes import EpisodeDetectConfig, reconcile_by_episodes
from kff_v2.fifo import add_fifo_wait_columns
from kff_v2.reconcile import ReconcileConfig, reconcile_minute_flows


@dataclass(frozen=True)
class EstimateQueueOptions:
    """Options for timestamp-to-queue estimation."""

    in_timestamp_col: str = "timestamp"
    out_timestamp_col: str = "timestamp"
    freq: str = "1min"
    use_episode_splitting: bool = True
    reconcile: ReconcileConfig = ReconcileConfig(
        q0=0.0,
        w_in=1.0,
        w_out=1.0,
        relative_inflow_error=True,
        relative_inflow_eps=0.01,
        relative_inflow_weight_min_scale=0.25,
        relative_inflow_weight_max_scale=16.0,
        multiplicative_inflow_prior=True,
        multiplicative_inflow_strength=2.0,
        multiplicative_alpha_min=0.2,
        multiplicative_alpha_max=4.0,
        adaptive_inflow_prior=True,
        activity_source="max_io",
        activity_window=7,
        activity_eps=0.5,
        inflow_weight_min_scale=0.25,
        inflow_weight_max_scale=4.0,
        smooth_in=0.0,
        smooth_out=0.0,
    )
    episodes: EpisodeDetectConfig = EpisodeDetectConfig()
    include_fifo_wait: bool = True


def _build_minute_flows(
    in_df: pd.DataFrame,
    out_df: pd.DataFrame,
    options: EstimateQueueOptions,
) -> pd.DataFrame:
    if options.in_timestamp_col not in in_df.columns:
        raise ValueError(f"in_df missing timestamp column: {options.in_timestamp_col}")
    if options.out_timestamp_col not in out_df.columns:
        raise ValueError(f"out_df missing timestamp column: {options.out_timestamp_col}")

    in_ts = pd.to_datetime(in_df[options.in_timestamp_col], utc=True, errors="coerce").dropna()
    out_ts = pd.to_datetime(out_df[options.out_timestamp_col], utc=True, errors="coerce").dropna()

    if len(in_ts) == 0 and len(out_ts) == 0:
        return pd.DataFrame(columns=["minute_start_utc", "in_count", "out_count"])

    all_min = pd.Series(pd.concat([in_ts, out_ts], ignore_index=True)).dt.floor(options.freq)
    start = all_min.min()
    end = all_min.max()
    minute_index = pd.date_range(start=start, end=end, freq=options.freq, tz="UTC")

    in_counts = in_ts.dt.floor(options.freq).value_counts().sort_index().reindex(minute_index, fill_value=0)
    out_counts = (
        out_ts.dt.floor(options.freq).value_counts().sort_index().reindex(minute_index, fill_value=0)
    )

    return pd.DataFrame(
        {
            "minute_start_utc": minute_index,
            "in_count": in_counts.to_numpy(dtype=float),
            "out_count": out_counts.to_numpy(dtype=float),
        }
    )


def _format_queue_output(reconciled: pd.DataFrame) -> pd.DataFrame:
    tid = pd.to_datetime(reconciled["minute_start_utc"], utc=True).dt.tz_localize(None)
    queue = pd.DataFrame({"Tid": tid})
    queue["Pax i kö"] = reconciled["occupancy_corrected_end"].astype(float).to_numpy()
    queue["Pax ur kö"] = reconciled["out_count_corrected"].astype(float).to_numpy()
    queue["Pax in i kö"] = reconciled["in_count_corrected"].astype(float).to_numpy()
    if "Väntetid" in reconciled.columns:
        queue["Väntetid"] = reconciled["Väntetid"].astype(float).to_numpy()
    queue = queue.set_index("Tid")
    queue.index.name = "Tid"
    return queue


def estimate_queue_from_timestamps(
    in_df: pd.DataFrame,
    out_df: pd.DataFrame,
    options: Optional[EstimateQueueOptions] = None,
    *,
    return_debug: bool = False,
):
    """Estimate corrected minute queue series from in/out timestamp DataFrames.

    Returns
    -------
    pandas.DataFrame
        Index `Tid`, columns:
        `Pax i kö`, `Pax ur kö`, `Pax in i kö`, `Väntetid`.

    If return_debug=True:
        Returns `(queue_df, debug_df)`, where `debug_df` contains measured and
        corrected minute series, including episode flags when enabled.
    """
    opts = options or EstimateQueueOptions()
    measured = _build_minute_flows(in_df, out_df, opts)

    if measured.empty:
        queue = pd.DataFrame(columns=["Pax i kö", "Pax ur kö", "Pax in i kö", "Väntetid"])
        queue.index = pd.DatetimeIndex([], name="Tid")
        if return_debug:
            return queue, measured
        return queue

    if opts.use_episode_splitting:
        reconciled = reconcile_by_episodes(
            measured,
            reconcile_config=opts.reconcile,
            episode_config=opts.episodes,
        )
    else:
        reconciled = reconcile_minute_flows(measured, config=opts.reconcile)
        reconciled["episode_id"] = pd.NA
        reconciled["in_episode"] = False

    reconciled = add_fifo_wait_columns(reconciled)
    queue = _format_queue_output(
        reconciled if opts.include_fifo_wait else reconciled.drop(columns=["Väntetid"], errors="ignore")
    )
    if return_debug:
        debug = reconciled.copy()
        debug["Tid"] = pd.to_datetime(debug["minute_start_utc"], utc=True).dt.tz_localize(None)
        debug = debug.set_index("Tid")
        return queue, debug
    return queue
