"""Public DataFrame-first interface for queue estimation from timestamps."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import warnings

import pandas as pd

from kff_v2.episodes import EpisodeDetectConfig, detect_queue_episodes, reconcile_by_episodes
from kff_v2.fifo import add_fifo_wait_columns
from kff_v2.presets import make_reconcile_config
from kff_v2.reconcile import ReconcileConfig, reconcile_minute_flows


@dataclass(frozen=True)
class EstimateQueueOptions:
    """Options for timestamp-to-queue estimation."""

    in_timestamp_col: str = "timestamp"
    out_timestamp_col: str = "timestamp"
    freq: str = "1min"
    use_episode_splitting: bool = True
    reconcile: ReconcileConfig = make_reconcile_config("default")
    episodes: EpisodeDetectConfig = EpisodeDetectConfig()
    include_fifo_wait: bool = True


def _build_minute_flows(
    inflow: pd.DataFrame,
    outflow: pd.DataFrame,
    options: EstimateQueueOptions,
) -> pd.DataFrame:
    if options.in_timestamp_col not in inflow.columns:
        raise ValueError(f"inflow missing timestamp column: {options.in_timestamp_col}")
    if options.out_timestamp_col not in outflow.columns:
        raise ValueError(f"outflow missing timestamp column: {options.out_timestamp_col}")

    in_ts = pd.to_datetime(
        inflow[options.in_timestamp_col], utc=True, errors="coerce", format="mixed"
    ).dropna()
    out_ts = pd.to_datetime(
        outflow[options.out_timestamp_col], utc=True, errors="coerce", format="mixed"
    ).dropna()

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


def _attach_episode_debug_columns(
    debug: pd.DataFrame,
    episodes_df: pd.DataFrame,
) -> pd.DataFrame:
    out = debug.copy()
    out["episode_start"] = False
    out["episode_end"] = False
    out["episode_start_tid"] = pd.NaT
    out["episode_end_tid"] = pd.NaT
    out["episode_duration_minutes"] = float("nan")

    if episodes_df.empty:
        return out

    episode_meta = episodes_df.loc[
        :,
        ["episode_id", "start_ts_utc", "end_ts_utc", "duration_minutes"],
    ].copy()
    episode_meta["start_ts_utc"] = pd.to_datetime(episode_meta["start_ts_utc"], utc=True).dt.tz_localize(None)
    episode_meta["end_ts_utc"] = pd.to_datetime(episode_meta["end_ts_utc"], utc=True).dt.tz_localize(None)
    episode_meta = episode_meta.set_index("episode_id")

    out["episode_start_tid"] = out["episode_id"].map(episode_meta["start_ts_utc"])
    out["episode_end_tid"] = out["episode_id"].map(episode_meta["end_ts_utc"])
    out["episode_duration_minutes"] = out["episode_id"].map(episode_meta["duration_minutes"]).astype(float)
    out["episode_start"] = out["Tid"] == out["episode_start_tid"]
    out["episode_end"] = out["Tid"] == out["episode_end_tid"]
    return out


def estimate_queue_from_timestamps(
    inflow: pd.DataFrame,
    outflow: pd.DataFrame,
    options: Optional[EstimateQueueOptions] = None,
    *,
    trust: str | None = None,
    w_in: float | None = None,
    w_out: float | None = None,
    multiplicative_strength: float | None = None,
    use_episode_splitting: bool | None = None,
    include_fifo_wait: bool | None = None,
    return_debug: bool = False,
):
    """Estimate corrected minute queue series from in/out timestamp DataFrames.

    Minimal interface (recommended):
    - `w_in`, `w_out`, `multiplicative_strength`,
      `use_episode_splitting`, `include_fifo_wait`
    - Parameters set to `None` use effective defaults:
      `w_in=1.0`, `w_out=1.0`, `multiplicative_strength=2.0`,
      `use_episode_splitting=True`, `include_fifo_wait=True`.
    - `trust` is deprecated backward compatibility (`outflow`, `inflow`,
      `balanced`, `default`) and maps to default weight pairs.

    Advanced interface:
    - pass `options=EstimateQueueOptions(...)`
    - when `options` is used, compact arguments above must be left as `None`.

    Returns
    -------
    pandas.DataFrame
        Index `Tid`, columns:
        `Pax i kö`, `Pax ur kö`, `Pax in i kö`, `Väntetid`.

    If return_debug=True:
        Returns `(queue_df, debug_df)`, where `debug_df` contains measured and
        corrected minute series, including episode flags when enabled.
    """
    compact_args = (
        trust is not None
        or w_in is not None
        or w_out is not None
        or multiplicative_strength is not None
        or use_episode_splitting is not None
        or include_fifo_wait is not None
    )
    if options is not None and compact_args:
        raise ValueError("Use either `options` or compact arguments, not both.")

    if options is not None:
        opts = options
    else:
        rec = make_reconcile_config("default")
        if trust is not None:
            warnings.warn(
                "`trust` is deprecated; use explicit `w_in` and `w_out` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            trust_to_weights = {
                "default": (1.0, 1.0),
                "balanced": (1.0, 1.0),
                "outflow": (1.0, 100.0),
                "inflow": (100.0, 1.0),
            }
            if trust not in trust_to_weights:
                raise ValueError(
                    "Invalid `trust` value. Expected one of: default, balanced, outflow, inflow."
                )
            w_in_def, w_out_def = trust_to_weights[trust]
            rec = replace(rec, w_in=w_in_def, w_out=w_out_def)
        if w_in is not None:
            rec = replace(rec, w_in=float(w_in))
        if w_out is not None:
            rec = replace(rec, w_out=float(w_out))
        if multiplicative_strength is not None:
            rec = replace(
                rec,
                multiplicative_inflow_strength=float(multiplicative_strength),
                multiplicative_outflow_strength=float(multiplicative_strength),
            )
        opts = EstimateQueueOptions(
            reconcile=rec,
            use_episode_splitting=(
                bool(use_episode_splitting)
                if use_episode_splitting is not None
                else EstimateQueueOptions.use_episode_splitting
            ),
            include_fifo_wait=(
                bool(include_fifo_wait)
                if include_fifo_wait is not None
                else EstimateQueueOptions.include_fifo_wait
            ),
        )

    measured = _build_minute_flows(inflow, outflow, opts)

    if measured.empty:
        queue = pd.DataFrame(columns=["Pax i kö", "Pax ur kö", "Pax in i kö", "Väntetid"])
        queue.index = pd.DatetimeIndex([], name="Tid")
        if return_debug:
            return queue, measured
        return queue

    episodes_df = pd.DataFrame()
    if opts.use_episode_splitting:
        episodes_df = detect_queue_episodes(measured, opts.episodes)
        reconciled = reconcile_by_episodes(
            measured,
            reconcile_config=opts.reconcile,
            episode_config=opts.episodes,
            episodes=episodes_df,
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
        debug = _attach_episode_debug_columns(debug, episodes_df)
        debug = debug.set_index("Tid")
        return queue, debug
    return queue
