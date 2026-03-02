#!/usr/bin/env python3
"""Benchmark max-queue reconstruction error for single-arrival scenario."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kff_v2 import estimate_queue_from_timestamps


def _load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Missing timestamp column in {path}")
    return df


def _true_queue_from_perfect(in_df: pd.DataFrame, out_df: pd.DataFrame) -> pd.Series:
    ts_in = pd.to_datetime(in_df["timestamp"], utc=True)
    ts_out = pd.to_datetime(out_df["timestamp"], utc=True)
    all_min = pd.concat([ts_in, ts_out], ignore_index=True).dt.floor("1min")
    idx = pd.date_range(start=all_min.min(), end=all_min.max(), freq="1min", tz="UTC")
    in_counts = ts_in.dt.floor("1min").value_counts().sort_index().reindex(idx, fill_value=0)
    out_counts = ts_out.dt.floor("1min").value_counts().sort_index().reindex(idx, fill_value=0)
    return (in_counts - out_counts).cumsum()


def main() -> None:
    root = Path("data/scenarios/single_arrival_flight")
    in_ppc = _load_events(root / "PPC_in" / "events.csv")
    out_ppc = _load_events(root / "PPC_out" / "events.csv")
    in_rpc50 = _load_events(root / "RPC50L_in" / "events.csv")

    q_true_series = _true_queue_from_perfect(in_ppc, out_ppc)
    q_true = float(q_true_series.max())
    t_peak_true = q_true_series.idxmax().tz_localize(None)

    rows = []
    for inflow_name, in_df in [("PPC", in_ppc), ("RPC50L", in_rpc50)]:
        queue = estimate_queue_from_timestamps(in_df, out_ppc)
        q_method = float(queue["Pax i kö"].max()) if not queue.empty else 0.0
        t_peak_method = queue["Pax i kö"].idxmax() if not queue.empty else pd.NaT
        pe = 0.0 if q_true <= 0 else 100.0 * (q_true - q_method) / q_true
        peak_time_err_min = (
            float((t_peak_method - t_peak_true).total_seconds() / 60.0)
            if pd.notna(t_peak_method)
            else float("nan")
        )
        rows.append(
            {
                "scenario": "single_arrival_flight",
                "inflow_pc": inflow_name,
                "outflow_pc": "PPC",
                "Q_true": q_true,
                "Q_method": q_method,
                "PE": pe,
                "t_peak_true": t_peak_true,
                "t_peak_method": t_peak_method,
                "peak_time_error_min": peak_time_err_min,
            }
        )

    metrics = pd.DataFrame(rows)
    out_path = root / "metrics.csv"
    metrics.to_csv(out_path, index=False)

    # Print DataFrame for quick iterative evaluation.
    print(metrics.to_string(index=False))
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()

