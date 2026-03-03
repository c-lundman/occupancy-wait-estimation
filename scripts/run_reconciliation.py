#!/usr/bin/env python3
"""Run QP reconciliation on all lossy variants for a synthetic day."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from occupancy_wait_estimation import (
    ReconcileConfig,
    add_fifo_wait_columns,
    correction_size_metrics,
    occupancy_error_metrics,
    occupancy_physical_metrics,
    reconcile_minute_flows,
    wait_time_metrics,
)


def main() -> None:
    day = "day_2026-01-15"
    lossy_root = Path("data/synthetic/lossy") / day
    perfect_path = Path("data/synthetic/perfect") / day / "minute_flows.csv"
    out_root = Path("data/synthetic/reconciled") / day
    out_root.mkdir(parents=True, exist_ok=True)

    perfect = pd.read_csv(perfect_path)[["minute_start_utc", "occupancy_end"]].rename(
        columns={"occupancy_end": "occupancy_truth_end"}
    )

    config = ReconcileConfig(
        q0=0.0,
        w_in=1.0,
        w_out=10.0,
        smooth_in=0.05,
        smooth_out=0.05,
    )

    for variant_dir in sorted([p for p in lossy_root.iterdir() if p.is_dir()]):
        input_path = variant_dir / "minute_flows.csv"
        measured = pd.read_csv(input_path)

        reconciled = reconcile_minute_flows(measured, config=config)
        merged = reconciled.merge(perfect, on="minute_start_utc", how="left")
        merged["occupancy_abs_err"] = (
            merged["occupancy_corrected_end"] - merged["occupancy_truth_end"]
        ).abs()
        merged = add_fifo_wait_columns(merged, episode_col="episode_id", in_episode_col="in_episode")

        out_dir = out_root / variant_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_dir / "reconciled_minute_flows.csv", index=False)

        occ_err = occupancy_error_metrics(merged["occupancy_corrected_end"], merged["occupancy_truth_end"])
        summary = {
            "variant": variant_dir.name,
            "config": {
                "q0": config.q0,
                "w_in": config.w_in,
                "w_out": config.w_out,
                "smooth_in": config.smooth_in,
                "smooth_out": config.smooth_out,
            },
            "naive_occupancy": {
                **occupancy_physical_metrics(measured["naive_occupancy_end"]),
            },
            "reconciled_occupancy": {
                **occupancy_physical_metrics(merged["occupancy_corrected_end"]),
                "mae_vs_truth": occ_err["mae"],
                "p95_abs_err_vs_truth": occ_err["p95_abs_err"],
            },
            "fifo_wait_minutes": {
                **wait_time_metrics(merged["Väntetid"]),
            },
            "correction_size": {
                **correction_size_metrics(
                    merged["in_count_measured"],
                    merged["out_count_measured"],
                    merged["in_count_corrected"],
                    merged["out_count_corrected"],
                )
            },
        }
        with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    print(f"Wrote reconciled outputs to: {out_root}")


if __name__ == "__main__":
    main()
