#!/usr/bin/env python3
"""Plot inflow/outflow/queue diagnostics for bunched-arrivals scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from occupancy_wait_estimation import EstimateQueueOptions, ReconcileConfig, estimate_queue_from_timestamps


def _bin_counts(df: pd.DataFrame, col: str = "timestamp") -> pd.Series:
    ts = pd.to_datetime(df[col], utc=True)
    return ts.dt.floor("1min").value_counts().sort_index()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inflow-variant",
        choices=["RPC50L", "FLPC05"],
        default="RPC50L",
        help="Inflow measurement variant to reconstruct from.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path("data/scenarios/multiple_bunched_arrivals")
    inflow_dir = f"{args.inflow_variant}_in"
    out_png = Path(
        f"local/plots/multiple_bunched_arrivals_{args.inflow_variant.lower()}_wout100_eps001_mult2_diagnostics.png"
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)

    in_ppc = pd.read_csv(root / "PPC_in" / "events.csv")
    in_measured = pd.read_csv(root / inflow_dir / "events.csv")
    out_ppc = pd.read_csv(root / "PPC_out" / "events.csv")

    in_ppc_c = _bin_counts(in_ppc)
    in_measured_c = _bin_counts(in_measured)
    out_ppc_c = _bin_counts(out_ppc)

    idx = pd.date_range(
        start=min(in_ppc_c.index.min(), in_measured_c.index.min(), out_ppc_c.index.min()),
        end=max(in_ppc_c.index.max(), in_measured_c.index.max(), out_ppc_c.index.max()),
        freq="1min",
        tz="UTC",
    )

    in_ppc_s = in_ppc_c.reindex(idx, fill_value=0).astype(float)
    in_measured_s = in_measured_c.reindex(idx, fill_value=0).astype(float)
    out_ppc_s = out_ppc_c.reindex(idx, fill_value=0).astype(float)
    q_true = (in_ppc_s - out_ppc_s).cumsum()

    opts = EstimateQueueOptions(
        reconcile=ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=100.0,
            relative_inflow_error=True,
            relative_inflow_eps=0.01,
            relative_inflow_weight_min_scale=0.25,
            relative_inflow_weight_max_scale=16.0,
            multiplicative_inflow_prior=True,
            multiplicative_inflow_strength=2.0,
            multiplicative_alpha_min=0.2,
            multiplicative_alpha_max=4.0,
            adaptive_inflow_prior=True,
            activity_source="out",
            activity_window=7,
            activity_eps=0.5,
            inflow_weight_min_scale=0.25,
            inflow_weight_max_scale=4.0,
            smooth_in=0.0,
            smooth_out=0.0,
        )
    )
    queue = estimate_queue_from_timestamps(in_measured, out_ppc, options=opts)

    q_idx = pd.DatetimeIndex(queue.index).tz_localize("UTC")
    corrected = queue.copy()
    corrected.index = q_idx
    in_corr = corrected["Pax in i kö"].reindex(idx, fill_value=0.0)
    out_corr = corrected["Pax ur kö"].reindex(idx, fill_value=0.0)
    q_corr = corrected["Pax i kö"].reindex(idx, fill_value=0.0)

    x = idx.tz_localize(None)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)

    axes[0].plot(x, in_ppc_s.values, label="PPC_in", linewidth=1.5)
    axes[0].plot(x, in_measured_s.values, label=f"{args.inflow_variant}_in", linewidth=1.2)
    axes[0].plot(x, in_corr.values, label="Corrected inflow", linewidth=1.2)
    axes[0].set_ylabel("pax/min")
    axes[0].set_title(f"Inflow ({args.inflow_variant})")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(x, out_ppc_s.values, label="PPC_out", linewidth=1.5)
    axes[1].plot(x, out_corr.values, label="Corrected outflow", linewidth=1.2)
    axes[1].set_ylabel("pax/min")
    axes[1].set_title("Outflow")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(x, q_true.values, label="True queue", linewidth=1.5)
    axes[2].plot(x, q_corr.values, label="Corrected queue", linewidth=1.2)
    axes[2].set_ylabel("pax")
    axes[2].set_title("Queue Length")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")
    axes[2].set_xlabel("Time")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {out_png}")


if __name__ == "__main__":
    main()
