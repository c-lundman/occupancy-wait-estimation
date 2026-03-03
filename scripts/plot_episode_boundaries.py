#!/usr/bin/env python3
"""Plot scenario flows/queue with detected episode boundaries overlaid."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from occupancy_wait_estimation import EpisodeDetectConfig, detect_queue_episodes, estimate_queue_from_timestamps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", default="multiple_bunched_arrivals")
    p.add_argument("--out", default=None)
    return p.parse_args()


def _bin_counts(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    return ts.dt.floor("1min").value_counts().sort_index()


def main() -> None:
    args = parse_args()
    root = Path("data/scenarios") / args.scenario
    in_ppc = pd.read_csv(root / "PPC_in" / "events.csv")
    in_rpc = pd.read_csv(root / "RPC50L_in" / "events.csv")
    out_ppc = pd.read_csv(root / "PPC_out" / "events.csv")

    in_ppc_c = _bin_counts(in_ppc)
    in_rpc_c = _bin_counts(in_rpc)
    out_ppc_c = _bin_counts(out_ppc)

    idx = pd.date_range(
        start=min(in_ppc_c.index.min(), in_rpc_c.index.min(), out_ppc_c.index.min()),
        end=max(in_ppc_c.index.max(), in_rpc_c.index.max(), out_ppc_c.index.max()),
        freq="1min",
        tz="UTC",
    )
    in_ppc_s = in_ppc_c.reindex(idx, fill_value=0).astype(float)
    in_rpc_s = in_rpc_c.reindex(idx, fill_value=0).astype(float)
    out_ppc_s = out_ppc_c.reindex(idx, fill_value=0).astype(float)
    q_true = (in_ppc_s - out_ppc_s).cumsum()

    measured = pd.DataFrame(
        {
            "minute_start_utc": idx,
            "in_count": in_rpc_s.values,
            "out_count": out_ppc_s.values,
        }
    )
    ep_cfg = EpisodeDetectConfig()
    episodes = detect_queue_episodes(measured, ep_cfg)

    queue = estimate_queue_from_timestamps(in_rpc, out_ppc, w_in=1.0, w_out=100.0)
    q_idx = pd.DatetimeIndex(queue.index).tz_localize("UTC")
    q_corr = queue["Pax i kö"].copy()
    q_corr.index = q_idx
    q_corr = q_corr.reindex(idx, fill_value=0.0)
    in_corr = queue["Pax in i kö"].copy()
    in_corr.index = q_idx
    in_corr = in_corr.reindex(idx, fill_value=0.0)

    x = idx.tz_localize(None)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(x, in_ppc_s.values, label="PPC_in", linewidth=1.5)
    axes[0].plot(x, in_rpc_s.values, label="RPC50L_in", linewidth=1.2)
    axes[0].plot(x, in_corr.values, label="Corrected inflow", linewidth=1.2)
    axes[0].set_title(f"Inflow with episode boundaries: {args.scenario}")
    axes[0].set_ylabel("pax/min")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, out_ppc_s.values, label="PPC_out", linewidth=1.5)
    axes[1].set_title("Outflow")
    axes[1].set_ylabel("pax/min")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, q_true.values, label="True queue", linewidth=1.5)
    axes[2].plot(x, q_corr.values, label="Corrected queue", linewidth=1.2)
    axes[2].set_title("Queue")
    axes[2].set_ylabel("pax")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xlabel("Time")

    # Overlay detected episodes on all panels.
    for _, row in episodes.iterrows():
        s = pd.to_datetime(row["start_ts_utc"], utc=True).tz_localize(None)
        e = pd.to_datetime(row["end_ts_utc"], utc=True).tz_localize(None)
        for ax in axes:
            ax.axvspan(s, e, color="orange", alpha=0.12)
            ax.axvline(s, color="orange", alpha=0.5, linewidth=0.8)
            ax.axvline(e, color="orange", alpha=0.5, linewidth=0.8)

    out = (
        Path(args.out)
        if args.out
        else Path("local/plots") / f"{args.scenario}_episode_boundaries.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote plot: {out}")
    print(f"Detected episodes: {len(episodes)}")


if __name__ == "__main__":
    main()
