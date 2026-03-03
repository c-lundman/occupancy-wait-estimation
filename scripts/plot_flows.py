#!/usr/bin/env python3
"""Plot perfect vs lossy vs reconciled flows and occupancy."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from occupancy_wait_estimation import add_fifo_wait_columns

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install with: "
        ".venv/bin/python -m pip install matplotlib"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        default="daily",
        choices=["daily", "banked"],
        help="Dataset family to plot.",
    )
    parser.add_argument(
        "--key",
        default=None,
        help="Dataset key folder name. Defaults by family if omitted.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Lossy/reconciled variant name. If omitted with --all, plots all variants.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all variants available under the selected family/key.",
    )
    parser.add_argument(
        "--day",
        default="day_2026-01-15",
        help="Deprecated alias for --key (kept for backward compatibility).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path (single-plot mode only).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively in addition to saving.",
    )
    return parser.parse_args()


def _resolve_paths(repo_root: Path, family: str, key: str, variant: str) -> tuple[Path, Path, Path]:
    if family == "daily":
        perfect_path = repo_root / "data" / "synthetic" / "perfect" / key / "minute_flows.csv"
        lossy_path = repo_root / "data" / "synthetic" / "lossy" / key / variant / "minute_flows.csv"
        reconciled_path = (
            repo_root
            / "data"
            / "synthetic"
            / "reconciled"
            / key
            / variant
            / "reconciled_minute_flows.csv"
        )
        return perfect_path, lossy_path, reconciled_path
    if family == "banked":
        perfect_path = repo_root / "data" / "synthetic" / "perfect_banked" / key / "minute_flows.csv"
        lossy_path = repo_root / "data" / "synthetic" / "lossy_banked" / key / variant / "minute_flows.csv"
        reconciled_path = (
            repo_root
            / "data"
            / "synthetic"
            / "reconciled_banked"
            / key
            / variant
            / "reconciled_minute_flows.csv"
        )
        return perfect_path, lossy_path, reconciled_path
    raise ValueError(f"Unsupported family: {family}")


def _plot_one(repo_root: Path, family: str, key: str, variant: str, out: Path, show: bool) -> None:
    perfect_path, lossy_path, reconciled_path = _resolve_paths(repo_root, family, key, variant)

    if not perfect_path.exists():
        raise FileNotFoundError(f"Missing perfect flows: {perfect_path}")
    if not lossy_path.exists():
        raise FileNotFoundError(f"Missing lossy flows: {lossy_path}")
    if not reconciled_path.exists():
        raise FileNotFoundError(
            f"Missing reconciled flows: {reconciled_path}. "
            "Run reconciliation for this dataset first."
        )

    perfect = pd.read_csv(perfect_path)
    lossy = pd.read_csv(lossy_path)
    reconciled = pd.read_csv(reconciled_path)

    for df in (perfect, lossy, reconciled):
        df["minute_start_utc"] = pd.to_datetime(df["minute_start_utc"], utc=True)

    # Build "true" minute FIFO wait from perfect corrected-equivalent flows.
    perfect_for_wait = perfect[["minute_start_utc", "in_count", "out_count"]].rename(
        columns={"in_count": "in_count_corrected", "out_count": "out_count_corrected"}
    )
    perfect_for_wait = add_fifo_wait_columns(perfect_for_wait)

    merged = perfect[["minute_start_utc", "in_count", "out_count", "occupancy_end"]].rename(
        columns={
            "in_count": "in_perfect",
            "out_count": "out_perfect",
            "occupancy_end": "occ_perfect",
        }
    )
    merged = merged.merge(
        lossy[["minute_start_utc", "in_count", "out_count", "naive_occupancy_end"]].rename(
            columns={
                "in_count": "in_lossy",
                "out_count": "out_lossy",
                "naive_occupancy_end": "occ_lossy",
            }
        ),
        on="minute_start_utc",
        how="inner",
    )
    merged = merged.merge(
        reconciled[
            [
                "minute_start_utc",
                "in_count_corrected",
                "out_count_corrected",
                "occupancy_corrected_end",
                "Väntetid",
            ]
        ].rename(
            columns={
                "in_count_corrected": "in_qp",
                "out_count_corrected": "out_qp",
                "occupancy_corrected_end": "occ_qp",
                "Väntetid": "wait_qp",
            }
        ),
        on="minute_start_utc",
        how="inner",
    )
    merged = merged.merge(
        perfect_for_wait[["minute_start_utc", "Väntetid"]].rename(columns={"Väntetid": "wait_true"}),
        on="minute_start_utc",
        how="inner",
    )

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    x = merged["minute_start_utc"]

    axes[0].plot(x, merged["in_perfect"], label="Perfect inflow", linewidth=1.2)
    axes[0].plot(x, merged["in_lossy"], label="Lossy inflow", linewidth=1.0, alpha=0.8)
    axes[0].plot(x, merged["in_qp"], label="QP corrected inflow", linewidth=1.0)
    axes[0].set_ylabel("Inflow (pax/min)")
    axes[0].set_title(f"Inflow comparison: {family} / {key} / {variant}")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(x, merged["out_perfect"], label="Perfect outflow", linewidth=1.2)
    axes[1].plot(x, merged["out_lossy"], label="Lossy outflow", linewidth=1.0, alpha=0.8)
    axes[1].plot(x, merged["out_qp"], label="QP corrected outflow", linewidth=1.0)
    axes[1].set_ylabel("Outflow (pax/min)")
    axes[1].set_title("Outflow comparison")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(x, merged["occ_perfect"], label="Perfect occupancy", linewidth=1.2)
    axes[2].plot(x, merged["occ_lossy"], label="Naive lossy occupancy", linewidth=1.0, alpha=0.8)
    axes[2].plot(x, merged["occ_qp"], label="QP corrected occupancy", linewidth=1.0)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("Occupancy (pax)")
    axes[2].set_title("Occupancy comparison")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.25)
    axes[3].plot(x, merged["wait_true"], label="True FIFO wait (perfect)", linewidth=1.2)
    axes[3].plot(x, merged["wait_qp"], label="Reconstructed FIFO wait (QP)", linewidth=1.0)
    axes[3].set_ylabel("Väntetid (min)")
    axes[3].set_title("FIFO wait-time comparison")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.25)
    axes[3].set_xlabel("Time (UTC)")

    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Wrote plot: {out}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    key = args.key or args.day
    if args.key is None and args.day == "day_2026-01-15":
        if args.family == "daily":
            key = "day_2026-01-15"
        elif args.family == "banked":
            key = "multi_2026-01-20_2026-01-22"

    if args.all:
        if args.family == "daily":
            reconciled_root = repo_root / "data" / "synthetic" / "reconciled" / key
        else:
            reconciled_root = repo_root / "data" / "synthetic" / "reconciled_banked" / key
        variants = sorted([p.name for p in reconciled_root.iterdir() if p.is_dir()])
        for variant in variants:
            out = repo_root / "local" / "plots" / f"{args.family}_{key}_{variant}_flows.png"
            _plot_one(repo_root, args.family, key, variant, out=out, show=False)
        return

    if not args.variant:
        raise SystemExit("Provide --variant (or use --all).")
    out = (
        Path(args.out)
        if args.out
        else repo_root / "local" / "plots" / f"{args.family}_{key}_{args.variant}_flows.png"
    )
    _plot_one(repo_root, args.family, key, args.variant, out=out, show=args.show)


if __name__ == "__main__":
    main()
