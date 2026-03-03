#!/usr/bin/env python3
"""Generate a bunched-arrivals scenario with stronger temporal overlap."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import numpy as np
import pandas as pd


def _iso_z(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _add_flight_burst(
    minute_flux: np.ndarray,
    center_min: float,
    pax: int,
    width_min: int,
) -> None:
    left = int(max(0, np.floor(center_min - width_min / 2)))
    right = int(min(len(minute_flux), np.ceil(center_min + width_min / 2)))
    if right <= left:
        return
    x = np.linspace(-2.3, 2.3, right - left)
    w = np.exp(-0.5 * x * x)
    w = w / w.sum()
    alloc = np.floor(w * pax).astype(int)
    residual = pax - int(alloc.sum())
    if residual > 0:
        alloc[np.argsort(-w)[:residual]] += 1
    minute_flux[left:right] += alloc


def generate_scenario(
    *,
    seed: int = 2029,
    horizon_min: int = 180,
    service_pax_per_min: float = 11.0,
) -> tuple[list[datetime], list[datetime], dict]:
    rng = random.Random(seed)
    t0 = datetime(2026, 2, 4, 8, 0, 0, tzinfo=timezone.utc)

    # Keep same number of flights + pax as overlapping scenario if available.
    base_meta_path = Path("data/scenarios/multiple_overlapping_arrivals/scenario.json")
    if base_meta_path.exists():
        base = json.loads(base_meta_path.read_text(encoding="utf-8"))
        pax_per_flight = [int(x) for x in base.get("pax_per_flight", [])]
    else:
        pax_per_flight = []

    if not pax_per_flight:
        # Fallback: deterministic 8-flight allocation.
        pax_per_flight = [150, 175, 185, 200, 190, 165, 125, 120]
    n_flights = len(pax_per_flight)

    # Gaussian-like bunching centered near 70 minutes.
    # Use minimum separation so all 8 flights remain visible as distinct arrivals,
    # while still strongly clustering around the center.
    mean_min = 70.0
    std_min = 14.0
    low_min, high_min = 20.0, 125.0
    min_sep_min = 4.0
    centers: list[float] = []
    attempts = 0
    while len(centers) < n_flights and attempts < 50_000:
        attempts += 1
        c = rng.gauss(mean_min, std_min)
        if c < low_min or c > high_min:
            continue
        if any(abs(c - x) < min_sep_min for x in centers):
            continue
        centers.append(c)
    if len(centers) < n_flights:
        # Deterministic fallback preserving bunching.
        centers = [46.0, 56.0, 64.0, 69.0, 73.0, 78.0, 86.0, 99.0][:n_flights]
    centers = np.array(sorted(centers))
    # Slightly narrower unloading windows to preserve per-flight structure.
    widths = [rng.randint(6, 10) for _ in range(n_flights)]

    in_flux = np.zeros(horizon_min, dtype=int)
    for c, p, w in zip(centers, pax_per_flight, widths):
        _add_flight_burst(in_flux, c, p, w)

    in_times: list[datetime] = []
    for m, c in enumerate(in_flux):
        minute_start = t0 + timedelta(minutes=int(m))
        for _ in range(int(c)):
            in_times.append(minute_start + timedelta(seconds=rng.random() * 60.0))
    in_times.sort()

    service_sec = 60.0 / service_pax_per_min
    prev_departure = t0
    out_times: list[datetime] = []
    for in_ts in in_times:
        start_service = max(in_ts, prev_departure)
        out_ts = start_service + timedelta(seconds=service_sec)
        out_times.append(out_ts)
        prev_departure = out_ts

    meta = {
        "scenario": "multiple_bunched_arrivals",
        "seed": seed,
        "n_flights": n_flights,
        "horizon_min": horizon_min,
        "service_pax_per_min": service_pax_per_min,
        "flight_centers_min": [round(float(x), 2) for x in centers],
        "pax_per_flight": pax_per_flight,
        "start_ts_utc": _iso_z(t0),
        "end_ts_utc": _iso_z(out_times[-1]) if out_times else _iso_z(t0),
    }
    return in_times, out_times, meta


def write_events(path: Path, timestamps: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": [_iso_z(ts) for ts in timestamps]}).to_csv(path, index=False)


def make_rpc50l(in_times: list[datetime], seed: int = 999) -> list[datetime]:
    rng = random.Random(seed)
    return [ts for ts in in_times if rng.random() >= 0.5]


def make_flpc_limit_per_minute(in_times: list[datetime], *, max_pax_per_min: int) -> list[datetime]:
    if max_pax_per_min < 0:
        raise ValueError("max_pax_per_min must be non-negative")
    if not in_times:
        return []
    ts = pd.Series(pd.to_datetime(in_times, utc=True)).sort_values()
    minute = ts.dt.floor("1min")
    keep = minute.groupby(minute).cumcount() < max_pax_per_min
    return [x.to_pydatetime() for x in ts[keep]]


def main() -> None:
    in_times, out_times, meta = generate_scenario()
    root = Path("data/scenarios/multiple_bunched_arrivals")
    in_rpc50 = make_rpc50l(in_times)
    in_flpc05 = make_flpc_limit_per_minute(in_times, max_pax_per_min=30)

    write_events(root / "PPC_in" / "events.csv", in_times)
    write_events(root / "PPC_out" / "events.csv", out_times)
    write_events(root / "RPC50L_in" / "events.csv", in_rpc50)
    write_events(root / "FLPC05_in" / "events.csv", in_flpc05)

    with (root / "scenario.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Wrote scenario to: {root}")
    print(f"PPC in events: {len(in_times)}")
    print(f"PPC out events: {len(out_times)}")
    print(f"RPC50L in events: {len(in_rpc50)}")
    print(f"FLPC05 in events: {len(in_flpc05)}")


if __name__ == "__main__":
    main()
