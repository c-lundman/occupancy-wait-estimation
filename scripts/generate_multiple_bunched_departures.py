#!/usr/bin/env python3
"""Generate a bunched-departures scenario with overlapping departure banks."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import pandas as pd


def _iso_z(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def _sample_passenger_arrivals_for_flight(
    rng: random.Random,
    t0: datetime,
    center_min: float,
    n_pax: int,
    fwhm_min: float = 40.0,
    low_clip_min: float = -90.0,
    high_clip_min: float = 15.0,
) -> list[datetime]:
    """Sample passenger arrivals around a flight center (departure-like spread)."""
    sigma = fwhm_min / 2.354820045
    times: list[datetime] = []
    for _ in range(n_pax):
        x = rng.gauss(center_min - 35.0, sigma)  # arrivals mostly before flight center
        x = min(max(x, center_min + low_clip_min), center_min + high_clip_min)
        times.append(t0 + timedelta(minutes=x))
    return times


def generate_scenario(
    *,
    seed: int = 2030,
    service_pax_per_min: float = 8.0,
) -> tuple[list[datetime], list[datetime], dict]:
    rng = random.Random(seed)
    t0 = datetime(2026, 2, 5, 8, 0, 0, tzinfo=timezone.utc)

    # Reuse the bunched-arrivals center/pax profile to keep comparable bank structure.
    base_meta_path = Path("data/scenarios/multiple_bunched_arrivals/scenario.json")
    if not base_meta_path.exists():
        raise FileNotFoundError(
            "Missing base scenario metadata at "
            f"{base_meta_path}. Generate multiple_bunched_arrivals first."
        )
    base = json.loads(base_meta_path.read_text(encoding="utf-8"))
    base_centers = [float(x) for x in base["flight_centers_min"]]
    pax_per_flight = [int(x) for x in base["pax_per_flight"]]

    # Keep the same clustered tendency but enforce clearer separation between
    # departures so multiple peaks are visible in queue inflow.
    min_sep_min = 12.0
    centers = sorted(base_centers)
    for i in range(1, len(centers)):
        if centers[i] < centers[i - 1] + min_sep_min:
            centers[i] = centers[i - 1] + min_sep_min
    # Recentre to preserve roughly same central timing and keep within horizon.
    orig_mean = sum(base_centers) / len(base_centers)
    new_mean = sum(centers) / len(centers)
    shift = orig_mean - new_mean
    centers = [c + shift for c in centers]
    low, high = 20.0, 145.0
    centers = [min(max(c, low), high) for c in centers]
    centers.sort()

    in_times: list[datetime] = []
    for c, p in zip(centers, pax_per_flight):
        in_times.extend(_sample_passenger_arrivals_for_flight(rng, t0, c, p, fwhm_min=40.0))
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
        "scenario": "multiple_bunched_departures",
        "seed": seed,
        "service_pax_per_min": service_pax_per_min,
        "flight_centers_min": centers,
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


def main() -> None:
    in_times, out_times, meta = generate_scenario()
    root = Path("data/scenarios/multiple_bunched_departures")

    write_events(root / "PPC_in" / "events.csv", in_times)
    write_events(root / "PPC_out" / "events.csv", out_times)
    write_events(root / "RPC50L_in" / "events.csv", make_rpc50l(in_times))

    with (root / "scenario.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    print(f"Wrote scenario to: {root}")
    print(f"PPC in events: {len(in_times)}")
    print(f"PPC out events: {len(out_times)}")
    print(f"RPC50L in events: {len(make_rpc50l(in_times))}")


if __name__ == "__main__":
    main()
