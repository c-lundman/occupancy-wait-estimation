#!/usr/bin/env python3
"""Generate timestamp datasets for the single-arrival-flight scenario."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random

import pandas as pd


def _iso_z(ts: datetime) -> str:
    return ts.isoformat().replace("+00:00", "Z")


def generate_single_arrival(
    *,
    seed: int = 123,
    n_pax: int = 200,
    deboarding_minutes: float = 10.0,
    service_pax_per_min: float = 10.0,
    service_start_delay_min: float = 0.0,
) -> tuple[list[datetime], list[datetime], dict]:
    rng = random.Random(seed)
    t0 = datetime(2026, 2, 1, 8, 0, 0, tzinfo=timezone.utc)

    # Front-loaded but still within deboarding window.
    in_times = []
    for _ in range(n_pax):
        u = min(max(rng.betavariate(2.0, 3.0), 0.0), 1.0)
        sec = u * deboarding_minutes * 60.0
        in_times.append(t0 + timedelta(seconds=sec))
    in_times.sort()

    service_sec = 60.0 / service_pax_per_min
    service_start = t0 + timedelta(minutes=service_start_delay_min)
    out_times = []
    prev_departure = service_start
    for in_ts in in_times:
        start_service = max(in_ts, prev_departure)
        out_ts = start_service + timedelta(seconds=service_sec)
        out_times.append(out_ts)
        prev_departure = out_ts

    meta = {
        "scenario": "single_arrival_flight",
        "seed": seed,
        "n_pax": n_pax,
        "deboarding_minutes": deboarding_minutes,
        "service_pax_per_min": service_pax_per_min,
        "service_start_delay_min": service_start_delay_min,
        "start_ts_utc": _iso_z(t0),
        "end_ts_utc": _iso_z(out_times[-1]),
    }
    return in_times, out_times, meta


def write_events(path: Path, timestamps: list[datetime]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": [_iso_z(ts) for ts in timestamps]}).to_csv(path, index=False)


def make_rpc50l(in_times: list[datetime], seed: int = 999) -> list[datetime]:
    rng = random.Random(seed)
    return [ts for ts in in_times if rng.random() >= 0.5]


def main() -> None:
    in_times, out_times, meta = generate_single_arrival()

    root = Path("data/scenarios/single_arrival_flight")
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

