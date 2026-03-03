# Session Notes

## Release State

- Current release tag: `v0.1.0`
- Tag commit: `113713a24e95f195c91e9fd17adeb0fc7c23db8d`

## Core Public API

- `estimate_queue_from_timestamps(in_df, out_df, options=None, return_debug=False)`
- Module: `src/occupancy_wait_estimation/interface.py`

Default output:

- Index: `Tid` (UTC minute resolution)
- Columns (in order):
  - `Pax i kö`
  - `Pax ur kö`
  - `Pax in i kö`
  - `Väntetid`

`Väntetid`:

- minute-based FIFO wait for pax just exiting in that minute,
- computed after queue correction,
- `NaN` for minutes with corrected outflow `0`.

## Modeling Notes

- Queue reconstruction uses constrained convex QP (`cvxpy` + `OSQP`).
- Episode splitting is enabled by default in the timestamp API.
- FIFO waits are computed from corrected minute flows using cumulative in/out matching with a lagging minute index per episode.

## Main Modules

- Reconciliation: `src/occupancy_wait_estimation/reconcile.py`
- Episode detection/splitting: `src/occupancy_wait_estimation/episodes.py`
- FIFO waits: `src/occupancy_wait_estimation/fifo.py`
- Metrics helpers: `src/occupancy_wait_estimation/metrics.py`

## Data + Scripts

Daily synthetic flow:

- Perfect generator: `scripts/generate_perfect_dataset.py`
- Lossy generator: `scripts/generate_lossy_datasets.py`
- Reconcile: `scripts/run_reconciliation.py`

Banked multi-day synthetic flow:

- Generator: `scripts/generate_banked_multiday_dataset.py`
- Episode reconcile: `scripts/run_episode_reconciliation.py`

Plots:

- `scripts/plot_flows.py`
- Includes inflow/outflow/occupancy and true-vs-reconstructed `Väntetid`.
- Plot output path: `local/plots/` (ignored by git).

## Quality and CI

- Tests: `tests/`
- Current status: full suite passing locally.
- CI workflow: `.github/workflows/ci.yml`
  - runs `ruff check .` and `pytest -q` on push/PR to `main`.

## Practical Next Steps (if continuing)

1. Add wait-time accuracy metrics vs true waits directly into reconciliation summaries.
2. Add optional per-sensor weighting (if multiple in/out streams are modeled explicitly).
3. Add CLI flags for solver/episode options in scripts for easier parameter sweeps.
