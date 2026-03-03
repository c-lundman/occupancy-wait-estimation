# kff-v2

Queue Flow Reconciliation and FIFO Wait Estimation for airport areas (e.g., border control), using timestamped in/out person-count events.

## Purpose

This package exists to turn noisy person-count timestamps into physically plausible queue estimates and actionable KPIs for airport operations. It is intended for offline analysis workflows where teams receive periodic data extracts and need robust occupancy and waiting-time estimates despite imperfect sensors.

## Problem Statement

Given timestamp measurements of people entering and exiting a bounded area, estimate:

- occupancy over time (queue length proxy),
- throughput and flow characteristics,
- FIFO-based waiting-time metrics.

Raw counting data can be faulty. A naive occupancy estimate from cumulative inflow minus outflow may become unphysical (negative occupancy). The project goal is to recover physically plausible queue dynamics from noisy measurements.

## Operating Context

- Data is processed in batch mode (e.g., nightly tabular updates), not live.
- Multiple inflow and outflow sensors may exist.
- Sensor quality can differ by direction (for example, outflow more reliable than inflow).

## Core Assumptions (Initial Version)

- Time is discretized into fixed bins (initial default: 1 minute).
- Queue area is represented as a single stock-and-flow system.
- First In, First Out (FIFO) approximation is used for wait estimation.
- Occupancy is nonnegative at all times.

## Why Not Naive Cumulative Difference

If inflow is undercounted while outflow is accurate, cumulative `(in - out)` quickly drifts negative. This violates physical reality and contaminates downstream waiting-time estimates.

## Chosen Modeling Approach

Use a constrained convex optimization (Quadratic Program, QP) to reconcile measured flows into latent "true" flows that are both data-consistent and physically feasible.

### Inputs per time bin `t`

- measured inflow `i*_t`,
- measured outflow `o*_t`,
- optional reliability weights for each stream.

### Decision variables

- corrected inflow `i_t`,
- corrected outflow `o_t`,
- occupancy `q_t`.

### Constraints (linear)

- flow balance: `q_t = q_{t-1} + i_t - o_t`,
- nonnegativity: `q_t >= 0`, `i_t >= 0`, `o_t >= 0`,
- initial condition `q_0` known or estimated,
- optional terminal condition on `q_T`.

### Objective (quadratic)

Minimize weighted corrections from measured data:

`sum_t [ w_in(t) * (i_t - i*_t)^2 + w_out(t) * (o_t - o*_t)^2 ]`

Optional regularization terms may be added to discourage unrealistic spikes in corrected flows.

### QP Formulation Details

Decision variables (unknowns solved by the optimizer):

- `i_t`: corrected inflow at minute `t`
- `o_t`: corrected outflow at minute `t`
- `q_t`: corrected occupancy at minute `t`

Why the constraints are linear:

- Each constraint is a linear expression in the decision variables (no products or squares of variables).
- Examples:
- `q_t - q_{t-1} - i_t + o_t = 0`
- `q_t >= 0`, `i_t >= 0`, `o_t >= 0`

Why the objective is quadratic:

- Data-fit terms use squared deviations:
- `(i_t - i*_t)^2`, `(o_t - o*_t)^2`
- Optional smoothness terms also use squared differences:
- `(i_t - i_{t-1})^2`, `(o_t - o_{t-1})^2`

This combination (quadratic objective + linear constraints) is a Quadratic Program (QP), which is convex in this setup. That gives a stable global optimum for nightly/batch estimation.

### Why this formulation

- Produces physically plausible occupancy trajectories.
- Encodes trust asymmetry (e.g., keep trusted outflow closer to measurement).
- Solves to a global optimum when kept in convex QP form.
- Fast and stable for nightly runs at 1-minute bins.

## FIFO Wait Estimation

After flow reconciliation:

- reconstruct cumulative arrivals and departures,
- pair `n`th arrival with `n`th departure under FIFO,
- compute wait `w_n = t_out[n] - t_in[n]`,
- report distributional metrics (p50, p90, p95, mean).

## Synthetic Data and Benchmark Plan

We will build controlled dummy datasets to test estimation quality against known truth.

### Ground-truth ("perfect") dataset

- Event-level in/out timestamps generated from a known queue process.
- Derived perfect binned flows, occupancy, and FIFO waits.
- Initial artifact: `data/synthetic/perfect/day_2026-01-15/` (deterministic seed).
- Generation command: `python3 scripts/generate_perfect_dataset.py`.

### Lossy variants derived from truth

- missed counts (false negatives),
- spurious counts (false positives),
- timestamp jitter,
- direction-dependent reliability (e.g., inflow worse than outflow).
- Initial artifacts: `data/synthetic/lossy/day_2026-01-15/` with:
- `mild_noise`,
- `asymmetric_inflow_loss`,
- `spurious_outflow`,
- `mixed_heavy_noise`.
- Generation command: `python3 scripts/generate_lossy_datasets.py`.

### Evaluation

Compare reconstructed outputs against ground truth:

- occupancy error over time,
- wait-time metric error (p50/p90/p95/mean),
- physicality violations before/after reconciliation,
- correction magnitude (distance from measured to corrected flows).

## Initial Time Resolution

Start with 1-minute bins.

Rationale:

- retains operational queue dynamics better than coarse 5-minute bins,
- avoids excessive noise from very fine bins,
- remains computationally lightweight for convex QP solvers at daily scale.

## Implementation Direction (Planned)

Python package focused on:

- event/flow preprocessing,
- QP-based reconciliation,
- FIFO wait estimation,
- evaluation utilities for synthetic experiments.

Planned solver stack:

- `cvxpy` modeling interface,
- `OSQP` backend solver (initial default).

## Current Implementation Status

Step 3 is now implemented with a first QP reconciler:

- module: `src/kff_v2/reconcile.py`,
- public API: `reconcile_minute_flows(df: pandas.DataFrame, config: ReconcileConfig) -> pandas.DataFrame`,
- default behavior: enforce nonnegative occupancy and adjust flows with weighted least-squares corrections.

Batch reconciliation command on synthetic variants:

- `.venv/bin/python scripts/run_reconciliation.py`

Plotting comparison command (perfect vs lossy vs reconciled):

- `.venv/bin/python scripts/plot_flows.py --variant mild_noise`
- output default: `local/plots/day_2026-01-15_mild_noise_flows.png` (ignored by git)

Multi-day banked traffic dataset and episode-based reconciliation:

- generate: `.venv/bin/python scripts/generate_banked_multiday_dataset.py`
- reconcile by episodes: `.venv/bin/python scripts/run_episode_reconciliation.py`

Banked synthetic data now also includes event-level files for easier manual loading:

- perfect: `data/synthetic/perfect_banked/<key>/events.csv`
- lossy variants: `data/synthetic/lossy_banked/<key>/<variant>/measured_events.csv`

Current outputs:

- `data/synthetic/reconciled/day_2026-01-15/<variant>/reconciled_minute_flows.csv`,
- `data/synthetic/reconciled/day_2026-01-15/<variant>/summary.json`.

Primary API for timestamp inputs:

- `estimate_queue_from_timestamps(in_df, out_df, options=None, return_debug=False)`
- default output index: `Tid`
- default output columns:
- `Pax i kö`,
- `Pax ur kö`.
- `Pax in i kö`,
- `Väntetid`.

`Väntetid` semantics:

- minute-based FIFO waiting time for pax just exiting during that minute,
- computed after queue correction,
- `NaN` for minutes where corrected outflow is zero.

Example:

```python
import pandas as pd
from kff_v2 import estimate_queue_from_timestamps

in_df = pd.DataFrame({"timestamp": ["2026-01-20T06:00:05Z", "2026-01-20T06:00:31Z"]})
out_df = pd.DataFrame({"timestamp": ["2026-01-20T06:01:10Z"]})

queue_df = estimate_queue_from_timestamps(in_df, out_df)
print(queue_df.head())
```

Preset-based configuration example:

```python
import pandas as pd
from kff_v2 import EstimateQueueOptions, estimate_queue_from_timestamps, make_reconcile_config

opts = EstimateQueueOptions(
    reconcile=make_reconcile_config("trust_outflow", w_in=1.0, w_out=100.0)
)
queue_df = estimate_queue_from_timestamps(in_df, out_df, options=opts)
```

Available presets:

- `default`: recommended baseline for mixed quality flows.
- `trust_outflow`: strong trust in outflow (e.g. PPC out + lossy inflow).
- `balanced`: minimal priors, symmetric trust.
- `aggressive_peak_fill`: stronger inflow peak reconstruction.

## Phased Roadmap

1. Define synthetic generator and create perfect datasets.
2. Implement lossy data corruption pipeline.
3. Build reconciliation model (QP) and produce corrected flows/occupancy.
4. Compute FIFO waits and KPI summaries.
5. Evaluate against known truth and tune bin size/weights.
6. Expand to scenario library and robustness tests.

## Testing Strategy

1. Unit tests:
- time parsing/binning correctness,
- occupancy balance identity (`q_t = q_{t-1} + i_t - o_t`),
- FIFO pairing correctness on deterministic toy inputs.
2. Physics tests:
- reconciled outputs must satisfy `q_t >= 0` and nonnegative corrected flows.
3. Regression tests on synthetic truth:
- run estimator on lossy variants and compare to perfect ground truth,
- track occupancy error and wait-metric error (p50/p90/p95/mean).
4. Scenario tests:
- verify behavior across corruption patterns (`mild_noise`, `asymmetric_inflow_loss`, `spurious_outflow`, `mixed_heavy_noise`).
5. CI gate:
- run `pytest` on every change; add coverage threshold once core modules are implemented.

Current local smoke command (no external installs required):

- `.venv/bin/python -m unittest discover -s tests -v`

## CI

GitHub Actions workflow:

- `.github/workflows/ci.yml`

Runs on push and pull request to `main`:

- `ruff check .`
- `pytest -q`
