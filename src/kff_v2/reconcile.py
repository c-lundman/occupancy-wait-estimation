"""QP-based reconciliation of noisy minute-binned queue flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconcileConfig:
    """Configuration for QP flow reconciliation."""

    q0: float = 0.0
    w_in: float = 1.0
    w_out: float = 4.0
    relative_inflow_error: bool = False
    relative_inflow_eps: float = 1.0
    relative_inflow_weight_min_scale: float = 0.25
    relative_inflow_weight_max_scale: float = 16.0
    multiplicative_inflow_prior: bool = False
    multiplicative_inflow_strength: float = 2.0
    multiplicative_alpha_min: float = 0.0
    multiplicative_alpha_max: float = 10.0
    adaptive_inflow_prior: bool = False
    activity_source: str = "max_io"
    activity_window: int = 7
    activity_eps: float = 0.5
    inflow_weight_min_scale: float = 0.25
    inflow_weight_max_scale: float = 4.0
    smooth_in: float = 0.0
    smooth_out: float = 0.0
    nonnegative_flows: bool = True
    solver: str = "OSQP"
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    max_iter: int = 50_000


def _compute_inflow_weight_vector(
    in_measured: np.ndarray,
    out_measured: np.ndarray,
    cfg: ReconcileConfig,
) -> np.ndarray:
    """Compute time-varying inflow correction weights for prior shaping."""
    w = np.full_like(in_measured, float(cfg.w_in), dtype=float)

    if cfg.relative_inflow_error:
        eps = max(float(cfg.relative_inflow_eps), 1e-9)
        rel_scale = 1.0 / np.square(in_measured + eps)
        rel_scale = np.clip(
            rel_scale,
            float(cfg.relative_inflow_weight_min_scale),
            float(cfg.relative_inflow_weight_max_scale),
        )
        w = w * rel_scale

    if not cfg.adaptive_inflow_prior:
        return w

    if cfg.activity_source == "in":
        proxy = in_measured.copy()
    elif cfg.activity_source == "out":
        proxy = out_measured.copy()
    elif cfg.activity_source == "sum_io":
        proxy = in_measured + out_measured
    else:
        proxy = np.maximum(in_measured, out_measured)

    win = max(1, int(cfg.activity_window))
    proxy_s = (
        pd.Series(proxy, dtype=float)
        .rolling(window=win, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    raw_scale = 1.0 / (proxy_s + float(cfg.activity_eps))
    mean_scale = float(raw_scale.mean()) if len(raw_scale) else 1.0
    if mean_scale <= 0:
        mean_scale = 1.0
    scale = raw_scale / mean_scale
    scale = np.clip(
        scale,
        float(cfg.inflow_weight_min_scale),
        float(cfg.inflow_weight_max_scale),
    )
    return w * scale


def reconcile_minute_flows(
    df: pd.DataFrame,
    config: Optional[ReconcileConfig] = None,
) -> pd.DataFrame:
    """Reconcile noisy inflow/outflow series into physically feasible flows.

    Parameters
    ----------
    df:
        Input DataFrame with columns `minute_start_utc`, `in_count`, `out_count`.
    config:
        Optional reconciliation settings.

    Returns
    -------
    pandas.DataFrame
        Original timeline plus:
        `in_count_measured`, `out_count_measured`,
        `in_count_corrected`, `out_count_corrected`,
        `occupancy_corrected_end`.
    """
    cfg = config or ReconcileConfig()
    required_cols = {"minute_start_utc", "in_count", "out_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work = work.sort_values("minute_start_utc").reset_index(drop=True)

    in_measured = work["in_count"].astype(float).to_numpy()
    out_measured = work["out_count"].astype(float).to_numpy()
    n = len(work)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "minute_start_utc",
                "in_count_measured",
                "out_count_measured",
                "in_count_corrected",
                "out_count_corrected",
                "occupancy_corrected_end",
            ]
        )

    i = cp.Variable(n)
    o = cp.Variable(n)
    q = cp.Variable(n)
    alpha = cp.Variable() if cfg.multiplicative_inflow_prior else None
    w_in_vec = _compute_inflow_weight_vector(in_measured, out_measured, cfg)
    w_out_vec = np.full(n, float(cfg.w_out), dtype=float)

    objective_terms = [
        cp.sum(cp.multiply(w_in_vec, cp.square(i - in_measured))),
        cp.sum(cp.multiply(w_out_vec, cp.square(o - out_measured))),
    ]
    if cfg.smooth_in > 0 and n > 1:
        objective_terms.append(cfg.smooth_in * cp.sum_squares(i[1:] - i[:-1]))
    if cfg.smooth_out > 0 and n > 1:
        objective_terms.append(cfg.smooth_out * cp.sum_squares(o[1:] - o[:-1]))
    if cfg.multiplicative_inflow_prior:
        objective_terms.append(
            float(cfg.multiplicative_inflow_strength)
            * cp.sum_squares(i - alpha * in_measured)
        )

    constraints = [
        q[0] == cfg.q0 + i[0] - o[0],
        q >= 0,
    ]
    if n > 1:
        constraints.append(q[1:] == q[:-1] + i[1:] - o[1:])
    if cfg.nonnegative_flows:
        constraints.extend([i >= 0, o >= 0])
    if cfg.multiplicative_inflow_prior:
        constraints.append(alpha >= float(cfg.multiplicative_alpha_min))
        if cfg.multiplicative_alpha_max > cfg.multiplicative_alpha_min:
            constraints.append(alpha <= float(cfg.multiplicative_alpha_max))

    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
    solve_kwargs = {
        "solver": cfg.solver,
    }
    if cfg.solver.upper() == "OSQP":
        solve_kwargs.update(
            {
                "eps_abs": cfg.eps_abs,
                "eps_rel": cfg.eps_rel,
                "max_iter": cfg.max_iter,
            }
        )
    problem.solve(**solve_kwargs)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"QP solve failed with status: {problem.status}")

    i_val = np.asarray(i.value).reshape(-1)
    o_val = np.asarray(o.value).reshape(-1)
    q_val = np.asarray(q.value).reshape(-1)
    if cfg.nonnegative_flows:
        i_val = np.maximum(i_val, 0.0)
        o_val = np.maximum(o_val, 0.0)
    q_val = np.maximum(q_val, 0.0)

    out = pd.DataFrame(
        {
            "minute_start_utc": work["minute_start_utc"],
            "in_count_measured": in_measured,
            "out_count_measured": out_measured,
            "in_count_corrected": i_val,
            "out_count_corrected": o_val,
            "occupancy_corrected_end": q_val,
        }
    )
    if cfg.multiplicative_inflow_prior and alpha is not None and alpha.value is not None:
        out["inflow_alpha"] = float(alpha.value)
    return out
