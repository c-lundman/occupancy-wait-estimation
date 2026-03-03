"""Reusable metric utilities for queue reconstruction outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _safe_quantile(series: pd.Series, q: float) -> float:
    s = series.dropna()
    if s.empty:
        return 0.0
    return float(s.quantile(q))


def occupancy_physical_metrics(occupancy: pd.Series, *, negative_eps: float = 1e-6) -> dict[str, Any]:
    occ = occupancy.astype(float)
    if occ.empty:
        return {"negative_minutes": 0, "min": 0.0, "max": 0.0}
    return {
        "negative_minutes": int((occ < -negative_eps).sum()),
        "min": float(occ.min()),
        "max": float(occ.max()),
    }


def occupancy_error_metrics(corrected: pd.Series, truth: pd.Series) -> dict[str, float]:
    err = (corrected.astype(float) - truth.astype(float)).abs()
    if err.empty:
        return {"mae": 0.0, "p95_abs_err": 0.0, "max_abs_err": 0.0}
    return {
        "mae": float(err.mean()),
        "p95_abs_err": float(err.quantile(0.95)),
        "max_abs_err": float(err.max()),
    }


def wait_time_metrics(wait_minutes: pd.Series) -> dict[str, float]:
    s = wait_minutes.astype(float)
    return {
        "mean": float(s.dropna().mean()) if not s.dropna().empty else 0.0,
        "p50": _safe_quantile(s, 0.50),
        "p90": _safe_quantile(s, 0.90),
        "p95": _safe_quantile(s, 0.95),
        "max": float(s.dropna().max()) if not s.dropna().empty else 0.0,
    }


def correction_size_metrics(
    in_measured: pd.Series,
    out_measured: pd.Series,
    in_corrected: pd.Series,
    out_corrected: pd.Series,
) -> dict[str, float]:
    in_delta = (in_corrected.astype(float) - in_measured.astype(float)).abs()
    out_delta = (out_corrected.astype(float) - out_measured.astype(float)).abs()
    return {
        "in_abs_adjust_sum": float(in_delta.sum()),
        "out_abs_adjust_sum": float(out_delta.sum()),
        "in_abs_adjust_mean": float(in_delta.mean()) if not in_delta.empty else 0.0,
        "out_abs_adjust_mean": float(out_delta.mean()) if not out_delta.empty else 0.0,
    }

