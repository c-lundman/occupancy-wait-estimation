"""User-facing presets for reconciliation configuration."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from occupancy_wait_estimation.reconcile import ReconcileConfig

ReconcilePreset = Literal[
    "default",
    "trust_outflow",
    "trust_inflow",
    "balanced",
    "aggressive_peak_fill",
]


def make_reconcile_config(
    preset: ReconcilePreset = "default",
    **overrides: float | bool | str,
) -> ReconcileConfig:
    """Build a `ReconcileConfig` from a named preset with optional overrides."""
    if preset == "default":
        cfg = ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=1.0,
            relative_inflow_error=True,
            relative_inflow_eps=0.01,
            relative_inflow_weight_min_scale=0.25,
            relative_inflow_weight_max_scale=16.0,
            relative_outflow_error=True,
            relative_outflow_eps=0.01,
            relative_outflow_weight_min_scale=0.25,
            relative_outflow_weight_max_scale=16.0,
            multiplicative_inflow_prior=True,
            multiplicative_inflow_strength=2.0,
            multiplicative_alpha_min=0.2,
            multiplicative_alpha_max=4.0,
            multiplicative_outflow_prior=True,
            multiplicative_outflow_strength=2.0,
            multiplicative_beta_min=0.2,
            multiplicative_beta_max=4.0,
            adaptive_inflow_prior=True,
            adaptive_outflow_prior=True,
            activity_source="max_io",
            activity_window=7,
            activity_eps=0.5,
            inflow_weight_min_scale=0.25,
            inflow_weight_max_scale=4.0,
            outflow_weight_min_scale=0.25,
            outflow_weight_max_scale=4.0,
            smooth_in=0.0,
            smooth_out=0.0,
        )
    elif preset == "trust_outflow":
        cfg = replace(make_reconcile_config("default"), w_in=1.0, w_out=100.0)
    elif preset == "trust_inflow":
        cfg = replace(make_reconcile_config("default"), w_in=100.0, w_out=1.0)
    elif preset == "balanced":
        cfg = ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=1.0,
            relative_inflow_error=False,
            multiplicative_inflow_prior=False,
            adaptive_inflow_prior=False,
            smooth_in=0.0,
            smooth_out=0.0,
        )
    elif preset == "aggressive_peak_fill":
        cfg = ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=100.0,
            relative_inflow_error=True,
            relative_inflow_eps=0.01,
            relative_inflow_weight_min_scale=0.5,
            relative_inflow_weight_max_scale=32.0,
            multiplicative_inflow_prior=True,
            multiplicative_inflow_strength=4.0,
            multiplicative_alpha_min=0.2,
            multiplicative_alpha_max=6.0,
            adaptive_inflow_prior=True,
            activity_source="out",
            activity_window=7,
            activity_eps=0.5,
            inflow_weight_min_scale=0.25,
            inflow_weight_max_scale=2.0,
            smooth_in=0.0,
            smooth_out=0.0,
        )
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if not overrides:
        return cfg
    return replace(cfg, **overrides)
