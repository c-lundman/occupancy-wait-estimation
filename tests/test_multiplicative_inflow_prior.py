import numpy as np
import pandas as pd

from kff_v2 import ReconcileConfig, reconcile_minute_flows


def test_multiplicative_inflow_prior_promotes_proportional_scaling() -> None:
    # Synthetic case where true inflow is about 2x measured inflow profile.
    measured_in = np.array([1.0, 3.0, 6.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    true_in = 2.0 * measured_in
    measured_out = np.array([0.0, 0.0, 0.0, 2.0, 5.0, 4.0, 2.0, 1.0])
    # Keep outflow trusted and slightly high so inflow must be corrected upward.
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=len(measured_in), freq="1min"),
            "in_count": measured_in,
            "out_count": measured_out,
        }
    )

    base = reconcile_minute_flows(
        df,
        ReconcileConfig(
            w_in=1.0,
            w_out=40.0,
            relative_inflow_error=True,
            relative_inflow_eps=0.01,
            adaptive_inflow_prior=False,
            multiplicative_inflow_prior=False,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )
    mul = reconcile_minute_flows(
        df,
        ReconcileConfig(
            w_in=1.0,
            w_out=40.0,
            relative_inflow_error=True,
            relative_inflow_eps=0.01,
            adaptive_inflow_prior=False,
            multiplicative_inflow_prior=True,
            multiplicative_inflow_strength=8.0,
            multiplicative_alpha_min=0.5,
            multiplicative_alpha_max=4.0,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )

    mask = measured_in > 0
    base_ratio = base.loc[mask, "in_count_corrected"].to_numpy() / measured_in[mask]
    mul_ratio = mul.loc[mask, "in_count_corrected"].to_numpy() / measured_in[mask]

    # Multiplicative prior should produce a flatter (more proportional) ratio profile.
    assert float(np.std(mul_ratio)) < float(np.std(base_ratio))
    assert "inflow_alpha" in mul.columns

