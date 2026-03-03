import numpy as np
import pandas as pd

from occupancy_wait_estimation import ReconcileConfig, reconcile_minute_flows


def test_relative_outflow_weighting_discourages_low_flow_outflow_adjustment() -> None:
    # Early outflow is inconsistent with trusted inflow and must be corrected down.
    # Relative outflow weighting should avoid correcting low-measured bins as much.
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1min"),
            "in_count": [0.0, 2.0, 8.0, 0.0],
            "out_count": [1.0, 9.0, 0.0, 0.0],
        }
    )

    base = reconcile_minute_flows(
        df,
        ReconcileConfig(
            q0=0.0,
            w_in=20.0,
            w_out=1.0,
            relative_outflow_error=False,
            adaptive_outflow_prior=False,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )
    rel = reconcile_minute_flows(
        df,
        ReconcileConfig(
            q0=0.0,
            w_in=20.0,
            w_out=1.0,
            relative_outflow_error=True,
            relative_outflow_eps=1.0,
            relative_outflow_weight_min_scale=0.25,
            relative_outflow_weight_max_scale=16.0,
            adaptive_outflow_prior=False,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )

    base_drop1 = float(base.loc[1, "out_count_measured"] - base.loc[1, "out_count_corrected"])
    rel_drop1 = float(rel.loc[1, "out_count_measured"] - rel.loc[1, "out_count_corrected"])
    assert rel_drop1 > base_drop1


def test_multiplicative_outflow_prior_promotes_proportional_scaling() -> None:
    measured_in = np.array([0.0, 0.0, 0.0, 2.0, 5.0, 4.0, 2.0, 1.0])
    measured_out = np.array([1.0, 3.0, 6.0, 3.0, 1.0, 0.0, 0.0, 0.0])
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
            w_in=40.0,
            w_out=1.0,
            relative_outflow_error=True,
            relative_outflow_eps=0.01,
            adaptive_outflow_prior=False,
            multiplicative_outflow_prior=False,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )
    mul = reconcile_minute_flows(
        df,
        ReconcileConfig(
            w_in=40.0,
            w_out=1.0,
            relative_outflow_error=True,
            relative_outflow_eps=0.01,
            adaptive_outflow_prior=False,
            multiplicative_outflow_prior=True,
            multiplicative_outflow_strength=8.0,
            multiplicative_beta_min=0.1,
            multiplicative_beta_max=2.0,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )

    mask = measured_out > 0
    base_ratio = base.loc[mask, "out_count_corrected"].to_numpy() / measured_out[mask]
    mul_ratio = mul.loc[mask, "out_count_corrected"].to_numpy() / measured_out[mask]

    assert float(np.std(mul_ratio)) < float(np.std(base_ratio))
    assert "outflow_beta" in mul.columns
