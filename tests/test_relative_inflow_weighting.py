import pandas as pd

from occupancy_wait_estimation import ReconcileConfig, reconcile_minute_flows


def test_relative_inflow_weighting_discourages_offpeak_adjustment() -> None:
    # Two bins with same required correction pressure from trusted outflow:
    # low-measured inflow bin should get less added inflow than high-measured bin.
    df = pd.DataFrame(
        {
            "minute_start_utc": pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1min"),
            "in_count": [0.0, 10.0, 0.0, 0.0],
            "out_count": [0.0, 0.0, 5.0, 5.0],
        }
    )

    base = reconcile_minute_flows(
        df,
        ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=20.0,
            relative_inflow_error=False,
            adaptive_inflow_prior=False,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )
    rel = reconcile_minute_flows(
        df,
        ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=20.0,
            relative_inflow_error=True,
            relative_inflow_eps=1.0,
            relative_inflow_weight_min_scale=0.25,
            relative_inflow_weight_max_scale=16.0,
            adaptive_inflow_prior=False,
            smooth_in=0.0,
            smooth_out=0.0,
        ),
    )

    # In the zero-measured first minute, relative weighting should reduce added inflow.
    base_add0 = float(base.loc[0, "in_count_corrected"] - base.loc[0, "in_count_measured"])
    rel_add0 = float(rel.loc[0, "in_count_corrected"] - rel.loc[0, "in_count_measured"])
    assert rel_add0 < base_add0

