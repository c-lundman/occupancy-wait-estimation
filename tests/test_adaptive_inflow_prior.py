import pandas as pd

from occupancy_wait_estimation import ReconcileConfig, reconcile_minute_flows


def test_adaptive_inflow_prior_reduces_quiet_period_injection() -> None:
    # Banked-like toy setup:
    # - true demand concentrated in center window
    # - inflow measured with losses
    # - outflow measured accurately
    n = 30
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=n, freq="1min")

    in_measured = [0.0] * n
    out_measured = [0.0] * n
    for k in range(10, 16):
        in_measured[k] = 3.0  # lossy inflow during bank
        out_measured[k + 2] = 5.0  # trusted outflow, shifted

    df = pd.DataFrame(
        {
            "minute_start_utc": idx,
            "in_count": in_measured,
            "out_count": out_measured,
        }
    )

    base = reconcile_minute_flows(
        df,
        ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=20.0,
            smooth_in=0.0,
            smooth_out=0.0,
            adaptive_inflow_prior=False,
        ),
    )
    adaptive = reconcile_minute_flows(
        df,
        ReconcileConfig(
            q0=0.0,
            w_in=1.0,
            w_out=20.0,
            smooth_in=0.0,
            smooth_out=0.0,
            adaptive_inflow_prior=True,
            activity_source="max_io",
            activity_window=5,
            activity_eps=0.25,
            inflow_weight_min_scale=0.25,
            inflow_weight_max_scale=6.0,
        ),
    )

    quiet_mask = pd.Series(True, index=range(n))
    quiet_mask.iloc[8:20] = False

    base_extra = (
        base.loc[quiet_mask, "in_count_corrected"] - base.loc[quiet_mask, "in_count_measured"]
    ).clip(lower=0.0).sum()
    adaptive_extra = (
        adaptive.loc[quiet_mask, "in_count_corrected"] - adaptive.loc[quiet_mask, "in_count_measured"]
    ).clip(lower=0.0).sum()

    assert adaptive_extra < base_extra

