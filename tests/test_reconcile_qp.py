from pathlib import Path

import pandas as pd

from occupancy_wait_estimation import ReconcileConfig, reconcile_minute_flows


REPO_ROOT = Path(__file__).resolve().parents[1]
LOSSY_DAY = REPO_ROOT / "data" / "synthetic" / "lossy" / "day_2026-01-15"


def test_reconcile_removes_negative_occupancy() -> None:
    path = LOSSY_DAY / "asymmetric_inflow_loss" / "minute_flows.csv"
    df = pd.read_csv(path)

    # Outflow is considered more reliable here, so penalize its correction more.
    cfg = ReconcileConfig(q0=0.0, w_in=1.0, w_out=12.0, smooth_in=0.05, smooth_out=0.05)
    result = reconcile_minute_flows(df, config=cfg)

    assert (result["occupancy_corrected_end"] >= -1e-6).all()


def test_reconcile_prefers_adjusting_inflow_when_outflow_trusted() -> None:
    path = LOSSY_DAY / "mild_noise" / "minute_flows.csv"
    df = pd.read_csv(path)
    cfg = ReconcileConfig(q0=0.0, w_in=1.0, w_out=10.0)
    result = reconcile_minute_flows(df, config=cfg)

    in_delta = (result["in_count_corrected"] - result["in_count_measured"]).abs().sum()
    out_delta = (result["out_count_corrected"] - result["out_count_measured"]).abs().sum()
    assert in_delta > out_delta


def test_reconcile_preserves_timeline_length() -> None:
    path = LOSSY_DAY / "spurious_outflow" / "minute_flows.csv"
    df = pd.read_csv(path)
    result = reconcile_minute_flows(df)
    assert len(result) == len(df)


def test_reconciled_occupancy_matches_flow_balance() -> None:
    path = LOSSY_DAY / "mixed_heavy_noise" / "minute_flows.csv"
    df = pd.read_csv(path)
    cfg = ReconcileConfig(q0=0.0)
    result = reconcile_minute_flows(df, config=cfg)

    lhs = result["occupancy_corrected_end"].to_numpy()
    rhs = (
        result["in_count_corrected"].to_numpy()
        - result["out_count_corrected"].to_numpy()
    ).cumsum()
    assert abs(lhs - rhs).max() < 5e-4
