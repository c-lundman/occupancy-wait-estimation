from pathlib import Path

import pandas as pd

from occupancy_wait_estimation import ReconcileConfig, reconcile_minute_flows


REPO_ROOT = Path(__file__).resolve().parents[1]
PERFECT_MINUTE = REPO_ROOT / "data" / "synthetic" / "perfect" / "day_2026-01-15" / "minute_flows.csv"


def test_qp_does_not_modify_perfect_data_materially() -> None:
    df = pd.read_csv(PERFECT_MINUTE)
    inp = df[["minute_start_utc", "in_count", "out_count"]]

    rec = reconcile_minute_flows(
        inp,
        ReconcileConfig(q0=0.0, w_in=1.0, w_out=1.0, smooth_in=0.0, smooth_out=0.0),
    )

    in_delta = (rec["in_count_corrected"] - rec["in_count_measured"]).abs()
    out_delta = (rec["out_count_corrected"] - rec["out_count_measured"]).abs()
    occ_truth = (df["in_count"] - df["out_count"]).cumsum()
    occ_delta = (rec["occupancy_corrected_end"] - occ_truth).abs()

    # Solver should only introduce tiny numerical-tolerance changes.
    assert float(in_delta.sum()) < 1e-2
    assert float(out_delta.sum()) < 1e-3
    assert float(occ_delta.max()) < 1e-2
    assert int((rec["occupancy_corrected_end"] < -1e-6).sum()) == 0

