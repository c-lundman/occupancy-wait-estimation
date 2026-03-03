from kff_v2 import make_reconcile_config


def test_trust_outflow_preset_biases_outflow_weight() -> None:
    cfg = make_reconcile_config("trust_outflow")
    assert cfg.w_out > cfg.w_in
    assert cfg.w_out == 100.0
    assert cfg.activity_source == "out"


def test_preset_allows_overrides() -> None:
    cfg = make_reconcile_config("trust_outflow", w_in=2.0, w_out=50.0)
    assert cfg.w_in == 2.0
    assert cfg.w_out == 50.0


def test_default_preset_keeps_relative_and_multiplicative_priors() -> None:
    cfg = make_reconcile_config("default")
    assert cfg.relative_inflow_error
    assert cfg.multiplicative_inflow_prior
