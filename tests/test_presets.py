from occupancy_wait_estimation import make_reconcile_config


def test_trust_outflow_preset_biases_outflow_weight() -> None:
    cfg = make_reconcile_config("trust_outflow")
    assert cfg.w_out > cfg.w_in
    assert cfg.w_out == 100.0


def test_preset_allows_overrides() -> None:
    cfg = make_reconcile_config("trust_outflow", w_in=2.0, w_out=50.0)
    assert cfg.w_in == 2.0
    assert cfg.w_out == 50.0


def test_default_preset_keeps_relative_and_multiplicative_priors() -> None:
    cfg = make_reconcile_config("default")
    assert cfg.relative_inflow_error
    assert cfg.relative_outflow_error
    assert cfg.multiplicative_inflow_prior
    assert cfg.multiplicative_outflow_prior
    assert cfg.adaptive_inflow_prior
    assert cfg.adaptive_outflow_prior


def test_trust_inflow_preset_biases_inflow_weight() -> None:
    cfg = make_reconcile_config("trust_inflow")
    assert cfg.w_in > cfg.w_out
    assert cfg.w_in == 100.0
