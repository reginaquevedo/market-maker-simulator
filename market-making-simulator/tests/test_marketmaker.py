from dataclasses import replace

import numpy as np

from marketmaker import MarketParams, AvellanedaStoikov, SymmetricStrategy, simulate

P = MarketParams()


def test_spread_is_positive_and_shrinks_over_time():
    s = AvellanedaStoikov()
    assert s.spread(0.0, P) > s.spread(0.99, P) > 0


def test_long_inventory_skews_quotes_down():
    s = AvellanedaStoikov()
    bid0, ask0 = s.quotes(np.array([100.0]), np.array([0.0]), 0.0, P)
    bid_long, ask_long = s.quotes(np.array([100.0]), np.array([5.0]), 0.0, P)
    assert bid_long < bid0 and ask_long < ask0


def test_symmetric_quotes_centered_on_mid():
    bid, ask = SymmetricStrategy().quotes(np.array([100.0]), np.array([7.0]), 0.3, P)
    assert np.isclose((bid + ask) / 2, 100.0)


def test_no_order_flow_means_no_trades_and_zero_pnl():
    res = simulate(AvellanedaStoikov(), replace(P, A=0.0), n_paths=50, seed=0)
    assert np.all(res.inventory == 0)
    assert np.allclose(res.pnl, 0)


def test_simulation_is_reproducible():
    a = simulate(AvellanedaStoikov(), P, n_paths=20, seed=3).final_pnl
    b = simulate(AvellanedaStoikov(), P, n_paths=20, seed=3).final_pnl
    assert np.array_equal(a, b)


def test_inventory_control_reduces_risk():
    res_as = simulate(AvellanedaStoikov(), P, n_paths=500, seed=0)
    res_sym = simulate(SymmetricStrategy(), P, n_paths=500, seed=0)
    assert res_as.final_pnl.std() < res_sym.final_pnl.std()
    assert np.abs(res_as.final_inventory).mean() < np.abs(res_sym.final_inventory).mean()
