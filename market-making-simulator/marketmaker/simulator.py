from dataclasses import dataclass

import numpy as np

from .params import MarketParams


@dataclass
class SimulationResult:
    strategy: str
    mid: np.ndarray        # (n_steps + 1, n_paths)
    bid: np.ndarray        # (n_steps, n_paths)
    ask: np.ndarray        # (n_steps, n_paths)
    inventory: np.ndarray  # (n_steps + 1, n_paths)
    pnl: np.ndarray        # (n_steps + 1, n_paths), marked to mid

    @property
    def final_pnl(self):
        return self.pnl[-1]

    @property
    def final_inventory(self):
        return self.inventory[-1]

    def summary(self) -> dict:
        pnl = self.final_pnl
        return {
            "strategy": self.strategy,
            "mean_pnl": float(pnl.mean()),
            "std_pnl": float(pnl.std()),
            "sharpe": float(pnl.mean() / pnl.std()),
            "mean_abs_final_inventory": float(np.abs(self.final_inventory).mean()),
            "std_final_inventory": float(self.final_inventory.std()),
        }


def simulate(strategy, p: MarketParams = MarketParams(), n_paths: int = 1000, seed: int = 0) -> SimulationResult:
    """Monte Carlo simulation of a market maker, vectorised across paths.

    Each step:
      1. Strategy posts bid/ask given mid-price s, inventory q and time t.
      2. Market orders hit each side with probability  A * exp(-k * distance) * dt.
      3. Cash and inventory update; mid-price moves by sigma * sqrt(dt) * Z.
    """
    rng = np.random.default_rng(seed)
    n = p.n_steps

    mid = np.empty((n + 1, n_paths))
    inv = np.empty((n + 1, n_paths))
    pnl = np.empty((n + 1, n_paths))
    bids = np.empty((n, n_paths))
    asks = np.empty((n, n_paths))

    s = np.full(n_paths, p.s0)
    q = np.zeros(n_paths)
    cash = np.zeros(n_paths)
    mid[0], inv[0], pnl[0] = s, q, 0.0

    for i in range(n):
        t = i * p.dt
        bid, ask = strategy.quotes(s, q, t, p)
        bids[i], asks[i] = bid, ask

        prob_bid = np.clip(p.A * np.exp(-p.k * (s - bid)) * p.dt, 0, 1)
        prob_ask = np.clip(p.A * np.exp(-p.k * (ask - s)) * p.dt, 0, 1)
        buy = rng.random(n_paths) < prob_bid    # our bid gets hit -> we buy
        sell = rng.random(n_paths) < prob_ask   # our ask gets lifted -> we sell

        q = q + buy - sell
        cash = cash - buy * bid + sell * ask
        s = s + p.sigma * np.sqrt(p.dt) * rng.standard_normal(n_paths)

        mid[i + 1], inv[i + 1] = s, q
        pnl[i + 1] = cash + q * s

    return SimulationResult(strategy.name, mid, bids, asks, inv, pnl)
