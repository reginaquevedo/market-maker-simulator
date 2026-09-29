from dataclasses import dataclass


@dataclass(frozen=True)
class MarketParams:
    """Parameters of the market and the market maker.

    s0     : initial mid-price
    sigma  : volatility of the mid-price (arithmetic Brownian motion)
    T      : trading horizon
    dt     : time step
    gamma  : risk aversion of the market maker
    k      : order-book liquidity (how fast fill probability decays with distance)
    A      : baseline order arrival intensity
    """
    s0: float = 100.0
    sigma: float = 2.0
    T: float = 1.0
    dt: float = 0.005
    gamma: float = 0.1
    k: float = 1.5
    A: float = 140.0

    @property
    def n_steps(self) -> int:
        return int(round(self.T / self.dt))
