import numpy as np

from .params import MarketParams


class AvellanedaStoikov:
    """Inventory-aware market maker (Avellaneda & Stoikov, 2008).

    Reservation price:  r = s - q * gamma * sigma^2 * (T - t)
    Optimal spread:     delta = gamma * sigma^2 * (T - t) + (2 / gamma) * ln(1 + gamma / k)

    When the maker is long (q > 0), r drops below the mid, so quotes shift
    down: the maker sells more easily and buys less, pulling inventory to 0.
    """

    name = "Avellaneda-Stoikov"

    def reservation_price(self, s, q, t, p: MarketParams):
        return s - q * p.gamma * p.sigma**2 * (p.T - t)

    def spread(self, t, p: MarketParams):
        return p.gamma * p.sigma**2 * (p.T - t) + (2 / p.gamma) * np.log(1 + p.gamma / p.k)

    def quotes(self, s, q, t, p: MarketParams):
        r = self.reservation_price(s, q, t, p)
        half = self.spread(t, p) / 2
        return r - half, r + half


class SymmetricStrategy:
    """Benchmark: same spread as A-S, but always centered on the mid-price.

    Ignores inventory, so any difference vs. A-S isolates the value of
    inventory management.
    """

    name = "Symmetric"

    def __init__(self):
        self._as = AvellanedaStoikov()

    def quotes(self, s, q, t, p: MarketParams):
        half = self._as.spread(t, p) / 2
        return s - half, s + half
