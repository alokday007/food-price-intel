"""SARIMA forecaster for the OVERALL FFPI series (SPEC §6, Phase 4).

This is the first *real* model — it must be judged against the bar Phase 3 set on
the exact same walk-forward harness: naive_1 (random walk) MAE=1.85 at H=1. To
keep that comparison valid, this module changes only the forecaster; it reuses
``walk_forward`` / ``metrics_by_horizon`` from ``baseline.py`` unchanged.

No-leakage is the crux (SPEC §6, §15). ``SarimaForecaster`` obeys the same
forecaster contract as the baselines — ``f(values, cutoff_idx, h) -> float`` that
may read ``values`` only at positions ``<= cutoff_idx`` — by fitting SARIMAX on
``values[: cutoff_idx + 1]`` alone and forecasting forward. The model is re-fit
every fold; it is never fit once on the full series.

Caching: within one walk-forward cutoff the harness calls the forecaster twice
(H=1 and H=3). We fit once per cutoff and serve both horizons from the cached
forecast. Crucially the cache key includes the *full array contents*, not just the
cutoff — so the leakage probe (which mutates post-cutoff values) forces a genuine
re-fit rather than a stale cache hit, and a real leak cannot hide behind the cache.
"""

from __future__ import annotations

import warnings

import numpy as np

# statsmodels is a heavier install; it is pinned in requirements.txt and already
# present in the image since Phase 0. Import lazily-friendly at module load.
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Documented starting point: non-seasonal (1,1,1), seasonal (1,1,1) at period 12.
DEFAULT_ORDER = (1, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 1, 1, 12)


class SarimaForecaster:
    """Re-fit-every-fold SARIMA forecaster matching the baseline contract.

    Call signature ``(values, cutoff_idx, h) -> float`` returns the SARIMA forecast
    for the month at position ``cutoff_idx + h``, fitting only on the training slice
    ``values[: cutoff_idx + 1]``.
    """

    def __init__(
        self,
        order=DEFAULT_ORDER,
        seasonal_order=DEFAULT_SEASONAL_ORDER,
        max_horizon: int = 3,
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.max_horizon = max_horizon
        # key -> np.ndarray of forecasts (length max_horizon)
        self._cache: dict[tuple, np.ndarray] = {}

    def _forecast_vector(self, values: np.ndarray, cutoff_idx: int) -> np.ndarray:
        # Key on cutoff AND full array bytes: identical across H=1/H=3 in a normal
        # fold (one fit), but distinct once the leakage probe mutates the future.
        key = (cutoff_idx, values.tobytes())
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        train = values[: cutoff_idx + 1]  # <-- only data at or before the cutoff
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # convergence/frequency chatter
            model = SARIMAX(
                train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            res = model.fit(disp=False)
            fc = np.asarray(res.forecast(steps=self.max_horizon), dtype=float)

        self._cache[key] = fc
        return fc

    def __call__(self, values: np.ndarray, cutoff_idx: int, h: int) -> float:
        if h > self.max_horizon:
            raise ValueError(f"h={h} exceeds max_horizon={self.max_horizon}")
        fc = self._forecast_vector(values, cutoff_idx)
        return float(fc[h - 1])
