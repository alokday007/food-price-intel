"""No-look-ahead leakage tests for the baseline forecasters (SPEC §6, §12, §15).

WHY THIS TEST MATTERS
---------------------
Look-ahead leakage is the subtle bug that silently *invalidates* a time-series
model: if a prediction for a future month is allowed to peek at data from that
month (or later), backtest error collapses toward zero and the model looks
brilliant while being useless in production. It leaves no stack trace — the code
runs fine and the metrics just lie. That is the same class of trap as the
phase-denominator bug called out in the SPEC.

So we assert leakage-freedom mechanically instead of trusting a code review:
for every walk-forward step, we violently corrupt every value *after* that step's
cutoff and require the prediction to be byte-for-byte unchanged. A prediction for
``cutoff+h`` is *defined* to be a function only of data at or before ``cutoff``;
if mutating the future moves the prediction, the forecaster leaked.

These tests are pure numpy/pandas — no database — so they pin the forecasting
core directly. ``test_planted_leak_is_caught`` proves the detector actually bites:
a deliberately leaky forecaster must fail the same check the real ones pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from apps.forecasting.baseline import (
    naive_last_predict,
    prediction_is_leak_free,
    seasonal_naive_predict,
    walk_forward,
)
from apps.forecasting.sarima import SarimaForecaster

HORIZONS = (1, 3)
INITIAL_TRAIN_END = pd.Timestamp("2015-12-01")


def _synthetic_series(n_months: int = 180) -> pd.Series:
    """A trend + annual-seasonal + noise monthly series starting 2010-01."""
    idx = pd.date_range("2010-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    rng = np.random.default_rng(42)
    values = 100 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.5, n_months)
    return pd.Series(values, index=idx, name="OVERALL")


@pytest.mark.parametrize(
    "forecaster",
    [seasonal_naive_predict, naive_last_predict],
    ids=["seasonal_naive", "naive_1"],
)
def test_no_leakage_across_all_walk_forward_steps(forecaster):
    """Every step's prediction must ignore all data after that step's cutoff."""
    series = _synthetic_series()
    values = series.to_numpy(dtype=float)
    periods = series.index

    steps = walk_forward(series, forecaster, HORIZONS, INITIAL_TRAIN_END)
    assert steps, "walk_forward produced no steps to check"

    for step in steps:
        cutoff_idx = periods.get_loc(step.cutoff)
        assert prediction_is_leak_free(values, forecaster, cutoff_idx, step.horizon), (
            f"LEAK: prediction for cutoff {step.cutoff:%Y-%m} h={step.horizon} "
            f"changed when post-cutoff data was mutated"
        )


def test_planted_leak_is_caught():
    """A forecaster that peeks one month ahead MUST fail the leakage probe.

    This guards the guard: it proves ``prediction_is_leak_free`` returns False on
    a genuine leak, so a green result on the real forecasters is meaningful and
    not a detector that can never fail.
    """

    def leaky_peek_ahead(values, cutoff_idx, h):
        # Reads position cutoff+h — i.e. the answer itself. This is the exact bug
        # the test exists to catch: using the target month's own value.
        return float(values[cutoff_idx + h])

    series = _synthetic_series()
    values = series.to_numpy(dtype=float)

    # Somewhere well inside the series so cutoff+h stays in range.
    assert not prediction_is_leak_free(values, leaky_peek_ahead, cutoff_idx=100, h=1), (
        "leakage probe failed to catch a forecaster that reads the target month"
    )


# --- SARIMA (Phase 4) -------------------------------------------------------
# SARIMA is the first model that *fits* per fold, so leakage-freedom is no longer
# obvious by inspection: it depends on the forecaster training on the pre-cutoff
# slice only. We fit a real SARIMAX here (short series + a few cutoffs to stay
# fast) and require the same probe to pass: corrupting the post-cutoff future must
# not move a prediction, i.e. the fit never touched it.


def _short_series(n_months: int = 84) -> pd.Series:
    """Shorter series so per-fold SARIMAX fits keep the test fast (starts 2010-01)."""
    idx = pd.date_range("2010-01-01", periods=n_months, freq="MS")
    t = np.arange(n_months)
    rng = np.random.default_rng(7)
    values = 100 + 0.15 * t + 6 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.0, n_months)
    return pd.Series(values, index=idx, name="OVERALL")


@pytest.mark.parametrize("h", [1, 3])
def test_no_leakage_sarima(h):
    """Re-fit SARIMA must not let post-cutoff data influence a fold's forecast."""
    series = _short_series()
    values = series.to_numpy(dtype=float)
    forecaster = SarimaForecaster(max_horizon=3)

    # A handful of cutoffs deep enough to have a full seasonal history to fit on.
    for cutoff_idx in (60, 66, 72):
        assert prediction_is_leak_free(values, forecaster, cutoff_idx, h), (
            f"LEAK: SARIMA forecast at cutoff_idx={cutoff_idx} h={h} changed when "
            f"post-cutoff data was mutated — the fit peeked at the future"
        )
