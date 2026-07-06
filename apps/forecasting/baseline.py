"""Seasonal-naive baseline + walk-forward validation harness (SPEC §6).

This module is deliberately free of any Django import so it stays a pure,
unit-testable numeric core: it operates on numpy arrays / pandas Series only.
That matters because the leakage test (tests/test_leakage.py) exercises this
code with synthetic data and no database at all.

The forecasters defined here are the *dumb bars to beat* — no fitting, no
parameters, no state. Every future model (SARIMA, LightGBM in Phase 4) must be
shown to beat these on MAE, or be honestly reported as adding no value.

The single most important invariant enforced here is **no look-ahead leakage**
(SPEC §6, §15): a prediction for month ``cutoff + h`` may read the series only
at positions at or before ``cutoff``. Both forecasters below satisfy this by
construction, and ``prediction_is_leak_free`` lets a test prove it mechanically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# FFPI seasonality is annual: 12 monthly observations per cycle.
SEASONAL_PERIOD = 12


@dataclass(frozen=True)
class Step:
    """One walk-forward observation: a single (cutoff, horizon) prediction.

    ``cutoff`` is the last month the forecaster was allowed to see; ``target`` is
    the month being predicted (``cutoff`` + ``horizon`` months).
    """

    cutoff: pd.Timestamp
    horizon: int
    target: pd.Timestamp
    y_true: float
    y_pred: float


# ---------------------------------------------------------------------------
# Forecasters
#
# Contract for every forecaster: ``f(values, cutoff_idx, h) -> float`` returns
# the prediction for the month at position ``cutoff_idx + h``, and MUST NOT read
# ``values`` at any position strictly greater than ``cutoff_idx``. Returning NaN
# means "not enough history to predict" (the step is then skipped).
# ---------------------------------------------------------------------------


def seasonal_naive_predict(values: np.ndarray, cutoff_idx: int, h: int) -> float:
    """Seasonal-naive: forecast month ``cutoff+h`` as the value 12 months earlier.

    The target month is at position ``cutoff_idx + h``; its same-month-last-year
    value sits at ``cutoff_idx + h - 12``. For any horizon ``h <= 12`` that source
    position is ``<= cutoff_idx``, so the prediction is a function of observed
    history only — no future value is ever read.
    """
    src = cutoff_idx + h - SEASONAL_PERIOD
    if src < 0:
        return float("nan")  # not yet a full year of history before the target
    return float(values[src])


def naive_last_predict(values: np.ndarray, cutoff_idx: int, h: int) -> float:
    """Random-walk naive (``naive-1``): carry the last observed value forward.

    The last month the forecaster may see is ``cutoff_idx``; every horizon is
    predicted as that value. Reads position ``cutoff_idx`` only.
    """
    return float(values[cutoff_idx])


FORECASTERS = {
    "seasonal_naive": seasonal_naive_predict,
    "naive_1": naive_last_predict,
}


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def walk_forward(
    series: pd.Series,
    forecaster,
    horizons,
    initial_train_end: pd.Timestamp,
) -> list[Step]:
    """Expanding-window walk-forward validation (SPEC §6).

    Starting from ``initial_train_end`` as the first cutoff, and advancing the
    cutoff one month at a time to the end of the series, produce a prediction for
    each requested horizon. The training window only ever grows (expanding
    window); it is never re-cut or shuffled.

    Because each forecaster is passed the full ``values`` array together with the
    integer ``cutoff_idx``, leakage-freedom is a property of the *forecaster*, not
    of slicing here — which is exactly what the leakage test verifies.
    """
    values = series.to_numpy(dtype=float)
    periods = series.index
    n = len(values)

    if initial_train_end not in periods:
        raise ValueError(
            f"initial_train_end {initial_train_end:%Y-%m} is not in the series "
            f"(range {periods.min():%Y-%m}..{periods.max():%Y-%m})."
        )

    start_idx = periods.get_loc(initial_train_end)
    steps: list[Step] = []

    # Cutoff walks from the initial train end to the last month for which at least
    # one horizon still has a real target inside the series.
    for cutoff_idx in range(start_idx, n):
        for h in horizons:
            target_idx = cutoff_idx + h
            if target_idx >= n:
                continue  # target lies beyond observed data — nothing to score
            y_pred = forecaster(values, cutoff_idx, h)
            if np.isnan(y_pred):
                continue  # forecaster lacked enough history for this step
            steps.append(
                Step(
                    cutoff=periods[cutoff_idx],
                    horizon=int(h),
                    target=periods[target_idx],
                    y_true=float(values[target_idx]),
                    y_pred=y_pred,
                )
            )
    return steps


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / y_true)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def metrics_by_horizon(steps: list[Step]) -> dict[int, dict]:
    """Aggregate MAE / RMSE / MAPE over all walk-forward steps, per horizon."""
    out: dict[int, dict] = {}
    for h in sorted({s.horizon for s in steps}):
        rows = [s for s in steps if s.horizon == h]
        y_true = np.array([s.y_true for s in rows])
        y_pred = np.array([s.y_pred for s in rows])
        stats = _metrics(y_true, y_pred)
        stats["n"] = len(rows)
        out[h] = stats
    return out


def count_folds(steps: list[Step]) -> int:
    """Number of distinct cutoffs (expanding-window folds) that produced a step."""
    return len({s.cutoff for s in steps})


# ---------------------------------------------------------------------------
# Leakage probe (used by the mandatory leakage test)
# ---------------------------------------------------------------------------


def prediction_is_leak_free(
    values: np.ndarray,
    forecaster,
    cutoff_idx: int,
    h: int,
    perturbation: float = 1.0e6,
) -> bool:
    """Return True iff ``forecaster``'s prediction ignores all post-cutoff data.

    Method: compute the prediction, then violently corrupt every value *strictly
    after* the cutoff and recompute. A leak-free forecaster must return the exact
    same number, because a prediction for ``cutoff+h`` is defined to depend only
    on data at or before ``cutoff``. If mutating the future changes the answer,
    the forecaster peeked ahead — this returns False.

    This is the mechanical core of the leakage test: it converts "no look-ahead"
    from a code-review claim into an assertion a machine can fail on.
    """
    base = forecaster(values, cutoff_idx, h)
    mutated = np.array(values, dtype=float)
    mutated[cutoff_idx + 1 :] += perturbation
    after = forecaster(mutated, cutoff_idx, h)
    if np.isnan(base) and np.isnan(after):
        return True
    return base == after
