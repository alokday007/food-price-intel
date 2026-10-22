"""LightGBM forecasters for the FFPI series (Phase 9a — evaluation only).

Two variants, both judged on the *unchanged* walk-forward harness from
``baseline.py`` (same cutoffs, same horizons, same MAE) so their numbers are
directly comparable with naive_1 and SARIMA:

* ``LgbmSeriesForecaster`` — one model per series, trained on that series only.
* ``LgbmPooledForecaster`` — one model trained on all six series together, with
  the series code as a categorical feature.

Design (fixed up front, documented, NOT tuned):

* TARGET is the change ``y[t+h] - y[t]``, never the level. Trees cannot
  extrapolate beyond the training range, and the FFPI regularly sets new highs;
  predicting the change sidesteps that. Forecast = ``y[t]`` + predicted change.
  naive_1 is exactly "predicted change = 0", so the model only has to learn the
  deviation from the baseline — and if it learns nothing, it degrades to naive_1
  rather than to something worse.
* DIRECT multi-step: a separate model per horizon (H=1, H=3), each trained on
  its own target. No recursive feeding of predictions back in.
* FEATURES for time ``t`` use only values at or before ``t`` (``build_features``):
  lags 1..12 of the monthly change, rolling mean/std of the change over 3/6/12
  months, month of year, and the level relative to its trailing 12-month mean.
  No cross-series features in 9a (pooling shares *model parameters*, not inputs).
* HYPERPARAMETERS are the single ``LGBM_PARAMS`` dict below — no search, no
  per-series tuning, no early stopping on evaluation folds.

No-leakage contract (SPEC §6, §15), identical to the baselines: a forecaster
``f(values, cutoff_idx, h)`` may read ``values`` only at positions
``<= cutoff_idx``. Both variants enforce it structurally by slicing *first*
(``values[: cutoff_idx + 1]``) and building every feature and every training
label from that slice alone. Training rows are only those ``t`` whose label
month ``t + h`` is itself ``<= cutoff`` — a label from after the cutoff would
be a leak even if every feature were clean.

The pooled variant additionally reads the *other* five series. It receives the
full panel but slices each one at the fold's cutoff date (the date of
``cutoff_idx`` in the target series) before doing anything with it, so no other
series' observation after the cutoff can ever reach training or prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# Fixed hyperparameters for every series, both variants and both horizons.
# Chosen once as conventional small-data defaults (shallow trees, modest learning
# rate, row/column subsampling for variance reduction) — NOT searched or tuned on
# the evaluation folds. deterministic + force_row_wise + n_jobs=1 + a fixed seed
# make every fit bit-for-bit reproducible (asserted by the determinism test).
LGBM_PARAMS = {
    "objective": "regression",  # L2 on the change; MAE is only the report metric
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,  # bagging only activates with a non-zero frequency
    "colsample_bytree": 0.8,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
    "verbose": -1,
}

LAGS = tuple(range(1, 13))
ROLL_WINDOWS = (3, 6, 12)

# Rows before this position lack a full feature vector: chg_lag12 at t is
# y[t-11] - y[t-12], and the 12-month rolling stats also reach back to y[t-12].
MIN_HISTORY = 12

FEATURES = (
    [f"chg_lag{k}" for k in LAGS]
    + [f"chg_mean{w}" for w in ROLL_WINDOWS]
    + [f"chg_std{w}" for w in ROLL_WINDOWS]
    + ["month", "level_rel_12m"]
)
SERIES_FEATURE = "series"


def build_features(values: np.ndarray, periods: pd.DatetimeIndex) -> pd.DataFrame:
    """Feature matrix, one row per position ``t``, using only ``values[<= t]``.

    Let ``chg[t] = y[t] - y[t-1]`` (the monthly change ending at ``t``):

    * ``chg_lag{k}`` = ``chg[t-k+1]`` — so ``chg_lag1`` is the latest change
      (``y[t] - y[t-1]``) and ``chg_lag12`` reaches back to ``y[t-12]``.
    * ``chg_mean{w}`` / ``chg_std{w}`` — trailing mean / std of ``chg`` over the
      ``w`` months ending at ``t`` (inclusive).
    * ``month`` — calendar month of ``t`` (the target month is ``t + h``, a fixed
      offset, so the direct model per horizon can learn seasonality from this).
    * ``level_rel_12m`` — ``y[t] / mean(y[t-11..t]) - 1``.

    Every operation is a backward-looking shift or trailing window, so row ``t``
    is unchanged by anything at positions ``> t`` (pinned by the feature test).
    Rows ``t < MIN_HISTORY`` contain NaNs and are dropped by the callers.
    """
    if len(values) != len(periods):
        raise ValueError(f"values ({len(values)}) and periods ({len(periods)}) differ in length")
    y = pd.Series(np.asarray(values, dtype=float))
    chg = y.diff()
    cols = {f"chg_lag{k}": chg.shift(k - 1) for k in LAGS}
    for w in ROLL_WINDOWS:
        cols[f"chg_mean{w}"] = chg.rolling(w).mean()
    for w in ROLL_WINDOWS:
        cols[f"chg_std{w}"] = chg.rolling(w).std()
    cols["month"] = pd.Series(periods.month, dtype=float)
    cols["level_rel_12m"] = y / y.rolling(12).mean() - 1.0
    return pd.DataFrame(cols, columns=FEATURES)


def _training_frame(train: np.ndarray, periods: pd.DatetimeIndex, h: int):
    """(X, y_change, X_at_cutoff) built from one already-truncated training slice.

    ``train`` ends at the cutoff. Labels are ``train[t+h] - train[t]``, which is
    NaN for the last ``h`` rows (their target month is after the cutoff), so
    those rows are dropped — only labels observed by the cutoff are learned from.
    """
    feats = build_features(train, periods)
    label = pd.Series(train).shift(-h) - pd.Series(train)
    rows = feats.index >= MIN_HISTORY
    rows &= label.notna().to_numpy()
    return feats.loc[rows], label.loc[rows], feats.iloc[[-1]]


def _fit(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)
    return model


class LgbmSeriesForecaster:
    """Per-series direct LightGBM; obeys ``f(values, cutoff_idx, h) -> float``.

    ``periods`` is the target series' date index (needed only for the month
    feature). One fresh model per (cutoff, horizon), trained on
    ``values[: cutoff_idx + 1]`` alone. No cache: the harness asks for each
    (cutoff, h) exactly once, and without a cache the leakage probe always
    forces a genuine re-fit.
    """

    name = "lgbm_series"

    def __init__(self, periods: pd.DatetimeIndex):
        self.periods = periods

    def __call__(self, values: np.ndarray, cutoff_idx: int, h: int) -> float:
        if len(values) != len(self.periods):
            raise ValueError("values do not match this forecaster's periods")
        train = np.asarray(values[: cutoff_idx + 1], dtype=float)  # <-- the only read
        X, y, x_now = _training_frame(train, self.periods[: cutoff_idx + 1], h)
        if len(X) < LGBM_PARAMS["min_child_samples"]:
            return float("nan")
        model = _fit(X, y)
        return float(train[-1] + model.predict(x_now)[0])


class LgbmPooledForecaster:
    """One direct LightGBM per (cutoff, horizon) trained on ALL series together.

    Built once from the full ``panel`` ({code: date-indexed Series}); call
    ``for_series(code)`` to get the harness-contract callable for one target.

    For a fold with cutoff date ``c`` (the date at ``cutoff_idx`` in the target):
    the target contributes ``values[: cutoff_idx + 1]`` (the array the harness
    passed, so a planted-future probe on it is honoured) and every other series
    contributes ``panel[code].loc[:c]``. Nothing after ``c`` is ever read.

    Because those truncated slices are identical whichever series is the
    target, the model for a given (cutoff, h) is fit once and reused across all
    six targets. The cache key is the exact bytes of every slice used, so a
    changed input always means a re-fit. (Leakage tests still use a fresh
    instance for the perturbed run, so a cache can never mask a leak.)
    """

    name = "lgbm_pooled"

    def __init__(self, panel: dict[str, pd.Series]):
        self.panel = panel
        self.codes = sorted(panel)
        self._models: dict[tuple, LGBMRegressor] = {}
        # Total split gain per horizon, summed over every distinct fitted model.
        self.gain: dict[int, np.ndarray] = {}
        self.n_fits = 0

    def _other_series_upto(self, code: str, cutoff: pd.Timestamp) -> pd.Series:
        """Another series truncated at the fold's cutoff date (inclusive)."""
        return self.panel[code].loc[:cutoff]

    def _slices(self, target: str, values: np.ndarray, cutoff_idx: int):
        periods = self.panel[target].index
        cutoff = periods[cutoff_idx]
        out = {}
        for code in self.codes:
            if code == target:
                out[code] = (np.asarray(values[: cutoff_idx + 1], dtype=float), periods[: cutoff_idx + 1])
            else:
                s = self._other_series_upto(code, cutoff)
                out[code] = (s.to_numpy(dtype=float), s.index)
        return out

    def _with_series(self, X: pd.DataFrame, code: str) -> pd.DataFrame:
        X = X.copy()
        X[SERIES_FEATURE] = pd.Categorical([code] * len(X), categories=self.codes)
        return X

    def _model(self, slices: dict, h: int) -> LGBMRegressor:
        key = (h,) + tuple(
            (code, arr.tobytes(), idx[-1].value if len(idx) else None)
            for code, (arr, idx) in slices.items()
        )
        model = self._models.get(key)
        if model is not None:
            return model
        Xs, ys = [], []
        for code, (arr, idx) in slices.items():
            X, y, _ = _training_frame(arr, idx, h)
            Xs.append(self._with_series(X, code))
            ys.append(y)
        model = _fit(pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True))
        self._models[key] = model
        self.n_fits += 1
        g = model.booster_.feature_importance(importance_type="gain")
        self.gain[h] = self.gain.get(h, 0.0) + g
        return model

    def for_series(self, target: str):
        if target not in self.panel:
            raise KeyError(f"{target} not in panel {self.codes}")
        n = len(self.panel[target])

        def forecast(values: np.ndarray, cutoff_idx: int, h: int) -> float:
            if len(values) != n:
                raise ValueError(f"values do not match the {target} series in the panel")
            slices = self._slices(target, values, cutoff_idx)
            train, idx = slices[target]
            _, _, x_now = _training_frame(train, idx, h)
            model = self._model(slices, h)
            return float(train[-1] + model.predict(self._with_series(x_now, target))[0])

        return forecast

    def feature_names(self) -> list[str]:
        return FEATURES + [SERIES_FEATURE]


# ---------------------------------------------------------------------------
# Phase 9a.1 — pooled LightGBM with a scale-free target
# (pre-registered in docs/experiments/9a1_prereg.md; the 9a classes above are
# untouched). The ONLY change vs ``LgbmPooledForecaster``: the target is
# ``log(y[t+h] / y[t])`` and the ``chg_*`` features are built on monthly LOG
# changes, so no single series' scale dominates the shared L2 loss. Forecast =
# ``y[t] * exp(predicted log change)``. Params, month, level_rel_12m, series
# categorical, direct per-horizon models and slicing are identical.
# ---------------------------------------------------------------------------


def _assert_positive(values: np.ndarray) -> np.ndarray:
    y = np.asarray(values, dtype=float)
    if not np.all(y > 0):  # also rejects NaN
        raise ValueError("log-change features need strictly positive values")
    return y


def build_log_features(values: np.ndarray, periods: pd.DatetimeIndex) -> pd.DataFrame:
    """``build_features`` with ``chg[t] = log(y[t]) - log(y[t-1])``.

    Same columns, same backward-only construction: the ``chg_*`` columns are
    ``build_features`` applied to ``log(y)``; ``month`` is unchanged and
    ``level_rel_12m`` is recomputed on the raw level (it is already scale-free).
    """
    y = _assert_positive(values)
    feats = build_features(np.log(y), periods)
    level = pd.Series(y)
    feats["level_rel_12m"] = level / level.rolling(12).mean() - 1.0
    return feats


def _log_training_frame(train: np.ndarray, periods: pd.DatetimeIndex, h: int):
    """``_training_frame`` with label ``log(train[t+h] / train[t])``."""
    feats = build_log_features(train, periods)
    log_y = pd.Series(np.log(_assert_positive(train)))
    label = log_y.shift(-h) - log_y
    rows = feats.index >= MIN_HISTORY
    rows &= label.notna().to_numpy()
    return feats.loc[rows], label.loc[rows], feats.iloc[[-1]]


class LgbmPooledPctForecaster(LgbmPooledForecaster):
    """``LgbmPooledForecaster`` with the scale-free (log-change) target.

    Inherits the panel slicing (``_slices`` / ``_other_series_upto``), so the
    same cutoff-date truncation protects every series.
    """

    name = "lgbm_pooled_pct"

    def _model(self, slices: dict, h: int) -> LGBMRegressor:
        key = (h,) + tuple(
            (code, arr.tobytes(), idx[-1].value if len(idx) else None)
            for code, (arr, idx) in slices.items()
        )
        model = self._models.get(key)
        if model is not None:
            return model
        Xs, ys = [], []
        for code, (arr, idx) in slices.items():
            X, y, _ = _log_training_frame(arr, idx, h)
            Xs.append(self._with_series(X, code))
            ys.append(y)
        model = _fit(pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True))
        self._models[key] = model
        self.n_fits += 1
        g = model.booster_.feature_importance(importance_type="gain")
        self.gain[h] = self.gain.get(h, 0.0) + g
        return model

    def for_series(self, target: str):
        if target not in self.panel:
            raise KeyError(f"{target} not in panel {self.codes}")
        n = len(self.panel[target])

        def forecast(values: np.ndarray, cutoff_idx: int, h: int) -> float:
            if len(values) != n:
                raise ValueError(f"values do not match the {target} series in the panel")
            slices = self._slices(target, values, cutoff_idx)
            train, idx = slices[target]
            _, _, x_now = _log_training_frame(train, idx, h)
            model = self._model(slices, h)
            log_chg = model.predict(self._with_series(x_now, target))[0]
            return float(train[-1] * np.exp(log_chg))

        return forecast
