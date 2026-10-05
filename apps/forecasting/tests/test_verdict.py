"""The BEATS / TIES / LOSES rule (Phase 8a.1), pinned at its boundary.

The tie band is |SARIMA - naive_1| / naive_1 < 2%, strictly. These tests fix the
edges so a refactor can't quietly turn a 2.00% gap into a tie (or a 1.99% gap
into a win) and change the published scoreboard.

The "exactly 2%" cases use naive_1 MAE = 50 and a gap of 1, so the relative delta
is 1/50 — the same IEEE double as the 0.02 threshold literal, a true boundary.
"""

from __future__ import annotations

import pytest

from apps.forecasting.management.commands.evaluate_sarima import (
    TIE_REL_THRESHOLD,
    relative_delta,
    verdict,
)


def test_threshold_is_two_percent():
    assert TIE_REL_THRESHOLD == 0.02


@pytest.mark.parametrize(
    ("sarima_mae", "naive_mae", "expected"),
    [
        (49.005, 50.0, "TIES"),   # -1.99%
        (50.995, 50.0, "TIES"),   # +1.99%
        (49.0, 50.0, "BEATS"),    # exactly -2.00%
        (51.0, 50.0, "LOSES"),    # exactly +2.00%
        (40.0, 50.0, "BEATS"),    # clear win, -20%
        (60.0, 50.0, "LOSES"),    # clear loss, +20%
        (50.0, 50.0, "TIES"),     # identical
    ],
)
def test_verdict_boundaries(sarima_mae, naive_mae, expected):
    assert verdict(sarima_mae, naive_mae) == expected


def test_exact_boundary_really_is_exact():
    """Guard the guard: the 'exactly 2%' inputs must hit the threshold exactly."""
    assert abs(relative_delta(49.0, 50.0)) == TIE_REL_THRESHOLD
    assert relative_delta(51.0, 50.0) == TIE_REL_THRESHOLD
