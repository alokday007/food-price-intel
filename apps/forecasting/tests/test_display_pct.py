"""Displayed percentages must read exactly as the live site shows them.

The site renders ``(Math.abs(rel_delta) * 100).toFixed(1)`` in the browser.
``_site_to_fixed_1`` below re-implements that from the ECMAScript definition of
``Number.prototype.toFixed`` — "let n be an integer for which n / 10^f - x is as
close to zero as possible; if there are two such n, pick the larger n" — in
exact rational arithmetic. It deliberately shares no code with
``format_delta_pct`` (Fraction here, Decimal there), so the README check below
is a comparison against the site's rule, not against the helper itself.
"""

from __future__ import annotations

import json
import math
import re
from fractions import Fraction

import pytest

from apps.forecasting.display import format_delta_pct
from apps.forecasting.management.commands.render_scoreboard import (
    END,
    HORIZON_WORDS,
    README_PATH,
    START,
)
from apps.forecasting.views import SCOREBOARD_PATH, SERIES_LABELS


def _site_to_fixed_1(rel_delta: float) -> str:
    """What the site prints for this stored rel_delta (magnitude, no sign/%)."""
    x = Fraction(abs(rel_delta) * 100)  # same double the browser computes, exactly
    n = math.floor(x * 10 + Fraction(1, 2))  # nearest n; exact tie -> larger n
    return f"{n // 10}.{n % 10}"


@pytest.mark.parametrize(
    "rel_delta, expected",
    [
        (-0.0225, "-2.3%"),  # Vegetable Oils H=1: Python's :.1% says -2.2%
        (0.0225, "+2.3%"),
        (-0.07564, "-7.6%"),
        (0.25, "+25.0%"),
        (-0.2, "-20.0%"),
        (0.0, "+0.0%"),
    ],
)
def test_format_delta_pct(rel_delta, expected):
    assert format_delta_pct(rel_delta) == expected
    assert format_delta_pct(rel_delta)[1:-1] == _site_to_fixed_1(rel_delta)


def test_reference_reproduces_the_python_vs_site_discrepancy():
    """Pin the case that motivated the helper: the two rounding rules disagree."""
    assert f"{0.0225:.1%}" == "2.2%"
    assert _site_to_fixed_1(0.0225) == "2.3"


def _readme_rows() -> list[tuple[str, str, str]]:
    """(series label, horizon words, Δ cell) for every row of the README table."""
    readme = README_PATH.read_text(encoding="utf-8")
    block = readme.split(START, 1)[1].split(END, 1)[0]
    rows = []
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 7 and re.fullmatch(r"[+-]\d+\.\d%", cells[5]):
            rows.append((cells[0], cells[1], cells[5]))
    return rows


def test_every_readme_percentage_matches_the_site():
    scoreboard = json.loads(SCOREBOARD_PATH.read_text(encoding="utf-8"))
    expected = {}
    for code, horizons in scoreboard["groups"].items():
        for h, row in horizons.items():
            expected[(SERIES_LABELS.get(code, code), HORIZON_WORDS.get(h, h))] = row["rel_delta"]

    rows = _readme_rows()
    assert len(rows) == len(expected), f"README has {len(rows)} delta rows, scoreboard {len(expected)}"
    for label, horizon, cell in rows:
        rel_delta = expected[(label, horizon)]
        sign = "-" if rel_delta < 0 else "+"
        assert cell == f"{sign}{_site_to_fixed_1(rel_delta)}%", (
            f"{label} {horizon}: README shows {cell}, site shows "
            f"{_site_to_fixed_1(rel_delta)}% for rel_delta {rel_delta}"
        )
