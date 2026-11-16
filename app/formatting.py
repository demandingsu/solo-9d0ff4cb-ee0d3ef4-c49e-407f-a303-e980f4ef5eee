"""Formatting helpers for templates.

Fractions are stored exactly; this module renders them as decimals only at
the very edge (HTML/JSON).  Display rounding never feeds back into the
computation.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction

from .inputs import fraction_to_text

DISPLAY_PLACES = 8


def frac(value: Fraction | str) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


def _digits(rational: Fraction) -> int:
    return len(str(abs(rational.numerator))) + len(str(rational.denominator))


def decimal(value: Fraction | str, places: int = DISPLAY_PLACES) -> str:
    """Render an exact rational as a decimal string for display.

    Normal magnitudes use a fixed-point rendering rounded to ``places``
    digits.  Values far outside that window are shown in scientific
    notation so a legitimately huge input does not flood the page; the
    exact value is still available in the JSON API / database payload.
    """

    rational = frac(value)
    if rational == 0:
        return "0"

    with localcontext() as ctx:
        ctx.prec = _digits(rational) + places + 8
        numeric = Decimal(rational.numerator) / Decimal(rational.denominator)
        exponent = numeric.adjusted() if numeric != 0 else 0
        if not -6 <= exponent <= 15:
            return f"{numeric:.{max(places, 8)}e}"
        quantum = Decimal(1).scaleb(-places)
        rounded = numeric.quantize(quantum, rounding=ROUND_HALF_UP)

    if rounded == 0:
        return f"{numeric:.{places}e}"

    text = format(rounded, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def percent(value: Fraction | str, places: int = 4) -> str:
    return decimal(frac(value) * 100, places)


def signed(value: Fraction | str, places: int = 6) -> str:
    rational = frac(value)
    text = decimal(rational, places)
    if rational > 0 and not text.startswith("-"):
        return "+" + text
    return text


def exact(value: Fraction | str) -> str:
    rational = frac(value)
    if rational.denominator == 1:
        return str(rational.numerator)
    return f"{decimal(rational, 12)} (= {fraction_to_text(rational)})"


def ratio_text(value: Fraction | str) -> str:
    """Weight as a percentage with a readable precision."""

    return percent(value, 4) + "%"
