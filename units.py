"""Unit normalisation - the second half of gold case K6.

Kept out of `claimguard.py` deliberately. `structural_relation` abstains on a
unit mismatch and says why (`structural:unit-mismatch`, `'hour' vs 'minute' -
normalise before comparing`). That abstention is correct behaviour and it is
what the demo shows first. This module is the follow-up: normalise, re-run, and
watch K6 flip from UNDECIDED to CORROBORATES without touching the checker.

The ordering matters for the pitch. A safety tool that reports "4 hours
conflicts with 240 minutes" is worse than useless - it trains the operator to
dismiss it. So the sequence is:

    abstain  ->  visible, honest, no false conflict
    normalise ->  corroborates, and K1 (4h vs 6h) still conflicts

If normalisation ran silently from the start, you could not tell the difference
between a checker that understands units and one that got lucky.

## Why a whitelist and not a units library

`pint` would handle this and much more. It is also a dependency, and the
failure mode of an over-general converter here is worse than the failure mode
of an incomplete one: converting units that only look compatible ("6 doses" vs
"6 hours") invents agreement. Unknown units are left exactly as they are, so
they keep abstaining.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

# `_num` is imported rather than re-written on purpose. Normalisation and
# comparison must agree on what the number in a value string *is*; two parsers
# would eventually disagree on some edge ("1,500", "4.0") and produce a
# conflict that exists only between them.
from claimguard import Claim, ValueType, _num

# dimension -> {unit: factor to the canonical unit}
# Canonical is the smallest unit in each dimension, so conversion is exact in
# integers for every value this corpus contains and no float rounding enters
# a comparison that is supposed to be deterministic.
DIMENSIONS: dict[str, dict[str, float]] = {
    "time": {
        "minute": 1.0, "min": 1.0,
        "hour": 60.0, "hr": 60.0, "h": 60.0,
        "day": 1440.0,
        "week": 10080.0,
    },
    "mass": {
        "mg": 1.0, "milligram": 1.0,
        "g": 1000.0, "gram": 1000.0,
        "kg": 1_000_000.0, "kilogram": 1_000_000.0,
    },
    "volume": {
        "ml": 1.0, "millilitre": 1.0, "milliliter": 1.0,
        "l": 1000.0, "litre": 1000.0, "liter": 1000.0,
    },
}

CANONICAL = {"time": "minute", "mass": "mg", "volume": "ml"}

_UNIT_TO_DIM = {u: dim for dim, table in DIMENSIONS.items() for u in table}


def dimension_of(unit: str | None) -> str | None:
    return _UNIT_TO_DIM.get(unit.strip().lower()) if unit else None


def normalise_claim(claim: Claim) -> Claim:
    """Rewrite a numeric claim into its canonical unit. Non-numeric: unchanged.

    Returns the claim untouched when the unit is unknown, so an unrecognised
    unit still reaches `structural_relation` as a mismatch and still abstains.
    """
    if claim.value_type is not ValueType.NUMERIC or not claim.unit:
        return claim

    dim = dimension_of(claim.unit)
    if dim is None:
        return claim

    factor = DIMENSIONS[dim][claim.unit.strip().lower()]
    magnitude = _num(claim.value)
    if magnitude is None:
        return claim

    scaled = magnitude * factor
    # Integers stay integers: "240" reads better than "240.0" in an escalation
    # a human is meant to act on.
    text = str(int(scaled)) if scaled == int(scaled) else str(scaled)
    return replace(claim, value=text, unit=CANONICAL[dim])


def normalise(claims: Sequence[Claim]) -> list[Claim]:
    """Whole ledger. Pure - returns new claims, mutates nothing."""
    return [normalise_claim(c) for c in claims]


def describe(before: Sequence[Claim], after: Sequence[Claim]) -> list[str]:
    """What normalisation actually changed, for the demo to print."""
    out = []
    for b, a in zip(before, after, strict=True):
        if (b.value, b.unit) != (a.value, a.unit):
            out.append(
                f"{b.client_id}/{b.referent}: {b.value} {b.unit} -> {a.value} {a.unit}"
            )
    return sorted(out)
