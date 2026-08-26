"""Adapter for SiloSafe's closed claim contract. Additive — changes nothing.

SiloSafe's wire claim is a frozen dataclass with exactly seven fields:

    claim_id, referent, value, value_type, unit, evidence_token, confidence

`_exact_keys` raises on anything extra, and `test_silo_report_rejects_quote_field`
asserts a `quote` key is refused. That is a deliberate closed schema, not an
oversight, and this module does not argue with it: no field is added, no field
is required, and nothing here needs the contract to change.

## What `evidence_token` is, precisely

    payload = "|".join([case_id, silo_id, record_id, quote])
    token   = "ev_" + sha256(payload)[:16]

That is a **commitment**, not a locator. It is genuinely useful and stronger
than a locator in one respect: it leaks nothing. A `doc_id:start-end` tells you
a document exists, what it is called, and roughly where in it the claim sits.
A hash tells you none of that.

What it buys: the silo cannot later change its story. Recompute the digest from
the same four parts and it matches; alter the quote by one word and it does not.

What it cannot buy, and this is the gap worth naming:

    a quote that was NEVER in record_id mints a perfectly valid token

Nothing in the scheme checks `quote ⊆ record(record_id)`. The digest commits to
whatever the silo says, true or invented. So a fabricated citation is
indistinguishable from a real one, at the aggregator and at the reviewer.

Two additive fixes, neither touching the contract:

1. `mint_checked()` — the silo verifies the quote appears in the record BEFORE
   minting. Fabrication is caught where the record exists, which is the only
   place it can be caught. Silo-side only; the wire claim is unchanged.
2. `Reveal` — on request, the owning silo returns the preimage parts. Anyone
   recomputes the digest and checks it. Selective disclosure, opt-in, per
   escalation, and only for the escalations a human is actually looking at.

## Provenance inside claimguard

`claimguard.Claim` wants `spans`. A commitment is not a span, and pretending
otherwise would be the same category error this module exists to point out. So
the token is carried as `SpanRef(doc_id=f"ev:{token}", start=0, end=0)` and
rendered as `commitment ev_…`, never as `doc:start-end`. `is_commitment()`
tells the two apart, and the CaseReport says which kind of provenance it holds.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from claimguard import Claim, SpanRef, ValueType

CONTRACT_KEYS = frozenset({
    "claim_id", "referent", "value", "value_type", "unit",
    "evidence_token", "confidence",
})

_PREFIX = "ev:"


# ------------------------------------------------------------- the token


def mint(case_id: str, silo_id: str, record_id: str, quote: str) -> str:
    """SiloSafe's own digest, reproduced so this module can verify one."""
    payload = "|".join([case_id, silo_id, record_id, quote])
    return "ev_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def mint_checked(
    case_id: str, silo_id: str, record_id: str, quote: str, record_text: str
) -> str:
    """Mint only if the quote actually appears in the record.

    The fix for the gap above, and it costs one substring check. It has to
    happen HERE: the aggregator has no records and the reviewer has no records,
    so the silo is the only party that can ever catch a fabricated citation.
    After this point nobody can.

    Raises rather than returning a token, because a claim whose evidence does
    not exist should not be emitted at all.
    """
    if quote not in record_text:
        raise ValueError(
            f"quote not found in {record_id!r}: {quote[:60]!r}. "
            f"A token minted from it would be well-formed and unverifiable — "
            f"refusing to mint."
        )
    return mint(case_id, silo_id, record_id, quote)


@dataclass(frozen=True)
class Reveal:
    """The preimage, disclosed on request for one escalation."""

    token: str
    case_id: str
    silo_id: str
    record_id: str
    quote: str

    def verifies(self) -> bool:
        return mint(self.case_id, self.silo_id, self.record_id, self.quote) == self.token


def check_reveal(token: str, reveal: Reveal) -> bool:
    """Anyone can run this. No silo data required — just the four parts."""
    return reveal.token == token and reveal.verifies()


# --------------------------------------------------- contract <-> Claim


def is_commitment(span: SpanRef) -> bool:
    """True when this provenance is a hash, not a resolvable location."""
    return span.doc_id.startswith(_PREFIX)


def token_of(span: SpanRef) -> str | None:
    return span.doc_id[len(_PREFIX):] if is_commitment(span) else None


def to_claim(
    record: dict, silo_id: str, *, load_bearing: bool = False
) -> Claim:
    """SiloSafe wire claim -> `claimguard.Claim`. Read-only, no mutation.

    `client_id` comes from the silo the report arrived from, because the
    contract does not carry it and cross-source blocking needs it. `polarity`
    likewise is not in the contract; it defaults to True, which is a real
    limitation — see `polarity_note()`.
    """
    extra = set(record) - CONTRACT_KEYS
    if extra:
        raise ValueError(f"not a SiloSafe wire claim; unexpected keys {sorted(extra)}")

    vtype = str(record.get("value_type", "text")).strip().lower()
    if vtype not in {v.value for v in ValueType}:
        vtype = "text"

    token = str(record.get("evidence_token") or "")
    return Claim(
        claim_id=str(record["claim_id"]),
        client_id=silo_id,
        referent=str(record["referent"]),
        predicate=str(record["referent"]).replace("_", " "),
        value=str(record["value"]),
        value_type=ValueType(vtype),
        spans=[SpanRef(doc_id=f"{_PREFIX}{token}", start=0, end=0, quote=None)],
        polarity=True,
        unit=(str(record["unit"]) if record.get("unit") else None),
        local_confidence=record.get("confidence"),
        load_bearing=load_bearing,
    )


def to_claims(
    report: Sequence[dict], silo_id: str, *, load_bearing: Sequence[str] = ()
) -> list[Claim]:
    """A whole silo report. `load_bearing` names the referents that interrupt."""
    lb = set(load_bearing)
    return [
        to_claim(r, silo_id, load_bearing=r.get("referent") in lb) for r in report
    ]


def polarity_note() -> str:
    """The contract's one real blind spot, stated plainly.

    `polarity` is not a field, so a negated claim and its positive twin arrive
    as the SAME value string. `therapy is permitted` and `therapy is not
    permitted` both land as `permitted`, and the gate reads them as agreement —
    a contradiction rendered as consensus, which is the exact failure mode this
    system exists to catch.

    It is not fixable at the aggregator: the information was destroyed upstream.
    Either the extractor folds the negation into `value` (`not_permitted`), or
    the contract gains a field. The first needs no schema change and is the
    cheaper ask.
    """
    return (
        "SiloSafe's contract has no `polarity` field. A negated claim and its "
        "positive twin therefore share a value string and read as agreement. "
        "Fold the negation into `value` at extraction (e.g. 'not_permitted'), "
        "or the conflict is invisible."
    )
