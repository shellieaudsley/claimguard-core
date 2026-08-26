"""What may leave a client, enforced rather than asserted. Self-contained.

`claimguard.escalate(max_quote_chars=0)` already withholds quotes. That is the
floor, not the policy. The rule that actually separates "we share statistics"
from "we share statistics, except when the statistic is a document" is:

    min_support - an attribution resolving to a SINGLE source is a pointer at
    that source's contents, and is withheld ENTIRELY rather than truncated
    into a smaller version of the same leak.

## Provenance

`Budget` and the release logic follow `contra/src/contra/disclosure.py`
(`Budget.federated`, `DisclosureGate.filter_evidence`) from the `fl-artifact`
project. Reproduced rather than imported so this repo ships as one folder.

Two things changed in the move, both simplifications:

- It operates on `claimguard.SpanRef` directly. The original works on
  `contra.types.SpanRef`; importing it meant converting every span across the
  boundary and back, which was pure ceremony.
- The per-span character cap is retained but is close to inert here, because
  spans reach the server with `quote=None` already. It stays because the
  budget is also the local-inspection path, where quotes do exist.

## Order of operations is deliberate

Support is checked **before** any truncation. Truncating a low-support
attribution first would emit a shorter pointer at the same single document,
which is the leak it was supposed to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from claimguard import Claim, Escalation, SpanRef


@dataclass(frozen=True)
class Budget:
    max_spans_per_query: int = 8
    max_chars_per_span: int = 400
    min_support: int = 1  # never return an attribution resolving to one source

    @classmethod
    def local(cls) -> Budget:
        """Wide open. What you run on your own machine."""
        return cls(max_spans_per_query=64, max_chars_per_span=100_000, min_support=1)

    @classmethod
    def federated(cls) -> Budget:
        """The defensible default when the spans belong to someone else."""
        return cls(max_spans_per_query=3, max_chars_per_span=240, min_support=2)

    @classmethod
    def locators_only(cls) -> Budget:
        return cls(max_spans_per_query=8, max_chars_per_span=0, min_support=1)

    def describe(self) -> str:
        return (
            f"<={self.max_spans_per_query} spans, <={self.max_chars_per_span} chars, "
            f"min_support={self.min_support}"
        )


@dataclass(frozen=True)
class Released:
    """The span pointers an escalation is allowed to carry off-client."""

    spans: tuple[dict, ...]
    withheld: int
    policy: str


@dataclass
class DisclosureGate:
    budget: Budget = field(default_factory=Budget.federated)

    def filter_spans(
        self, owned: Sequence[tuple[SpanRef, str]]
    ) -> tuple[list[dict], int]:
        """Apply the budget to (span, owning client) pairs.

        Returns (released, n_withheld). Support is counted over distinct
        *sources* - ten spans from one document are still one document, and
        one client's spans are still one client.
        """
        if not owned:
            return [], 0

        if self.budget.min_support > 1:
            distinct_docs = len({s.doc_id for s, _ in owned})
            distinct_owners = len({owner for _, owner in owned})
            if max(distinct_docs, distinct_owners) < self.budget.min_support:
                return [], len(owned)

        released: list[dict] = []
        for span, owner in owned[: self.budget.max_spans_per_query]:
            quote = span.quote
            if self.budget.max_chars_per_span <= 0 or quote is None:
                quote = None
            elif len(quote) > self.budget.max_chars_per_span:
                quote = quote[: self.budget.max_chars_per_span]
            released.append({
                "doc_id": span.doc_id,
                "start": span.start,
                "end": span.end,
                "quote": quote,
                "client_id": owner,
            })
        return released, len(owned) - len(released)


DEFAULT_GATE = DisclosureGate(budget=Budget.federated())


def policy() -> str:
    return f"Budget.federated ({DEFAULT_GATE.budget.describe()})"


def available() -> bool:
    """Kept so callers need not care whether the gate is vendored or imported.

    It is now always available - that is the point of vendoring it - but the
    view still asks, so a future build that makes the gate optional cannot
    silently print a privacy claim it is not enforcing.
    """
    return True


def release(esc: Escalation, claims: Sequence[Claim]) -> Released:
    """Apply the gate to one escalation's span pointers.

    `claims` is the ledger the escalation came from. It is a parameter rather
    than something inferred from the `Escalation`, because `escalate()`
    flattens both claims' spans into one list and which client owns which span
    is not recoverable from the result. Guessing it would attribute evidence to
    the wrong site, which is a worse bug than not gating at all.
    """
    owned = _owned_spans(esc, claims)
    spans, withheld = DEFAULT_GATE.filter_spans(owned)
    return Released(spans=tuple(spans), withheld=withheld, policy=policy())


def _owned_spans(esc: Escalation, claims: Sequence[Claim]) -> list[tuple[SpanRef, str]]:
    """(span, owning client) for both sides of the escalation, exactly."""
    index = {c.claim_id: c for c in claims}
    out: list[tuple[SpanRef, str]] = []
    for claim_id in esc.claim_ids:
        claim = index.get(claim_id)
        if claim is None:  # ledger and escalation disagree - report nothing
            continue
        out.extend((s, claim.client_id) for s in claim.spans)
    return out
