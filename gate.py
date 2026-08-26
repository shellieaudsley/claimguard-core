"""Adapter: claimguard -> SiloSafe's verdict vocabulary.

SiloSafe's pipeline is
    dataset-scoped workers -> typed claims -> validator -> deterministic gate
    -> sanitised CaseReport -> final chat model
and its gate emits `INCOMPLETE` | `HOLD` | `READY_FOR_HUMAN_REVIEW`.

claimguard is a gate of that shape already. This module maps one onto the other
so the two can be composed rather than chosen between.

## What this adds to a plain deterministic gate

A gate that answers "do the claims agree?" with yes/no has three blind spots,
and all three are measured rather than argued:

**1. It cannot abstain.** `4 hours` vs `240 minutes` are the same interval. A
gate that compares values raises HOLD on them — a false conflict on a safety
tool, which is how a safety tool gets switched off. Layer 1 returns
`structural:unit-mismatch` and declines to judge, and normalisation then flips
it to agreement without ever having cried wolf.

**2. It cannot decide prose.** `on-call consultant` vs `duty registrar` share no
number, no date and no enum domain. A deterministic comparison either misses the
conflict or reports a string difference as one. Layer 2 (a 184M NLI
cross-encoder, CPU) reads it as contradiction at 0.995 — and abstains below
threshold rather than guessing.

**3. It interrupts on everything.** `refrigerated` vs `frozen` is a real
conflict on a claim no side-effectful step reads. Escalating it trains the
reviewer to dismiss the queue. `load_bearing` is the filter, and the suppressed
conflicts stay in the report where they can be audited.

## What it does NOT change

The verdict stays code-emitted. Nothing here asks a model whether to HOLD, and
the final chat model still cannot override it — it receives the verdict and the
evidence, both already computed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from aggregate import claimguard_merge
from claimguard import Claim, NliFn, Relation

INCOMPLETE = "INCOMPLETE"
HOLD = "HOLD"
READY = "READY_FOR_HUMAN_REVIEW"


@dataclass(frozen=True)
class CaseReport:
    """Sanitised. Carries locators and verdicts, never document text."""

    verdict: str
    reason: str
    escalations: tuple[dict, ...]
    suppressed: tuple[dict, ...]
    abstentions: tuple[dict, ...]
    corroborations: int
    claims_seen: int
    sources_seen: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "escalations": list(self.escalations),
            "suppressed": list(self.suppressed),
            "abstentions": list(self.abstentions),
            "corroborations": self.corroborations,
            "claims_seen": self.claims_seen,
            "sources_seen": self.sources_seen,
        }

    def for_model(self) -> str:
        """The text handed to the final chat model.

        Locators only. Both sides of every conflict, never resolved. The model
        is being asked to render a verdict it did not make and cannot change.
        """
        lines = [
            f"VERDICT: {self.verdict}",
            f"REASON: {self.reason}",
            f"({self.claims_seen} claims from {self.sources_seen} sources; "
            f"{self.corroborations} agreements)",
            "",
        ]
        if self.escalations:
            lines.append("UNRESOLVED CONFLICTS — a human must decide:")
            for e in self.escalations:
                locs = ", ".join(
                    f"{s['doc_id']}:{s['start']}-{s['end']}" for s in e["spans"]
                )
                # `structural:negation` means the two values are the SAME
                # string and the disagreement is entirely in polarity. Printing
                # them raw shows 'permitted' twice under "unresolved conflict",
                # which reads as a broken tool.
                neg = e["basis"] == "structural:negation"
                left = f"{e['values'][0]!r}"
                right = f"NOT {e['values'][1]!r}" if neg else f"{e['values'][1]!r}"
                lines.append(
                    f"  - {e['referent']}: {e['clients'][0]} says {left}; "
                    f"{e['clients'][1]} says {right}. "
                    f"Basis: {e['basis']}. Evidence: [{locs}]"
                )
        if self.abstentions:
            lines += ["", "NOT DECIDED — the gate declined, and that is correct:"]
            for a in self.abstentions:
                lines.append(f"  - {a['referent']}: {a['basis']} {a['detail']}")
        if self.suppressed:
            lines += ["", "CONFLICTS LOGGED, NOT ESCALATED — nothing acts on these:"]
            for s in self.suppressed:
                lines.append(f"  - {s['referent']}: {s['basis']}")
        return "\n".join(lines)


def evaluate(
    claims: Sequence[Claim],
    nli: NliFn | None = None,
    *,
    require_sources: int = 2,
    only_load_bearing: bool = True,
) -> CaseReport:
    """Claims in, verdict out. No model decides anything here.

    `INCOMPLETE` when there is not enough to compare - fewer sources than
    required, or no cross-source pair formed at all. That last case is the one
    worth having: an extractor that names the same concept differently at two
    sites produces zero pairs, and a gate without this check would return
    READY having compared nothing.
    """
    claims = list(claims)
    sources = {c.client_id for c in claims}
    result = claimguard_merge(
        claims, nli=nli, only_load_bearing=only_load_bearing
    )
    index = {c.claim_id: c for c in result.claims}

    def row(edge) -> dict:
        a = index[edge.a]
        return {
            "referent": a.referent,
            "clients": [index[edge.a].client_id, index[edge.b].client_id],
            "values": [index[edge.a].value, index[edge.b].value],
            "basis": edge.basis,
            "detail": edge.detail,
        }

    abstentions = tuple(
        row(e) for e in result.edges if e.relation is Relation.UNDECIDED
    )
    # An abstention on a claim something acts on is not alignment. Reporting it
    # as READY is the precise failure this whole design exists to prevent:
    # "I could not decide" rendered as "they agree". It downgrades to
    # INCOMPLETE, which is what it actually is.
    blocking_abstentions = tuple(
        row(e) for e in result.edges
        if e.relation is Relation.UNDECIDED
        and (index[e.a].load_bearing or index[e.b].load_bearing)
    )
    suppressed = tuple(row(e) for e in result.suppressed)
    corroborations = len(result.by_relation(Relation.CORROBORATES))

    if len(sources) < require_sources:
        verdict, reason = INCOMPLETE, (
            f"only {len(sources)} source(s) reported; {require_sources} required "
            f"before agreement can be assessed"
        )
    elif not result.edges:
        verdict, reason = INCOMPLETE, (
            "no cross-source pair formed. The sources share no referent, so "
            "nothing was compared - this is not agreement"
        )
    elif result.escalations:
        verdict, reason = HOLD, (
            f"{len(result.escalations)} unresolved conflict(s) on load-bearing "
            f"claims"
        )
    elif blocking_abstentions:
        bases = sorted({a["basis"] for a in blocking_abstentions})
        verdict, reason = INCOMPLETE, (
            f"{len(blocking_abstentions)} load-bearing pair(s) could not be "
            f"decided ({', '.join(bases)}). Not agreement - the gate declined"
        )
    else:
        verdict, reason = READY, (
            f"{corroborations} agreement(s), no unresolved conflict on a "
            f"load-bearing claim"
        )

    return CaseReport(
        verdict=verdict,
        reason=reason,
        escalations=tuple(e.to_dict() for e in result.escalations),
        suppressed=suppressed,
        abstentions=abstentions,
        corroborations=corroborations,
        claims_seen=len(claims),
        sources_seen=len(sources),
    )
