"""
claimguard - conflict-aware claim aggregation for federated agent teams.

Design:
  Claims carry provenance POINTERS (doc_id + char offsets), never documents.
  Pointers are dereferenceable only at the owning client.
  A claim may cite MULTIPLE spans -> summaries that draw on several passages
  are representable as one claim, which is what one-to-one span grounding
  could not do.

Relation typing is deterministic-first:
  Layer 1  structural comparison (numeric / temporal / enum / negation)  -> exact, auditable
  Layer 2  NLI cross-encoder, only on pairs Layer 1 cannot decide       -> lexical / paraphrase

No network calls. No Flower dependency. Pure functions, so this drops into
a ServerApp strategy, a ClientApp, or a notebook unchanged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from itertools import combinations
from typing import Callable, Iterable, Optional

# ---------------------------------------------------------------- schema


@dataclass(frozen=True)
class SpanRef:
    """Provenance pointer. Resolvable ONLY at the owning client."""

    doc_id: str
    start: int
    end: int
    quote: Optional[str] = None  # optional, hard-capped, may be None under strict policy

    def redacted(self, max_quote_chars: int = 0) -> "SpanRef":
        if max_quote_chars <= 0 or self.quote is None:
            return SpanRef(self.doc_id, self.start, self.end, None)
        return SpanRef(self.doc_id, self.start, self.end, self.quote[:max_quote_chars])


class ValueType(str, Enum):
    NUMERIC = "numeric"
    DATE = "date"
    ENUM = "enum"
    TEXT = "text"       # only TEXT falls through to NLI
    BOOLEAN = "boolean"


@dataclass
class Claim:
    claim_id: str
    client_id: str
    referent: str              # what the claim is ABOUT - the blocking key
    predicate: str             # surface form, used by NLI
    value: str
    value_type: ValueType
    spans: list[SpanRef] = field(default_factory=list)   # 1..n, not 1
    polarity: bool = True      # False = negated
    unit: Optional[str] = None
    local_confidence: Optional[float] = None
    load_bearing: bool = False  # cited by a side-effectful step

    def to_wire(self, max_quote_chars: int = 0) -> dict:
        d = asdict(self)
        d["value_type"] = self.value_type.value
        d["spans"] = [asdict(s.redacted(max_quote_chars)) for s in self.spans]
        return d


class Relation(str, Enum):
    CORROBORATES = "corroborates"
    CONFLICTS = "conflicts"
    REFINES = "refines"
    INDEPENDENT = "independent"
    UNDECIDED = "undecided"     # Layer 1 abstained -> hand to Layer 2


@dataclass
class RelationEdge:
    a: str
    b: str
    relation: Relation
    basis: str                  # "structural:numeric" | "nli" | ...
    detail: str = ""
    score: Optional[float] = None

    def to_dict(self) -> dict:
        """JSON-safe. `ConfigRecord` holds typed scalars, not dataclasses, so
        anything crossing a Flower record has to arrive as a string first."""
        d = asdict(self)
        d["relation"] = self.relation.value
        return d


# ------------------------------------------------------- layer 1: structural

_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _num(s: str) -> Optional[float]:
    m = _NUM.search(s.replace(",", ""))
    return float(m.group()) if m else None


def _date(s: str) -> Optional[tuple[int, int, int]]:
    m = _ISO.search(s)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def structural_relation(a: Claim, b: Claim, rel_tol: float = 0.01) -> RelationEdge:
    """Exact where it can be. Abstains (UNDECIDED) rather than guessing."""
    if a.referent != b.referent:
        return RelationEdge(a.claim_id, b.claim_id, Relation.INDEPENDENT, "structural:referent")

    if a.polarity != b.polarity:
        return RelationEdge(
            a.claim_id, b.claim_id, Relation.CONFLICTS, "structural:negation",
            "same referent, opposite polarity",
        )

    if a.value_type != b.value_type:
        return RelationEdge(a.claim_id, b.claim_id, Relation.UNDECIDED, "structural:type-mismatch")

    if a.value_type is ValueType.NUMERIC:
        x, y = _num(a.value), _num(b.value)
        if x is None or y is None:
            return RelationEdge(a.claim_id, b.claim_id, Relation.UNDECIDED, "structural:unparsed")
        if a.unit != b.unit:
            return RelationEdge(
                a.claim_id, b.claim_id, Relation.UNDECIDED, "structural:unit-mismatch",
                f"{a.unit!r} vs {b.unit!r} - normalise before comparing",
            )
        denom = max(abs(x), abs(y), 1e-9)
        agree = abs(x - y) / denom <= rel_tol
        return RelationEdge(
            a.claim_id, b.claim_id,
            Relation.CORROBORATES if agree else Relation.CONFLICTS,
            "structural:numeric", f"{x} vs {y}",
        )

    if a.value_type is ValueType.DATE:
        x, y = _date(a.value), _date(b.value)
        if x is None or y is None:
            return RelationEdge(a.claim_id, b.claim_id, Relation.UNDECIDED, "structural:unparsed")
        return RelationEdge(
            a.claim_id, b.claim_id,
            Relation.CORROBORATES if x == y else Relation.CONFLICTS,
            "structural:date", f"{a.value} vs {b.value}",
        )

    if a.value_type in (ValueType.ENUM, ValueType.BOOLEAN):
        same = a.value.strip().lower() == b.value.strip().lower()
        return RelationEdge(
            a.claim_id, b.claim_id,
            Relation.CORROBORATES if same else Relation.CONFLICTS,
            f"structural:{a.value_type.value}", f"{a.value} vs {b.value}",
        )

    return RelationEdge(a.claim_id, b.claim_id, Relation.UNDECIDED, "structural:text")


# ------------------------------------------------------------ blocking

def block(claims: list[Claim]) -> Iterable[tuple[Claim, Claim]]:
    """Exact-referent blocking. Swap in embedding kNN over `referent` for the
    fuzzy version - the interface does not change."""
    buckets: dict[str, list[Claim]] = {}
    for c in claims:
        buckets.setdefault(c.referent, []).append(c)
    for group in buckets.values():
        for a, b in combinations(group, 2):
            if a.client_id != b.client_id:   # cross-client disagreement is the signal
                yield a, b


# --------------------------------------------------- layer 2: NLI fallback

NliFn = Callable[[str, str], tuple[str, float]]  # -> (entail|contradict|neutral, score)


def _no_nli(premise: str, hypothesis: str) -> tuple[str, float]:
    return ("neutral", 0.0)


# Which Layer 1 abstentions Layer 2 is allowed to see.
#
# Layer 1 abstains for four different reasons and they are not interchangeable.
# Only ONE of them is a question a language model can answer:
#
#   structural:text            two prose values -> NLI is exactly the right tool
#   structural:unit-mismatch   4 hours vs 240 minutes -> needs arithmetic
#   structural:type-mismatch   a number against a string -> needs a schema fix
#   structural:unparsed        the value did not parse -> needs a better extractor
#
# Handing the last three to a cross-encoder does not produce an abstention, it
# produces a confident wrong answer: measured on gold case K6, NLI reads
# "interval is 4" against "interval is 240" and returns contradiction, turning
# the corpus's designed false-positive trap into exactly the false positive it
# was built to catch. The spec states the rule in section 4 - "Only TEXT
# reaches NLI" - and the acceptance table requires K6 to keep reporting
# `structural:unit-mismatch`, so this constant is what makes the two agree.
NLI_ELIGIBLE_BASES = frozenset({"structural:text"})


def resolve(
    claims: list[Claim],
    nli: NliFn = _no_nli,
    nli_threshold: float = 0.7,
    nli_bases: frozenset[str] = NLI_ELIGIBLE_BASES,
) -> list[RelationEdge]:
    edges: list[RelationEdge] = []
    for a, b in block(claims):
        edge = structural_relation(a, b)
        if edge.relation is not Relation.UNDECIDED or edge.basis not in nli_bases:
            # Abstained for a reason Layer 2 cannot help with. The edge keeps
            # its diagnostic basis, which is what tells the operator whether
            # the fix is unit normalisation, a schema correction, or a better
            # extractor.
            edges.append(edge)
            continue
        label, score = nli(f"{a.predicate} {a.value}", f"{b.predicate} {b.value}")
        if score < nli_threshold:
            edge.basis, edge.detail = "nli:low-confidence", f"{label}@{score:.2f}"
            edges.append(edge)
            continue
        mapped = {
            "contradiction": Relation.CONFLICTS,
            "contradict": Relation.CONFLICTS,
            "entailment": Relation.CORROBORATES,
            "entail": Relation.CORROBORATES,
        }.get(label, Relation.INDEPENDENT)
        edges.append(RelationEdge(a.claim_id, b.claim_id, mapped, "nli", label, score))
    return edges


# --------------------------------------------------- escalation policy

@dataclass
class Escalation:
    claim_ids: tuple[str, str]
    clients: tuple[str, str]
    referent: str
    values: tuple[str, str]
    basis: str
    spans: list[dict]
    reason: str

    def to_dict(self) -> dict:
        """JSON-safe, and already quote-free: `spans` were redacted by
        `escalate()` before they ever reached this object."""
        d = asdict(self)
        d["claim_ids"] = list(self.claim_ids)
        d["clients"] = list(self.clients)
        d["values"] = list(self.values)
        return d


def escalate(
    claims: list[Claim],
    edges: list[RelationEdge],
    max_quote_chars: int = 0,
    only_load_bearing: bool = True,
) -> list[Escalation]:
    """Interrupt the human only for unresolved conflict on a claim that a
    side-effectful step depends on. Everything else passes silently -
    alert fatigue is itself a safety failure."""
    idx = {c.claim_id: c for c in claims}
    out = []
    for e in edges:
        if e.relation is not Relation.CONFLICTS:
            continue
        a, b = idx[e.a], idx[e.b]
        if only_load_bearing and not (a.load_bearing or b.load_bearing):
            continue
        out.append(
            Escalation(
                claim_ids=(a.claim_id, b.claim_id),
                clients=(a.client_id, b.client_id),
                referent=a.referent,
                values=(a.value, b.value),
                basis=e.basis,
                spans=[asdict(s.redacted(max_quote_chars)) for s in (*a.spans, *b.spans)],
                reason=e.detail or "conflicting values on shared referent",
            )
        )
    return out


# --------------------------------------------------------------- demo

if __name__ == "__main__":
    claims = [
        Claim("c1", "site-a", "notice_period", "notice period is", "30",
              ValueType.NUMERIC, [SpanRef("policy_v3", 1204, 1281)], unit="days",
              load_bearing=True),
        Claim("c2", "site-b", "notice_period", "notice period is", "60",
              ValueType.NUMERIC, [SpanRef("policy_v7", 88, 140)], unit="days",
              load_bearing=True),
        Claim("c3", "site-c", "notice_period", "notice period is", "30",
              ValueType.NUMERIC, [SpanRef("handbook", 12, 60)], unit="days"),
        # multi-span claim: a summary drawing on three passages, one claim
        Claim("c4", "site-a", "escalation_path", "escalation goes to", "regional lead",
              ValueType.TEXT,
              [SpanRef("policy_v3", 200, 260), SpanRef("policy_v3", 900, 980),
               SpanRef("memo_11", 5, 90)]),
        Claim("c5", "site-b", "escalation_path", "escalation goes to", "site director",
              ValueType.TEXT, [SpanRef("policy_v7", 400, 470)]),
    ]

    edges = resolve(claims)                    # no NLI wired -> c4/c5 stays UNDECIDED
    for e in edges:
        print(f"{e.a} <-> {e.b}  {e.relation.value:<14} [{e.basis}] {e.detail}")

    print("\nESCALATIONS (load-bearing only, quotes withheld):")
    for esc in escalate(claims, edges):
        print(json.dumps(asdict(esc), indent=2))
