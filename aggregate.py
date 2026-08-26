"""The two run modes. Same claims in, and that is the entire argument.

    naive_merge()       what a merge step does today
    claimguard_merge()  relation typing, then escalation

Both take the identical ledger produced by the identical extractor, so nothing
in the contrast is down to one path seeing better inputs than the other.

## Making the naive path a fair baseline

A strawman baseline proves nothing, and the temptation is real: it would be
easy to write a naive path so obviously broken that the comparison is
theatre. So `naive_merge` is written as the thing people actually ship -
group by referent, take the majority value, break ties by client order, emit
one answer with a confidence - and it is given every advantage:

- it sees all three sites' claims, not a subset
- it uses majority vote, which is strictly better than first-wins
- on K7 and K8, where the sites agree, it is exactly right

It still deletes K1, K2, K3 and K4, because *no* function from a set of values
to a single value can do otherwise. That is the point being demonstrated, and
it survives making the baseline as strong as possible. The failure is
structural, not a tuning error - the same shape of claim `fedcontra` makes
about rank fusion, one level up.

`dropped` on each `NaiveAnswer` is the receipt: how many distinct values were
discarded to produce the single confident one.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from claimguard import (
    Claim,
    Escalation,
    NliFn,
    Relation,
    RelationEdge,
    escalate,
    resolve,
)

# ------------------------------------------------------------------ naive


@dataclass(frozen=True)
class NaiveAnswer:
    referent: str
    value: str
    unit: str | None
    support: int          # how many sites backed the winning value
    total: int            # how many sites said anything about this referent
    dropped: tuple[str, ...]   # the values that lost, and vanished
    clients: tuple[str, ...]   # who backed the winner

    @property
    def confidence(self) -> float:
        """The number the human sees. Note it rises with agreement and says
        nothing at all about disagreement - a 2-of-3 majority on a live
        contradiction reads 0.67 and looks merely uncertain."""
        return self.support / self.total if self.total else 0.0

    @property
    def hid_a_conflict(self) -> bool:
        return bool(self.dropped)


def naive_merge(claims: Sequence[Claim]) -> list[NaiveAnswer]:
    """Concatenate ledgers, vote per referent, emit one value each.

    Negation is folded into the value the way a text-level merge would see it,
    because a merge step reading prose has no polarity field - "permitted" and
    "not permitted" are just two strings.
    """
    buckets: dict[str, list[Claim]] = defaultdict(list)
    # Sorted by owner, not by arrival. Over a real federation the replies come
    # back in whatever order the nodes finish, and an unsorted tie-break makes
    # the baseline's answer change between runs on identical input - which
    # would let someone dismiss the whole contrast as a race rather than a
    # design flaw. The arbitrariness being criticised is that a winner is
    # picked at all, not that the pick is unstable.
    for c in sorted(claims, key=lambda c: (c.client_id, c.claim_id)):
        buckets[c.referent].append(c)

    out: list[NaiveAnswer] = []
    for referent in sorted(buckets):
        group = buckets[referent]
        surfaces = [_surface(c) for c in group]
        counts = Counter(surfaces)
        # Majority, ties broken by client order - deterministic, and the
        # tie-break is exactly the arbitrary choice being criticised.
        best = max(sorted(counts), key=lambda v: (counts[v], -_first_index(surfaces, v)))
        winners = [c for c, s in zip(group, surfaces, strict=True) if s == best]
        out.append(
            NaiveAnswer(
                referent=referent,
                value=best,
                unit=winners[0].unit,
                support=counts[best],
                total=len(group),
                dropped=tuple(sorted(v for v in counts if v != best)),
                clients=tuple(sorted(c.client_id for c in winners)),
            )
        )
    return out


def _surface(c: Claim) -> str:
    return c.value if c.polarity else f"not {c.value}"


def _first_index(surfaces: Sequence[str], value: str) -> int:
    return surfaces.index(value)


# ------------------------------------------------------------- claimguard


@dataclass(frozen=True)
class GuardResult:
    claims: tuple[Claim, ...]
    edges: tuple[RelationEdge, ...]
    escalations: tuple[Escalation, ...]

    def by_relation(self, relation: Relation) -> list[RelationEdge]:
        return [e for e in self.edges if e.relation is relation]

    @property
    def detected(self) -> list[RelationEdge]:
        """Conflicts found, whether or not they reached a human."""
        return self.by_relation(Relation.CONFLICTS)

    @property
    def suppressed(self) -> list[RelationEdge]:
        """Conflicts found and deliberately not escalated. Alert fatigue is
        itself a safety failure, so this list existing is a feature - but it
        has to be visible, or it is just a silent drop with better manners."""
        escalated = {tuple(sorted(e.claim_ids)) for e in self.escalations}
        return [e for e in self.detected if tuple(sorted((e.a, e.b))) not in escalated]


def claimguard_merge(
    claims: Sequence[Claim],
    nli: NliFn | None = None,
    *,
    nli_threshold: float = 0.7,
    max_quote_chars: int = 0,
    only_load_bearing: bool = True,
) -> GuardResult:
    """Relation-type every cross-client pair, then apply escalation policy.

    `nli=None` runs Layer 1 only. That is a supported mode, not a degraded
    one: every structural case still resolves exactly, and the TEXT pairs stay
    `UNDECIDED` and visible rather than being guessed at.
    """
    claims = list(claims)
    edges = (
        resolve(claims, nli=nli, nli_threshold=nli_threshold)
        if nli is not None
        else resolve(claims, nli_threshold=nli_threshold)
    )
    escalations = escalate(
        claims, edges,
        max_quote_chars=max_quote_chars,
        only_load_bearing=only_load_bearing,
    )
    return GuardResult(tuple(claims), tuple(edges), tuple(escalations))
