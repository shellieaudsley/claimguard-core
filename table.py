"""The headline table: the merge step and claimguard, on identical claims.

Both columns are computed from ONE extraction pass, so nothing in the contrast
is down to one path seeing better inputs than the other. `naive_merge` is given
every advantage - all sites, majority vote - because a strawman baseline proves
nothing (see `aggregate.__doc__`).

    python table.py                 Layer 1 only
    python table.py --nli local     adds the cross-encoder; this is the
                                    configuration the README quotes
    python table.py --markdown      emit the README block instead

The two rows worth reading together are `competing values deleted` and
`conflicts detected`. They are the same disagreements, counted once as what the
merge threw away and once as what claimguard kept.
"""

from __future__ import annotations

import argparse
import sys

import extract
from aggregate import claimguard_merge, naive_merge
from claimguard import Relation

# (label, naive, guard) - `None` means the column has no such quantity, which
# is itself the finding: a merge emits no abstentions because it cannot abstain.
ROWS = "rows"


def counts(claims, nli=None) -> list[tuple[str, str, str]]:
    naive = naive_merge(claims)
    guard = claimguard_merge(claims, nli=nli)

    escalated = len(guard.escalations)
    suppressed = len(guard.suppressed)
    abstentions = len(guard.by_relation(Relation.UNDECIDED))

    # A false conflict is a pair reported as CONFLICTS whose values are in fact
    # the same quantity - K6's `4 hours` vs `240 minutes`. Layer 1 abstains on
    # those, so the count is the number of unit-mismatch pairs it did NOT call.
    false_conflicts_guard = sum(
        1 for e in guard.detected if e.basis == "structural:unit-mismatch"
    )

    return [
        ("answers emitted", str(len(naive)), "—"),
        ("competing values deleted",
         str(sum(len(a.dropped) for a in naive)), "0"),
        ("conflicts detected", "0", str(len(guard.detected))),
        ("  escalated to a human", "—", str(escalated)),
        ("  logged, not surfaced", "—", str(suppressed)),
        ("abstentions kept visible", "—", str(abstentions)),
        ("false conflicts raised", "0", str(false_conflicts_guard)),
    ]


def collect(corpus_path: str):
    """Same per-client slicing and wire round-trip as `acceptance.collect`."""
    corpus = extract.load_corpus(corpus_path)
    claims = []
    for client_id in extract.client_ids(corpus):
        cs, _ = extract.extract(extract.passages(corpus, client_id), client_id)
        claims.extend(extract.from_wire(extract.to_wire(cs, max_quote_chars=0)))
    return claims


def render_plain(rows) -> str:
    w = max(len(label) for label, _, _ in rows)
    out = [
        f"{'':<{w}}      merge step      claimguard",
        "─" * (w + 32),
    ]
    for label, a, b in rows:
        out.append(f"{label:<{w}} {a:>15} {b:>15}")
    return "\n".join(out)


def render_markdown(rows) -> str:
    out = ["| | merge step | claimguard |", "|---|---:|---:|"]
    for label, a, b in rows:
        label = label.replace("  ", "&nbsp;&nbsp;", 1) if label.startswith("  ") else label
        out.append(f"| {label} | {a} | {b} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", default="corpus.json")
    ap.add_argument("--nli", default="none", choices=["none", "local"])
    ap.add_argument("--markdown", action="store_true",
                    help="emit the README block instead of the console table")
    args = ap.parse_args(argv)

    fn = None
    if args.nli == "local":
        import nli

        fn = nli.load_or_none()

    claims = collect(args.corpus)
    rows = counts(claims, nli=fn)

    if args.markdown:
        print(render_markdown(rows))
    else:
        print(f"{args.corpus} | nli: {'on' if fn else 'off'} | "
              f"{len(claims)} claims from "
              f"{len({c.client_id for c in claims})} sites")
        print()
        print(render_plain(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
