"""The acceptance table: nine planted cases, and which layer must decide each.

`corpus.json` ships a `gold` key naming, per case, the mechanism that SHOULD
fire. That is stricter than "did it find the conflict": if a different layer
decides a case, the answer can be right for the wrong reason and the next
corpus will break it silently.

    python acceptance.py              Layer 1 only, no model, no network
    python acceptance.py --nli local  adds the cross-encoder for K4 and K8

K6 is the one to watch. `4 hours` vs `240 minutes` AGREE, and the required
result is UNDECIDED with basis `structural:unit-mismatch` - not a conflict.
A gate that reports it is a gate that gets switched off.
"""

from __future__ import annotations

import argparse
import sys

import extract
from aggregate import claimguard_merge
from claimguard import Relation

def collect(corpus_path: str, normalise: bool = False):
    """Extraction, exactly as the ClientApps do it, in one process.

    This is the honest local stand-in for the federated collect: same
    extractor, same per-client slicing, same wire serialisation - the claims
    are round-tripped through `to_wire`/`from_wire` so the local path cannot
    accidentally see a quote the federated path would have withheld.
    """
    corpus = extract.load_corpus(corpus_path)
    claims = []
    paths = set()
    for client_id in extract.client_ids(corpus):
        cs, path = extract.extract(extract.passages(corpus, client_id), client_id)
        paths.add(path)
        claims.extend(extract.from_wire(extract.to_wire(cs, max_quote_chars=0)))
    if normalise:
        claims = units.normalise(claims)
    return claims, paths


GOLD = {
    "K1": ("monitoring_interval", {"site-a", "site-b"}, Relation.CONFLICTS,
           "structural:numeric", True),
    "K2": ("protocol_effective_date", {"site-a", "site-c"}, Relation.CONFLICTS,
           "structural:date", True),
    "K3": ("concurrent_anticoagulant", {"site-a", "site-b"}, Relation.CONFLICTS,
           "structural:negation", True),
    "K4": ("escalation_authority", {"site-a", "site-b"}, Relation.CONFLICTS,
           "nli", True),
    "K5": ("specimen_storage_category", {"site-a", "site-c"}, Relation.CONFLICTS,
           "structural:enum", False),
    "K6": ("monitoring_interval", {"site-a", "site-c"}, Relation.UNDECIDED,
           "structural:unit-mismatch", False),
    "K7": ("nurse_ratio_day", {"site-a", "site-b"}, Relation.CORROBORATES,
           "structural:enum", False),
    "K8": ("escalation_authority", {"site-a", "site-c"}, Relation.CORROBORATES,
           "nli", False),
}


def check_gold(claims, result):
    """Returns (case, ok, detail) per gold row, plus the K9 no-edge check."""
    idx = {c.claim_id: c for c in result.claims}
    escalated = {tuple(sorted(e.claim_ids)) for e in result.escalations}
    rows = []

    for case, (referent, clients, relation, basis, should_escalate) in sorted(GOLD.items()):
        match = None
        for e in result.edges:
            a, b = idx[e.a], idx[e.b]
            if a.referent == referent and {a.client_id, b.client_id} == clients:
                match = (e, a, b)
                break
        if match is None:
            rows.append((case, False, f"no edge for {referent} across {sorted(clients)}"))
            continue
        e, a, b = match
        did_escalate = tuple(sorted((e.a, e.b))) in escalated
        problems = []
        if e.relation is not relation:
            problems.append(f"relation {e.relation.value} != {relation.value}")
        if not e.basis.startswith(basis):
            problems.append(f"basis {e.basis!r} != {basis!r}")
        if did_escalate != should_escalate:
            problems.append(
                f"escalated={did_escalate}, expected {should_escalate}")
        rows.append((case, not problems,
                     "; ".join(problems) or
                     f"{e.relation.value} [{e.basis}] "
                     f"{'escalates' if did_escalate else 'silent'}"))

    # K9 - the decoy referents must produce no cross-client edge at all.
    decoys = {"consent_window", "induction_scope"}
    stray = [e for e in result.edges if idx[e.a].referent in decoys]
    rows.append(("K9", not stray,
                 "no edges" if not stray
                 else f"{len(stray)} stray edge(s): "
                      f"{[idx[e.a].referent for e in stray]}"))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", default="corpus.json")
    ap.add_argument("--nli", default="none", choices=["none", "local"])
    args = ap.parse_args(argv)

    claims, paths = collect(args.corpus, normalise=False)
    fn = None
    if args.nli == "local":
        import nli

        fn = nli.load_or_none()

    result = claimguard_merge(claims, nli=fn)
    rows = check_gold(claims, result)

    engine = "on" if fn else "off"
    print(f"{args.corpus} | extractor: {', '.join(sorted(paths))} | nli: {engine}")
    print()
    for case, ok, detail in rows:
        print(f"  {case:<5} {'PASS' if ok else 'FAIL'}  {detail}")

    failed = [c for c, ok, _ in rows if not ok]
    print()
    if failed:
        print(f"{len(failed)} case(s) failed: {', '.join(failed)}")
        return 1
    print(f"all {len(rows)} gold cases pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
