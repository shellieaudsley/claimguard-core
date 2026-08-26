"""The silo side: documents in, claims JSONL out. Nothing else leaves.

    python silo.py --site site-a --out site-a.claims.jsonl
    python silo.py --site site-a --extractor local        # Qwen, in-process

This is the half that runs where the private data is. It reads documents,
extracts typed claims, and writes JSONL whose spans are pointers. It never
types a relation and never talks to another silo, because it cannot:

## Why relation typing does NOT belong here

`claimguard.block()` yields a pair only when `a.client_id != b.client_id` -
cross-source disagreement is the entire signal. So one silo's claims produce
**zero pairs**. Measured on the hard corpus:

    site-a alone   2 claims  ->  0 pairs  ->  0 edges
    all 3 silos   10 claims  ->  5 pairs  ->  5 edges

Putting the NLI head in the silo therefore buys nothing: there is nothing for
it to compare. DeBERTa runs once, at the aggregator, over claims that have
already arrived from several sources. A silo that loaded it would spend the
memory and produce an empty edge list.

## What crosses the boundary

Exactly what `claimio.py` reads on the other side: one claim per line, spans as
`doc_id` + character offsets, `quote` always null. The document text stays on
this machine, and `--check` proves the pointers resolve against it *here*,
where the documents exist, before anything is sent.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import extract


def emit(
    corpus_path: str,
    site: str,
    *,
    extractor: str = "rules",
    local_model: str = "3b",
) -> tuple[list, str]:
    """Extract this site's claims. Returns (claims, which_path_ran)."""
    corpus = extract.load_corpus(corpus_path)
    if site not in corpus["clients"]:
        raise SystemExit(
            f"unknown site {site!r}; corpus has {sorted(corpus['clients'])}"
        )

    call_llm = None
    if extractor == "local":
        import local_llm

        call_llm = local_llm.make_call_llm(local_model, extract.vocabulary())

    claims, path = extract.extract(
        extract.passages(corpus, site), site, call_llm=call_llm
    )
    return claims, path


def check_pointers(claims, corpus_path: str, site: str, docdir: str) -> list[str]:
    """Verify every span resolves against the real document, HERE.

    The aggregator cannot do this - it has no documents. So the only place the
    provenance claim is checkable is the machine that made it, before the
    claims are sent. A pointer that does not resolve is a fabricated citation,
    and it should not leave.
    """
    root = pathlib.Path(docdir) / site
    problems = []
    for claim in claims:
        for span in claim.spans:
            doc = root / f"{span.doc_id}.txt"
            if not doc.exists():
                problems.append(f"{claim.claim_id}: no document {doc}")
                continue
            text = doc.read_text()
            if span.end > len(text):
                problems.append(
                    f"{claim.claim_id}: {span.doc_id}:{span.start}-{span.end} "
                    f"runs past end of document ({len(text)} chars)"
                )
            elif span.quote and text[span.start : span.end] != span.quote:
                problems.append(
                    f"{claim.claim_id}: {span.doc_id}:{span.start}-{span.end} "
                    f"resolves to {text[span.start:span.end]!r}, "
                    f"not {span.quote!r}"
                )
    return problems


def resolve_escalations(esc_path: str, site: str, docdir: str) -> list[str]:
    """Turn pointers back into text — for THIS site's spans only.

    The demo payoff, and the whole architecture in one command. The aggregator
    holds `protocol_v7:814-863` and cannot read it: it has no documents. This
    machine has the document and no knowledge of the other site's. Neither
    party alone can see both sides of the conflict, which is the point.

    Spans belonging to other sites are skipped rather than guessed at.
    """
    root = pathlib.Path(docdir) / site
    lines = []
    for raw in open(esc_path):
        raw = raw.strip()
        if not raw:
            continue
        esc = json.loads(raw)
        for owner, span in zip(esc["clients"], esc["spans"], strict=False):
            if owner != site:
                continue
            doc = root / f"{span['doc_id']}.txt"
            if not doc.exists():
                lines.append(f"  {esc['referent']}: {span['doc_id']} not held here")
                continue
            text = doc.read_text()[span["start"] : span["end"]]
            lines.append(
                f"  {esc['referent']}\n"
                f"    pointer  {span['doc_id']}:{span['start']}-{span['end']}\n"
                f"    resolves {text!r}"
            )
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="silo.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="which silo this machine is")
    ap.add_argument("--corpus", default="corpus_hard.json")
    ap.add_argument("--out", default="-", help="claims JSONL, or - for stdout")
    ap.add_argument("--extractor", default="rules", choices=["rules", "local"],
                    help="local = Qwen in-process; documents never leave")
    ap.add_argument("--local-model", default="3b")
    ap.add_argument("--resolve", metavar="ESCALATIONS_JSONL", default=None,
                    help="turn this site's pointers back into text and exit")
    ap.add_argument("--check", metavar="DOCDIR", default=None,
                    help="verify every pointer resolves against these documents "
                         "before emitting")
    args = ap.parse_args(argv)

    if args.resolve:
        lines = resolve_escalations(
            args.resolve, args.site, args.check or "documents_hard"
        )
        print(f"{args.site} resolving its own pointers:")
        print("\n".join(lines) if lines else "  (no spans belong to this site)")
        return 0

    claims, path = emit(args.corpus, args.site, extractor=args.extractor,
                        local_model=args.local_model)

    if args.check:
        problems = check_pointers(claims, args.corpus, args.site, args.check)
        if problems:
            print(f"{len(problems)} unresolvable pointer(s) — refusing to emit:",
                  file=sys.stderr)
            for p in problems[:5]:
                print(f"  {p}", file=sys.stderr)
            return 1
        print(f"  all {sum(len(c.spans) for c in claims)} pointers resolve "
              f"against {args.check}/{args.site}", file=sys.stderr)

    stream = sys.stdout if args.out == "-" else open(args.out, "w")
    try:
        for claim in claims:
            # max_quote_chars=0: the quote is dropped HERE, on the machine that
            # holds the document, not filtered out later by the aggregator's
            # good manners.
            stream.write(json.dumps(claim.to_wire(0), sort_keys=True) + "\n")
    finally:
        if stream is not sys.stdout:
            stream.close()

    print(f"{args.site}: {len(claims)} claims via {path}, "
          f"{sum(len(c.spans) for c in claims)} pointers, no document text",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
