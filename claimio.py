"""File-level I/O: JSONL in, JSONL out. No Python imports required.

    cat claims.jsonl | python claimio.py > edges.jsonl

    python claimio.py --in claims.jsonl \\
                      --edges edges.jsonl \\
                      --escalations escalations.jsonl

This exists so somebody integrating from another language, another process, or
another repo never has to construct `Claim` dataclasses. The dataclasses stay
the in-process contract; this is the contract at the boundary.

## Formats

**Input** - one JSON object per line, one claim per line:

    {"claim_id": "a1", "client_id": "agent-infra",
     "referent": "request_timeout", "predicate": "the request timeout is",
     "value": "30", "value_type": "numeric",
     "spans": [{"doc_id": "runbook.md", "start": 1200, "end": 1240}],
     "polarity": true, "unit": "seconds", "load_bearing": true}

Required: `claim_id`, `client_id`, `referent`, `value`, `value_type`, and at
least one span. `predicate` defaults to the referent, `polarity` to true,
`unit` to null, `load_bearing` to false.

**Output** - `edges.jsonl` is every typed relation, one per line:

    {"a": "a1", "b": "a2", "relation": "conflicts",
     "basis": "structural:numeric", "detail": "30.0 vs 60.0", "score": null}

**Output** - `escalations.jsonl` is the subset a human must see:

    {"claim_ids": ["a1","a2"], "clients": ["agent-infra","agent-sre"],
     "referent": "request_timeout", "values": ["30","60"],
     "basis": "structural:numeric", "reason": "30.0 vs 60.0",
     "spans": [{"doc_id":"runbook.md","start":1200,"end":1240,"quote":null}]}

`quote` is always null on output. Span text is never echoed back, whatever was
supplied on input - the aggregator works on pointers.

## A note on the input

A malformed line is reported and skipped, never repaired. A claim whose
`value_type` is outside the enum, or which carries no span, is dropped: a claim
with no provenance is not a claim this system will reason about.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence

from aggregate import claimguard_merge
from claimguard import Claim, SpanRef, ValueType

_TYPES = {v.value for v in ValueType}


def claim_from_dict(d: dict) -> Claim:
    """One JSON object -> one Claim. Raises ValueError with the reason."""
    for key in ("claim_id", "client_id", "referent", "value", "value_type"):
        if not str(d.get(key, "")).strip():
            raise ValueError(f"missing required field {key!r}")

    vtype = str(d["value_type"]).strip().lower()
    if vtype not in _TYPES:
        raise ValueError(f"value_type {vtype!r} not in {sorted(_TYPES)}")

    spans = [
        SpanRef(str(s["doc_id"]), int(s["start"]), int(s["end"]),
                s.get("quote") or None)
        for s in d.get("spans", [])
        if isinstance(s, dict) and {"doc_id", "start", "end"} <= set(s)
    ]
    if not spans:
        raise ValueError("a claim must carry at least one span")

    return Claim(
        claim_id=str(d["claim_id"]),
        client_id=str(d["client_id"]),
        referent=str(d["referent"]),
        predicate=str(d.get("predicate") or d["referent"]),
        value=str(d["value"]),
        value_type=ValueType(vtype),
        spans=spans,
        polarity=bool(d.get("polarity", True)),
        unit=(str(d["unit"]) if d.get("unit") else None),
        local_confidence=d.get("local_confidence"),
        load_bearing=bool(d.get("load_bearing", False)),
    )


def read_claims(lines: Iterable[str], *, strict: bool = False) -> list[Claim]:
    """Parse JSONL. Bad lines go to stderr and are skipped."""
    claims: list[Claim] = []
    for n, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            claims.append(claim_from_dict(json.loads(line)))
        except Exception as exc:
            message = f"line {n}: {exc}"
            if strict:
                raise ValueError(message) from exc
            print(f"skipped {message}", file=sys.stderr)
    return claims


def write_jsonl(path_or_stream, rows: Sequence[dict]) -> None:
    stream = (
        open(path_or_stream, "w") if isinstance(path_or_stream, str)
        else path_or_stream
    )
    try:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    finally:
        if isinstance(path_or_stream, str):
            stream.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="claimio.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="infile", default="-",
                    help="claims JSONL, or - for stdin (default)")
    ap.add_argument("--edges", default="-",
                    help="typed relations out, or - for stdout (default)")
    ap.add_argument("--escalations", default=None,
                    help="escalations out; omit to skip")
    ap.add_argument("--nli", default="none", choices=["none", "local"],
                    help="none = Layer 1 only (no model, no network)")
    ap.add_argument("--normalise-units", action="store_true",
                    help="canonicalise units before comparing")
    ap.add_argument("--all-conflicts", action="store_true",
                    help="escalate every conflict, not only load-bearing ones")
    ap.add_argument("--strict", action="store_true",
                    help="fail on a malformed line instead of skipping it")
    args = ap.parse_args(argv)

    handle = sys.stdin if args.infile == "-" else open(args.infile)
    try:
        claims = read_claims(handle, strict=args.strict)
    finally:
        if handle is not sys.stdin:
            handle.close()

    if not claims:
        print("no valid claims on input", file=sys.stderr)
        return 1

    if args.normalise_units:
        import units

        claims = units.normalise(claims)

    fn = None
    if args.nli == "local":
        import nli

        fn = nli.load_or_none()

    result = claimguard_merge(
        claims, nli=fn, only_load_bearing=not args.all_conflicts
    )

    write_jsonl(
        sys.stdout if args.edges == "-" else args.edges,
        [e.to_dict() for e in result.edges],
    )
    if args.escalations:
        write_jsonl(args.escalations, [e.to_dict() for e in result.escalations])

    print(
        f"{len(claims)} claims -> {len(result.edges)} edges, "
        f"{len(result.escalations)} escalations "
        f"({len(result.suppressed)} detected but not escalated)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
