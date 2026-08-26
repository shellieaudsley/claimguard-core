"""Task B - passages to Claims, running on the client.

Documents never leave the site. This module is the only thing that ever sees
passage text; everything downstream sees `Claim`s whose spans are pointers.

Two extractors behind one interface:

    extract_rules(...)   deterministic, no model, always available
    extract_llm(...)     prompts a local model for JSON, validates, drops junk

`extract()` picks: LLM when a callable is supplied and produces valid records,
rules otherwise. The spec's order of work says to build the rules path first
and swap the LLM in behind the same interface, and to cut the LLM before
anything else if the day runs short. That is why the fallback is not an
afterthought here - it is the default.

## What the rules extractor is and is not

It is a table of eight referent patterns over this corpus. It is not a claim
extractor for arbitrary prose, and pretending otherwise would make the
acceptance run meaningless: a rules extractor tuned until the gold cases pass
tests nothing except the tuning.

So the honest reading of an all-green acceptance run on the rules path is
**"the aggregator is correct given correct extraction"**, which is exactly the
property the aggregator should be tested for in isolation. Whether extraction
itself is any good is the LLM path's question, and it is measured separately.

## The fields that carry the whole Layer 1 story

`value_type`, `unit` and `polarity` are what let Layer 1 decide K1, K2, K3 and
K6 without a model. Getting them wrong does not produce an error - it produces
a plausible answer from the wrong layer. Two specific failure modes, both
silent, both worth naming:

- `polarity` unset on a negated claim: K3 stops being a structural catch and
  becomes an NLI guess.
- `value_type=TEXT` on a number: the pair is routed to a model that will
  cheerfully compare "4" and "6" as prose.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from claimguard import Claim, SpanRef, ValueType

# Which referents a side-effectful step depends on. In a real deployment this
# is derived from the plan - the claims a tool call reads - not declared. It is
# a constant here because the corpus has no plan attached, and it is kept in
# one visible place rather than sprinkled through the rules, because it is the
# policy knob that decides who gets interrupted.
LOAD_BEARING = frozenset({
    "monitoring_interval",
    "protocol_effective_date",
    "concurrent_anticoagulant",
    "escalation_authority",
})


@dataclass(frozen=True)
class Rule:
    """One referent, one surface pattern, one typed value."""

    referent: str
    predicate: str
    value_type: ValueType
    pattern: re.Pattern
    # One line of plain English saying what this referent MEANS.
    #
    # Not decoration. When the referent vocabulary is handed to a model as a
    # list of bare snake_case identifiers, the model has to infer the meaning
    # from the name - and measurably fails: `Qwen2.5-0.5B-Instruct` obeyed an
    # 8-item vocabulary almost perfectly while using `consent_window` as a
    # dumping ground for three unrelated passages. Obeying a list and
    # understanding it are different things, and a gloss is the cheapest way
    # to ask for the second.
    gloss: str = ""
    # A passage must also contain one of these before the pattern counts.
    # Used where a role name appears in prose that is not an assertion about
    # that role - see `escalation_authority`.
    requires: re.Pattern | None = None
    # Builds the value string from the match; defaults to group "value".
    value_of: Callable[[re.Match], str] | None = None

    def value(self, m: re.Match) -> str:
        return self.value_of(m) if self.value_of else m.group("value")

    def unit(self, m: re.Match) -> str | None:
        try:
            u = m.group("unit")
        except IndexError:  # pattern has no unit group
            return None
        return _canon_unit(u) if u else None

    def polarity(self, m: re.Match) -> bool:
        try:
            return m.group("neg") is None
        except IndexError:  # pattern cannot express negation
            return True


def _canon_unit(u: str) -> str:
    """Singular/plural is not a unit difference. Anything else is."""
    u = u.strip().lower()
    return u[:-1] if u.endswith("s") else u


RULES: tuple[Rule, ...] = (
    # K1 (4h vs 6h) and K6 (4h vs 240min). Same rule, three sites, two units -
    # which is the point: the unit trap is not a special case in the extractor,
    # it is what happens when one referent is written up in local house style.
    Rule(
        referent="monitoring_interval",
        gloss="how often observations are taken during the observation phase",
        predicate="the minimum monitoring interval is",
        value_type=ValueType.NUMERIC,
        pattern=re.compile(
            r"minimum (?:monitoring )?interval\b[^.]*?\b(?:is|of)\s+"
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>hours?|minutes?|hrs?|mins?)",
            re.I,
        ),
    ),
    # K2. Version drift across sites.
    Rule(
        referent="protocol_effective_date",
        gloss="the date a protocol version comes into force",
        predicate="the protocol takes effect on",
        value_type=ValueType.DATE,
        pattern=re.compile(r"takes effect on\s+(?P<value>\d{4}-\d{2}-\d{2})", re.I),
    ),
    # K3. The value is identical on both sides; `polarity` carries the whole
    # disagreement, so Layer 1 fires on negation before it ever compares values.
    Rule(
        referent="concurrent_anticoagulant",
        gloss="whether anticoagulant therapy may be taken alongside the study",
        predicate="concurrent anticoagulant therapy is",
        value_type=ValueType.BOOLEAN,
        pattern=re.compile(
            r"concurrent anticoagulant therapy is\s+(?P<neg>not\s+)?(?P<value>permitted)",
            re.I,
        ),
    ),
    # K4 and K8. TEXT on purpose: no number, no date, no shared enum domain, so
    # Layer 1 must abstain and hand this to the model. `requires` keeps the
    # rule off "The consultant may delegate to a senior registrar", which names
    # a role without asserting who holds authority.
    Rule(
        referent="escalation_authority",
        gloss="which role holds decision authority when a participant deteriorates",
        predicate="escalation authority rests with the",
        value_type=ValueType.TEXT,
        pattern=re.compile(
            r"(?P<value>on-call consultant|duty registrar|site director|regional lead)",
            re.I,
        ),
        requires=re.compile(
            r"escalat|referred|deteriorat|changes unexpectedly|decision authority|"
            r"responsibility for the clinical decision",
            re.I,
        ),
    ),
    # K5. A real conflict that is deliberately not load-bearing.
    Rule(
        referent="specimen_storage_category",
        gloss="the temperature category specimens are stored at",
        predicate="the specimen storage category is",
        value_type=ValueType.ENUM,
        pattern=re.compile(
            r"specimen storage category[^.]*?\bis\s+(?P<value>refrigerated|frozen|ambient)",
            re.I,
        ),
    ),
    # K7. Agreement, and it must stay silent.
    Rule(
        referent="nurse_ratio_day",
        gloss="the nurse-to-participant staffing ratio on day shifts",
        predicate="the day-shift nurse-to-participant ratio is",
        value_type=ValueType.ENUM,
        pattern=re.compile(
            r"nurse-to-participant ratio of\s+(?P<n>\d+)\s+to\s+(?P<d>\d+)", re.I
        ),
        value_of=lambda m: f"{m.group('n')}:{m.group('d')}",
    ),
    # K9, first half. Deliberately NUMERIC in hours, exactly like
    # monitoring_interval: if blocking keyed on unit or on lexical similarity
    # rather than on referent, this would pair with it and produce a nonsense
    # conflict. Only site-b holds it, so a correct run yields no edge at all.
    Rule(
        referent="consent_window",
        gloss="how long a participant is given to consider the information sheet",
        predicate="the consent consideration window is",
        value_type=ValueType.NUMERIC,
        pattern=re.compile(
            r"minimum of\s+(?P<value>\d+)\s+(?P<unit>hours?|days?)\s+to consider", re.I
        ),
    ),
    # K9, second half. Mentions consent, observation and specimen handling -
    # every other referent's vocabulary at once - and must still pair with
    # nothing.
    Rule(
        referent="induction_scope",
        gloss="what new-staff induction training covers",
        predicate="induction covers",
        value_type=ValueType.TEXT,
        pattern=re.compile(r"induction covering\s+(?P<value>[^.]+)", re.I),
    ),
)


# ------------------------------------------------------------------ corpus


def load_corpus(path: str = "corpus.json") -> dict:
    with open(path) as fh:
        return json.load(fh)


def client_ids(corpus: dict) -> list[str]:
    return sorted(corpus["clients"])


def passages(corpus: dict, client_id: str) -> list[dict]:
    """What ships to a node. `gold` is never included - it stays server-side."""
    return corpus["clients"][client_id]["passages"]


# ------------------------------------------------------------ rules path


def extract_rules(passages: Sequence[dict], client_id: str) -> list[Claim]:
    """Deterministic extraction. No model, no network, no ordering surprises."""
    found: list[Claim] = []
    for p in passages:
        text, base = p["text"], p["start"]
        for rule in RULES:
            if rule.requires and not rule.requires.search(text):
                continue
            for m in rule.pattern.finditer(text):
                found.append(
                    Claim(
                        claim_id="",  # assigned after merge, so ids stay stable
                        client_id=client_id,
                        referent=rule.referent,
                        predicate=rule.predicate,
                        value=rule.value(m).strip(),
                        value_type=rule.value_type,
                        spans=[SpanRef(p["doc_id"], base + m.start(), base + m.end(),
                                       quote=m.group(0))],
                        polarity=rule.polarity(m),
                        unit=rule.unit(m),
                        local_confidence=1.0,
                        load_bearing=rule.referent in LOAD_BEARING,
                    )
                )
                break  # one claim per referent per passage
    return _merge_and_id(found, client_id)


def _merge_and_id(claims: Sequence[Claim], client_id: str) -> list[Claim]:
    """Collapse same-site restatements into one multi-span claim.

    A site asserting the same thing in three passages holds one claim citing
    three spans, not three claims - which is the schema's reason for `spans`
    being a list. On this corpus the rules path never triggers it (each
    referent is stated once per site); the LLM path does, because a summary
    genuinely draws on several passages. Pinned by a unit test rather than by
    the demo, so the behaviour is real even though the corpus does not show it.
    """
    merged: dict[tuple, Claim] = {}
    for c in claims:
        key = (c.referent, c.value.lower(), c.polarity, c.unit, c.value_type)
        if key in merged:
            merged[key].spans.extend(c.spans)
        else:
            merged[key] = c

    out = []
    for n, key in enumerate(sorted(merged, key=lambda k: (k[0], str(k[1]))), start=1):
        c = merged[key]
        c.spans.sort(key=lambda s: (s.doc_id, s.start))
        out.append(
            Claim(
                claim_id=f"{client_id}-c{n}",
                client_id=c.client_id,
                referent=c.referent,
                predicate=c.predicate,
                value=c.value,
                value_type=c.value_type,
                spans=c.spans,
                polarity=c.polarity,
                unit=c.unit,
                local_confidence=c.local_confidence,
                load_bearing=c.load_bearing,
            )
        )
    return out


# -------------------------------------------------------------- LLM path

PROMPT = """Extract every factual assertion in the passage below as JSON.

Return a JSON array. Each element must have exactly these keys:
  "referent"    snake_case topic key, e.g. "monitoring_interval"
  "predicate"   the assertion in plain words, without the value
  "value"       the value alone, as a string
  "value_type"  one of: numeric, date, enum, text, boolean
  "unit"        the unit if value_type is numeric, else null
  "polarity"    true if stated positively, false if the assertion is negated
  "quote"       the exact substring of the passage the assertion comes from

Rules:
  - "quote" must appear verbatim in the passage. Do not paraphrase it.
  - polarity is true by default. Set it false ONLY when the assertion itself
    is negated ("not permitted", "must not", "is not"). A passage that merely
    contains "no" or "not" somewhere else is still polarity true.
  - Use numeric for anything with a number and a unit.
  - Return [] if the passage asserts nothing.

Passage:
{passage}

JSON:"""

_ALLOWED_TYPES = {v.value for v in ValueType}


def extract_llm(
    passages: Sequence[dict], client_id: str, call_llm: Callable[[str], str]
) -> list[Claim]:
    """Prompt a local model per passage; keep only structurally valid records.

    Malformed records are dropped, never repaired. A repaired record is a
    record whose provenance you have quietly invented, and this whole system
    rests on provenance being checkable.
    """
    found: list[Claim] = []
    for p in passages:
        try:
            raw = call_llm(PROMPT.format(passage=p["text"]))
            records = _parse_json_array(raw)
        except Exception:
            continue  # this passage yields nothing; the run continues
        for rec in records:
            claim = _validate(rec, p, client_id)
            if claim is not None:
                found.append(claim)
    return _merge_and_id(found, client_id)


def _parse_json_array(raw: str) -> list[dict]:
    """Tolerant of surrounding chatter, intolerant of malformed JSON."""
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    parsed = json.loads(raw[start : end + 1])
    return [r for r in parsed if isinstance(r, dict)]


def _validate(rec: dict, passage: dict, client_id: str) -> Claim | None:
    """Structural validation. Every check here drops, none repairs.

    The load-bearing one is the quote check: the model is asked for a verbatim
    substring and the offset is derived by *searching the passage for it*. The
    model is never asked for a character offset and therefore cannot fabricate
    one - the same rule `contra` applies to its evidence spans.
    """
    referent = str(rec.get("referent", "")).strip()
    # The vocabulary's escape hatch. A model given somewhere to put what it
    # cannot classify stops forcing bad matches onto real referents; the
    # records land here and are dropped, which is what "omit it" was supposed
    # to achieve and did not.
    if referent.lower() in {"other", "none", "n/a", ""}:
        return None
    value = str(rec.get("value", "")).strip()
    quote = str(rec.get("quote", ""))
    vtype = str(rec.get("value_type", "")).strip().lower()

    if not referent or not value or not quote:
        return None
    if vtype not in _ALLOWED_TYPES:
        return None

    # Case-insensitive, because small models routinely lowercase what they
    # echo back. The offset is still DERIVED by searching the passage, and the
    # quote stored is the passage's own text at that offset rather than the
    # model's copy of it - so a model that alters the text loses, and a model
    # that merely lowercases it does not.
    offset = passage["text"].lower().find(quote.lower())
    if offset < 0:
        return None  # not a substring: the model paraphrased, so it is unusable
    quote = passage["text"][offset : offset + len(quote)]

    unit = rec.get("unit")
    return Claim(
        claim_id="",
        client_id=client_id,
        referent=referent,
        predicate=str(rec.get("predicate", referent)).strip() or referent,
        value=value,
        value_type=ValueType(vtype),
        spans=[SpanRef(passage["doc_id"], passage["start"] + offset,
                       passage["start"] + offset + len(quote), quote=quote)],
        polarity=bool(rec.get("polarity", True)),
        unit=_canon_unit(str(unit)) if unit else None,
        local_confidence=None,
        load_bearing=referent in LOAD_BEARING,
    )


# --------------------------------------------------------------- chooser


def extract(
    passages: Sequence[dict],
    client_id: str,
    call_llm: Callable[[str], str] | None = None,
) -> tuple[list[Claim], str]:
    """The one entry point the ClientApp calls. Returns (claims, which_path).

    Falls back to rules when the LLM path yields nothing at all, so a model
    that is missing, slow, or producing garbage costs the demo its extraction
    quality and nothing else.
    """
    if call_llm is not None:
        claims = extract_llm(passages, client_id, call_llm)
        if claims:
            return claims, "llm"
    return extract_rules(passages, client_id), "rules"


def vocabulary() -> dict[str, str]:
    """The controlled referent vocabulary, as {referent: gloss}.

    This is the blocking key's domain. It is the one field that has to mean the
    same thing at every site, so it is not something an extractor may invent.
    """
    return {r.referent: r.gloss for r in RULES}


def to_wire(claims: Sequence[Claim], max_quote_chars: int = 0) -> str:
    """Serialise for a `ConfigRecord`. Quotes withheld by default."""
    return json.dumps([c.to_wire(max_quote_chars=max_quote_chars) for c in claims])


def from_wire(blob: str) -> list[Claim]:
    """Rebuild `Claim`s server-side. Spans arrive as pointers, quote=None."""
    out = []
    for d in json.loads(blob):
        out.append(
            Claim(
                claim_id=d["claim_id"],
                client_id=d["client_id"],
                referent=d["referent"],
                predicate=d["predicate"],
                value=d["value"],
                value_type=ValueType(d["value_type"]),
                spans=[SpanRef(**s) for s in d["spans"]],
                polarity=d["polarity"],
                unit=d["unit"],
                local_confidence=d.get("local_confidence"),
                load_bearing=d.get("load_bearing", False),
            )
        )
    return out
