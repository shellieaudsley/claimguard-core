"""What must not silently break.

The suite is split by what each test protects:

    gold      SPEC.md section 8, the acceptance table
    layering  which layer is allowed to decide what
    wire      what leaves a client
    honesty   the claims the README makes, asserted rather than written down

Layer 2 tests skip when torch/transformers are absent rather than failing, so
the suite is green on a bare checkout. The skip is visible in the output; a
silently-passing NLI test would be worse than a red one.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import acceptance  # noqa: E402
import extract  # noqa: E402
import units  # noqa: E402
from aggregate import claimguard_merge, naive_merge  # noqa: E402
from claimguard import (  # noqa: E402
    Claim,
    Relation,
    SpanRef,
    ValueType,
    block,
    resolve,
    structural_relation,
)

CORPUS = str(Path(__file__).resolve().parent.parent / "corpus.json")


def _nli_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


needs_nli = pytest.mark.skipif(
    not _nli_available(),
    reason="torch/transformers not installed; Layer 2 is unavailable",
)


@pytest.fixture(scope="module")
def claims():
    cs, _ = acceptance.collect(CORPUS)
    return cs


@pytest.fixture(scope="module")
def nli_fn():
    import nli

    return nli.cross_encoder()


# ------------------------------------------------------------------- gold


def test_layer1_gold_cases_need_no_model(claims):
    """K1, K2, K3, K5, K6, K7 and K9 must all resolve with NLI switched off.

    This is the load-bearing property of the deterministic-first design: the
    majority of the acceptance table is decided exactly, and a model outage
    costs the demo two cases rather than all of them.
    """
    result = claimguard_merge(claims, nli=None)
    rows = {case: (ok, detail) for case, ok, detail in acceptance.check_gold(claims, result)}

    for case in ("K1", "K2", "K3", "K5", "K6", "K7", "K9"):
        ok, detail = rows[case]
        assert ok, f"{case} failed with Layer 1 alone: {detail}"


@needs_nli
def test_all_nine_gold_cases_pass(claims, nli_fn):
    result = claimguard_merge(claims, nli=nli_fn)
    failures = [(c, d) for c, ok, d in acceptance.check_gold(claims, result) if not ok]
    assert not failures, failures


@needs_nli
def test_k4_is_decided_by_the_model_and_nothing_else(claims, nli_fn):
    """The case that justifies keeping a model in the loop.

    Layer 1 must abstain (no number, no date, no polarity flip, no shared enum
    domain) and Layer 2 must return contradiction above threshold. If Layer 1
    ever starts deciding this, the model is no longer load-bearing and the
    architecture argument is weaker than claimed.
    """
    a = next(c for c in claims
             if c.referent == "escalation_authority" and c.client_id == "site-a")
    b = next(c for c in claims
             if c.referent == "escalation_authority" and c.client_id == "site-b")

    layer1 = structural_relation(a, b)
    assert layer1.relation is Relation.UNDECIDED
    assert layer1.basis == "structural:text"

    label, score = nli_fn(f"{a.predicate} {a.value}", f"{b.predicate} {b.value}")
    assert label == "contradiction"
    assert score > 0.7


# --------------------------------------------------------------- layering


def test_unit_mismatch_never_reaches_the_model():
    """Regression pin for the bug that made K6 fail.

    `resolve()` originally handed EVERY `UNDECIDED` edge to Layer 2, including
    the unit mismatch. A cross-encoder reads "interval is 4" against "interval
    is 240" and returns contradiction with high confidence, so the corpus's
    designed false-positive trap became the false positive it was built to
    catch - and it did so only when NLI was wired, which is not how the demo
    was being run.

    The stub here returns a confident contradiction for anything. If the gate
    regresses, this test fails; nothing else in the suite would.
    """
    def always_contradicts(premise, hypothesis):
        return ("contradiction", 0.99)

    four_hours = Claim("a", "site-a", "interval", "interval is", "4",
                       ValueType.NUMERIC, [SpanRef("d", 0, 1)], unit="hours")
    two_forty_min = Claim("b", "site-b", "interval", "interval is", "240",
                          ValueType.NUMERIC, [SpanRef("d", 0, 1)], unit="minutes")

    edges = resolve([four_hours, two_forty_min], nli=always_contradicts)
    assert len(edges) == 1
    assert edges[0].relation is Relation.UNDECIDED
    assert edges[0].basis == "structural:unit-mismatch"


def test_type_mismatch_and_unparsed_also_stay_off_the_model():
    """The other two abstention reasons Layer 2 cannot help with."""
    def always_contradicts(premise, hypothesis):
        return ("contradiction", 0.99)

    numeric = Claim("a", "site-a", "r", "p", "4", ValueType.NUMERIC,
                    [SpanRef("d", 0, 1)])
    enum = Claim("b", "site-b", "r", "p", "four", ValueType.ENUM,
                 [SpanRef("d", 0, 1)])
    unparsed_a = Claim("c", "site-a", "q", "p", "many", ValueType.NUMERIC,
                       [SpanRef("d", 0, 1)])
    unparsed_b = Claim("d", "site-b", "q", "p", "several", ValueType.NUMERIC,
                       [SpanRef("d", 0, 1)])

    bases = {e.basis for e in
             resolve([numeric, enum, unparsed_a, unparsed_b], nli=always_contradicts)}
    assert bases == {"structural:type-mismatch", "structural:unparsed"}


def test_text_pairs_do_reach_the_model():
    """The gate must not be so tight that Layer 2 never runs."""
    called = []

    def spy(premise, hypothesis):
        called.append((premise, hypothesis))
        return ("contradiction", 0.99)

    a = Claim("a", "site-a", "r", "authority is", "consultant", ValueType.TEXT,
              [SpanRef("d", 0, 1)])
    b = Claim("b", "site-b", "r", "authority is", "registrar", ValueType.TEXT,
              [SpanRef("d", 0, 1)])
    edges = resolve([a, b], nli=spy)
    assert called
    assert edges[0].relation is Relation.CONFLICTS
    assert edges[0].basis == "nli"


def test_low_confidence_nli_leaves_the_edge_undecided():
    """Abstention is a correct answer and must survive to the UI."""
    def unsure(premise, hypothesis):
        return ("contradiction", 0.4)

    a = Claim("a", "site-a", "r", "p", "x", ValueType.TEXT, [SpanRef("d", 0, 1)])
    b = Claim("b", "site-b", "r", "p", "y", ValueType.TEXT, [SpanRef("d", 0, 1)])
    edge = resolve([a, b], nli=unsure, nli_threshold=0.7)[0]
    assert edge.relation is Relation.UNDECIDED
    assert edge.basis == "nli:low-confidence"


def test_k6_flips_to_corroborates_after_normalisation(claims):
    before = {e.basis for e in claimguard_merge(claims, nli=None).edges
              if e.basis.startswith("structural:unit")}
    assert before == {"structural:unit-mismatch"}

    after = units.normalise(claims)
    idx = {c.claim_id: c for c in after}
    pair = [
        e for e in claimguard_merge(after, nli=None).edges
        if idx[e.a].referent == "monitoring_interval"
        and {idx[e.a].client_id, idx[e.b].client_id} == {"site-a", "site-c"}
    ]
    assert len(pair) == 1
    assert pair[0].relation is Relation.CORROBORATES


def test_normalisation_does_not_dissolve_the_real_conflict(claims):
    """K1 must survive the fix for K6. A normaliser that made everything agree
    would pass the K6 test and destroy the tool."""
    after = units.normalise(claims)
    idx = {c.claim_id: c for c in after}
    pair = [
        e for e in claimguard_merge(after, nli=None).edges
        if idx[e.a].referent == "monitoring_interval"
        and {idx[e.a].client_id, idx[e.b].client_id} == {"site-a", "site-b"}
    ]
    assert len(pair) == 1
    assert pair[0].relation is Relation.CONFLICTS


def test_unknown_units_are_left_alone_so_they_keep_abstaining():
    c = Claim("a", "site-a", "r", "p", "6", ValueType.NUMERIC,
              [SpanRef("d", 0, 1)], unit="doses")
    assert units.normalise_claim(c) is c


# ------------------------------------------------------------------- wire




def test_wire_payload_carries_pointers_and_no_document_text():
    corpus = extract.load_corpus(CORPUS)
    cs = extract.extract_rules(extract.passages(corpus, "site-a"), "site-a")
    blob = extract.to_wire(cs, max_quote_chars=0)

    # No passage sentence may appear anywhere in the serialised ledger.
    for passage in extract.passages(corpus, "site-a"):
        assert passage["text"] not in blob

    for record in json.loads(blob):
        assert record["spans"]
        for span in record["spans"]:
            assert span["quote"] is None
            assert span["end"] > span["start"]


def test_round_trip_through_the_wire_preserves_typing(claims):
    corpus = extract.load_corpus(CORPUS)
    original = extract.extract_rules(extract.passages(corpus, "site-b"), "site-b")
    restored = extract.from_wire(extract.to_wire(original))
    assert [c.claim_id for c in restored] == [c.claim_id for c in original]
    assert [c.value_type for c in restored] == [c.value_type for c in original]
    assert [c.polarity for c in restored] == [c.polarity for c in original]
    assert [c.unit for c in restored] == [c.unit for c in original]





# --------------------------------------------------------------- extraction


def test_every_claim_carries_provenance(claims):
    assert claims
    for c in claims:
        assert c.spans, c.claim_id
        for s in c.spans:
            assert s.end > s.start


def test_spans_resolve_against_the_source_document():
    """A pointer that does not resolve is decorative. Checked client-side,
    where the text still exists."""
    corpus = extract.load_corpus(CORPUS)
    for cid in extract.client_ids(corpus):
        passages = extract.passages(corpus, cid)
        docs = {}
        for p in passages:
            docs.setdefault(p["doc_id"], {})[p["start"]] = p["text"]
        for c in extract.extract_rules(passages, cid):
            for s in c.spans:
                base = max(k for k in docs[s.doc_id] if k <= s.start)
                text = docs[s.doc_id][base]
                assert text[s.start - base : s.end - base] == s.quote


def test_negated_claim_sets_polarity_not_value(claims):
    """K3 is a test of extraction, not of the checker. If polarity is unset the
    two claims read as identical strings and Layer 1 corroborates them."""
    a = next(c for c in claims
             if c.referent == "concurrent_anticoagulant" and c.client_id == "site-a")
    b = next(c for c in claims
             if c.referent == "concurrent_anticoagulant" and c.client_id == "site-b")
    assert a.value == b.value == "permitted"
    assert a.polarity is True
    assert b.polarity is False
    assert structural_relation(a, b).basis == "structural:negation"


def test_delegation_sentence_is_not_an_authority_assertion():
    """site-c's duty_roster says the consultant "may delegate to a senior
    registrar". That names a role without asserting who holds authority, and
    extracting it would put site-c on both sides of K8."""
    corpus = extract.load_corpus(CORPUS)
    cs = extract.extract_rules(extract.passages(corpus, "site-c"), "site-c")
    authority = [c for c in cs if c.referent == "escalation_authority"]
    assert len(authority) == 1
    assert authority[0].value == "on-call consultant"


def test_same_site_restatements_become_one_multi_span_claim():
    """The `spans: list` half of the schema. The corpus never triggers it - each
    site states each referent once - so it is pinned here instead of being
    left as an untested capability."""
    passages = [
        {"doc_id": "d1", "start": 0, "end": 60,
         "text": "The minimum monitoring interval is 4 hours."},
        {"doc_id": "d2", "start": 0, "end": 70,
         "text": "Observations are taken at a minimum interval of 4 hours."},
    ]
    cs = extract.extract_rules(passages, "site-x")
    interval = [c for c in cs if c.referent == "monitoring_interval"]
    assert len(interval) == 1
    assert len(interval[0].spans) == 2
    assert {s.doc_id for s in interval[0].spans} == {"d1", "d2"}


def test_llm_records_with_paraphrased_quotes_are_dropped():
    """The model is never asked for an offset. It is asked for a verbatim
    substring, and the offset is derived by searching for it - so a paraphrase
    is unusable rather than silently mis-located."""
    passage = {"doc_id": "d", "start": 100, "end": 150,
               "text": "The monitoring interval is 4 hours."}

    def paraphrasing_model(prompt):
        return json.dumps([{
            "referent": "monitoring_interval", "predicate": "interval is",
            "value": "4", "value_type": "numeric", "unit": "hours",
            "polarity": True, "quote": "the interval is four hours",
        }])

    assert extract.extract_llm([passage], "site-x", paraphrasing_model) == []


def test_llm_records_with_verbatim_quotes_get_absolute_offsets():
    passage = {"doc_id": "d", "start": 100, "end": 150,
               "text": "The monitoring interval is 4 hours."}

    def good_model(prompt):
        return json.dumps([{
            "referent": "monitoring_interval", "predicate": "interval is",
            "value": "4", "value_type": "numeric", "unit": "hours",
            "polarity": True, "quote": "interval is 4 hours",
        }])

    claim = extract.extract_llm([passage], "site-x", good_model)[0]
    span = claim.spans[0]
    assert span.start == 100 + passage["text"].index("interval is 4 hours")
    assert passage["text"][span.start - 100 : span.end - 100] == span.quote


def test_malformed_llm_output_falls_back_to_rules():
    corpus = extract.load_corpus(CORPUS)
    passages = extract.passages(corpus, "site-a")
    claims, path = extract.extract(passages, "site-a", call_llm=lambda p: "not json")
    assert path == "rules"
    assert claims


# ---------------------------------------------------------------- honesty


def test_naive_merge_deletes_every_load_bearing_conflict(claims):
    """The README's central claim, asserted rather than written down.

    The baseline is given every advantage - all three sites, majority vote -
    and still emits one confident value per referent for all four load-bearing
    disagreements. No function from a set of values to a single value can do
    otherwise, which is why this is a structural result and not a tuning gap.
    """
    answers = {a.referent: a for a in naive_merge(claims)}
    for referent in ("monitoring_interval", "protocol_effective_date",
                     "concurrent_anticoagulant", "escalation_authority"):
        answer = answers[referent]
        assert answer.hid_a_conflict, referent
        assert answer.value  # one confident value, disagreement gone


def test_naive_merge_is_right_where_the_sites_agree(claims):
    """Not a strawman: on K7 the baseline is exactly correct and says so."""
    answers = {a.referent: a for a in naive_merge(claims)}
    ratio = answers["nurse_ratio_day"]
    assert ratio.value == "1:6"
    assert not ratio.hid_a_conflict
    assert ratio.confidence == 1.0


def test_k5_is_detected_but_not_escalated(claims):
    """Alert fatigue is itself a safety failure. The conflict must appear in
    the edge list and must not interrupt anybody."""
    result = claimguard_merge(claims, nli=None)
    idx = {c.claim_id: c for c in result.claims}

    storage = [e for e in result.detected
               if idx[e.a].referent == "specimen_storage_category"]
    assert storage, "the conflict must be detected"
    assert all(e in result.suppressed for e in storage), "and not escalated"
    assert "specimen_storage_category" not in {e.referent for e in result.escalations}


def test_decoy_referents_produce_no_edges(claims):
    """K9. consent_window is NUMERIC in hours, exactly like monitoring_interval,
    and induction_scope name-checks every other referent's vocabulary. Exact
    referent blocking must reject both."""
    result = claimguard_merge(claims, nli=None)
    idx = {c.claim_id: c for c in result.claims}
    assert not [e for e in result.edges
                if idx[e.a].referent in {"consent_window", "induction_scope"}]


def test_blocking_never_pairs_a_client_with_itself(claims):
    for a, b in block(claims):
        assert a.client_id != b.client_id


def test_run_is_deterministic(claims):
    """Same claims, same edges, twice - so a demo re-run cannot surprise you."""
    first = claimguard_merge(claims, nli=None)
    second = claimguard_merge(claims, nli=None)
    assert [(e.a, e.b, e.relation, e.basis) for e in first.edges] == \
           [(e.a, e.b, e.relation, e.basis) for e in second.edges]


def test_naive_merge_does_not_depend_on_arrival_order(claims):
    """Nodes reply in completion order. If the baseline's answer moved between
    runs on identical input, the whole contrast could be waved away as a race
    rather than a design flaw - so the tie-break is pinned to client order."""
    import random

    reference = [(a.referent, a.value) for a in naive_merge(claims)]
    for seed in range(5):
        shuffled = list(claims)
        random.Random(seed).shuffle(shuffled)
        assert [(a.referent, a.value) for a in naive_merge(shuffled)] == reference


def test_guard_does_not_depend_on_arrival_order(claims):
    import random

    def signature(cs):
        result = claimguard_merge(cs, nli=None)
        idx = {c.claim_id: c for c in result.claims}
        return sorted(
            (idx[e.a].referent, tuple(sorted((idx[e.a].client_id, idx[e.b].client_id))),
             e.relation, e.basis)
            for e in result.edges
        )

    reference = signature(claims)
    for seed in range(5):
        shuffled = list(claims)
        random.Random(seed).shuffle(shuffled)
        assert signature(shuffled) == reference


def test_extraction_is_deterministic():
    a, _ = acceptance.collect(CORPUS)
    b, _ = acceptance.collect(CORPUS)
    assert [c.to_wire() for c in a] == [c.to_wire() for c in b]


# ------------------------------------------------------------- disclosure


@needs_nli
def test_escalation_spans_carry_no_quotes(claims, nli_fn):
    import disclosure

    result = claimguard_merge(claims, nli=nli_fn)
    assert result.escalations
    for esc in result.escalations:
        released = disclosure.release(esc, result.claims)
        assert released.spans
        for span in released.spans:
            assert span["quote"] is None
            assert span["client_id"] in esc.clients


@needs_nli
def test_single_source_attribution_is_withheld_entirely():
    """`min_support=2`: an attribution resolving to one source is a pointer at
    that source's contents. Truncating it would be a smaller version of the
    same leak, so it is refused outright."""
    import disclosure
    from claimguard import Escalation

    esc = Escalation(
        claim_ids=("a", "b"), clients=("site-a", "site-a"), referent="r",
        values=("x", "y"), basis="structural:enum", spans=[], reason="",
    )
    one_source = [
        Claim("a", "site-a", "r", "p", "x", ValueType.ENUM, [SpanRef("d1", 0, 5)]),
        Claim("b", "site-a", "r", "p", "y", ValueType.ENUM, [SpanRef("d1", 6, 9)]),
    ]
    released = disclosure.release(esc, one_source)
    assert released.spans == ()
    assert released.withheld == 2


def test_disclosure_degrades_loudly_not_silently():
    """If the gate is unavailable the printed policy must say so, because the
    footer above it makes a privacy claim the reader will believe."""
    import disclosure

    text = disclosure.policy()
    assert ("min_support=2" in text) == disclosure.available()
    if not disclosure.available():
        assert "NOT enforced" in text


def test_lowercased_quote_still_resolves_to_real_offsets():
    """Small models routinely lowercase what they echo back. The offset is
    still derived by searching the passage, and the quote stored is the
    passage's own text - so casing is forgiven and altered text is not.

    Measured on Qwen2.5-0.5B: every quote it returned was lowercased, so a
    case-sensitive check discarded usable records for a difference that
    carries no information.
    """
    passage = {"doc_id": "d", "start": 100, "end": 150,
               "text": "The Monitoring Interval is 4 hours."}

    def lowercasing_model(prompt):
        return json.dumps([{
            "referent": "monitoring_interval", "predicate": "interval is",
            "value": "4", "value_type": "numeric", "unit": "hours",
            "polarity": True, "quote": "the monitoring interval is 4 hours",
        }])

    claim = extract.extract_llm([passage], "site-x", lowercasing_model)[0]
    span = claim.spans[0]
    assert span.start == 100
    # The passage's own casing, not the model's copy of it.
    assert span.quote == "The Monitoring Interval is 4 hours"
    assert passage["text"][span.start - 100 : span.end - 100] == span.quote


def test_altered_quote_is_still_rejected():
    """Forgiving case must not become forgiving content."""
    passage = {"doc_id": "d", "start": 0, "end": 40,
               "text": "The monitoring interval is 4 hours."}

    def altering_model(prompt):
        return json.dumps([{
            "referent": "monitoring_interval", "predicate": "interval is",
            "value": "6", "value_type": "numeric", "unit": "hours",
            "polarity": True, "quote": "the monitoring interval is 6 hours",
        }])

    assert extract.extract_llm([passage], "site-x", altering_model) == []


def test_garbage_extraction_degrades_to_silence_not_to_noise():
    """The property that held when a 0.5B model produced nonsense.

    Measured: `Qwen2.5-0.5B-Instruct` under a controlled vocabulary tagged
    three unrelated site-b passages `consent_window`, producing 7 cross-client
    pairs between things that have nothing to do with each other. Every one
    resolved to UNDECIDED — some on `structural:type-mismatch`, the rest on
    `nli:low-confidence` — and **not one escalation was raised**.

    That is the whole case for deterministic-first layering with abstention: a
    checker fed garbage should go quiet, not start interrupting people. A
    version that guessed on type mismatches would have turned a bad extractor
    into a pager storm.
    """
    def confidently_wrong(premise, hypothesis):
        return ("contradiction", 0.55)  # below threshold, as measured

    # Same shape as the real failure: one referent, mismatched types, values
    # that share nothing.
    junk = [
        Claim("a", "site-a", "consent_window", "p", "induction_scope",
              ValueType.TEXT, [SpanRef("d1", 0, 5)], load_bearing=True),
        Claim("b", "site-b", "consent_window", "p", "12 hours or more",
              ValueType.NUMERIC, [SpanRef("d2", 0, 5)], unit="hour",
              load_bearing=True),
        Claim("c", "site-c", "consent_window", "p", "stable",
              ValueType.TEXT, [SpanRef("d3", 0, 5)], load_bearing=True),
    ]

    result = claimguard_merge(junk, nli=confidently_wrong)
    assert result.edges, "the pairs must still be formed and visible"
    assert all(e.relation is Relation.UNDECIDED for e in result.edges), \
        [(e.relation, e.basis) for e in result.edges]
    assert result.escalations == (), "garbage must not interrupt a human"


def test_cached_checkpoints_load_without_network_and_uncached_ones_may_fetch():
    """Offline must be decided per checkpoint, never forced through os.environ.

    The bug this pins: `local_llm.load()` and `nli._load()` both did
    `os.environ.setdefault("HF_HUB_OFFLINE", "1")`. On a warm laptop that is
    the right behaviour and hides the problem. On a fresh Colab runtime it
    disables all outgoing traffic, and every model load dies with
    `LocalEntryNotFoundError` — which reads like a Hugging Face permissions
    error and is nothing of the kind.

    Worse, `available()` forced offline too, so `measure_llm.py` reported every
    uncached model as "not in the cache" and skipped it silently rather than
    downloading it.

    A library must not reach into the process environment to make that choice
    for its caller.
    """
    # `make test` exports HF_HUB_OFFLINE=1, and huggingface_hub latches it into
    # `constants.HF_HUB_OFFLINE` at import — popping the env var alone does not
    # clear it. That is the exact stickiness `force_online()` exists for, and
    # the reason re-uploading a fixed file into a live notebook does not help.
    import huggingface_hub.constants as hub_constants

    import local_llm
    import nli

    before_env = os.environ.get("HF_HUB_OFFLINE")
    before_const = hub_constants.HF_HUB_OFFLINE
    try:
        # No module may set the flag as an import or call side effect.
        local_llm.force_online()
        local_llm.is_cached("base")
        nli._is_cached(nli.MODEL, nli.REVISION)
        assert "HF_HUB_OFFLINE" not in os.environ, \
            "a cache probe must not set HF_HUB_OFFLINE"

        # Uncached + online allowed -> runnable (it will download).
        assert local_llm.available("7b") is True

        # Uncached + caller asked for offline -> honestly unavailable.
        os.environ["HF_HUB_OFFLINE"] = "1"
        assert local_llm.available("7b") is False

        # And the env var alone is not enough to undo it, which is the whole
        # trap: the constant has to be cleared too.
        hub_constants.HF_HUB_OFFLINE = True
        os.environ.pop("HF_HUB_OFFLINE", None)
        assert local_llm.available("7b") is False, \
            "a latched constant must still read as offline"
        local_llm.force_online()
        assert local_llm.available("7b") is True, "force_online must clear both"
    finally:
        os.environ.pop("HF_HUB_OFFLINE", None)
        if before_env is not None:
            os.environ["HF_HUB_OFFLINE"] = before_env
        hub_constants.HF_HUB_OFFLINE = before_const


# ----------------------------------------------------------- agent app


def test_typed_edges_and_escalations_serialise_to_json():
    """`ConfigRecord` holds typed scalars, not dataclasses. Anything crossing a
    Flower record — federated or AgentApp — has to become a string first."""
    result = claimguard_merge([
        Claim("a", "s1", "r", "p", "30", ValueType.NUMERIC,
              [SpanRef("d", 0, 9)], unit="s", load_bearing=True),
        Claim("b", "s2", "r", "p", "60", ValueType.NUMERIC,
              [SpanRef("d", 0, 9)], unit="s", load_bearing=True),
    ])
    edges = json.loads(json.dumps([e.to_dict() for e in result.edges]))
    escs = json.loads(json.dumps([e.to_dict() for e in result.escalations]))
    assert edges[0]["relation"] == "conflicts"      # enum -> its value
    assert isinstance(escs[0]["clients"], list)     # tuple -> list
    assert all(s["quote"] is None for s in escs[0]["spans"])









# ---------------------------------------------------------------- jsonl io


def test_jsonl_round_trip_produces_typed_edges():
    """The boundary contract: JSONL in, JSONL out, no dataclass imports.

    This is what somebody integrating from another process or another language
    actually uses. If it breaks, the Python API can still be fine and the
    handoff is still broken.
    """
    import claimio

    lines = [
        json.dumps({
            "claim_id": "a", "client_id": "s1", "referent": "timeout",
            "value": "30", "value_type": "numeric", "unit": "seconds",
            "load_bearing": True,
            "spans": [{"doc_id": "d", "start": 0, "end": 9}],
        }),
        json.dumps({
            "claim_id": "b", "client_id": "s2", "referent": "timeout",
            "value": "60", "value_type": "numeric", "unit": "seconds",
            "load_bearing": True,
            "spans": [{"doc_id": "d2", "start": 0, "end": 9}],
        }),
    ]
    claims = claimio.read_claims(lines)
    assert len(claims) == 2

    result = claimguard_merge(claims)
    edge = result.edges[0].to_dict()
    assert edge["relation"] == "conflicts"
    assert edge["basis"] == "structural:numeric"


def test_jsonl_drops_claims_with_no_provenance():
    """A claim with no span is not a claim this system will reason about."""
    import claimio

    bad = json.dumps({"claim_id": "x", "client_id": "s", "referent": "r",
                      "value": "1", "value_type": "numeric", "spans": []})
    assert claimio.read_claims([bad]) == []
    with pytest.raises(ValueError, match="at least one span"):
        claimio.read_claims([bad], strict=True)


def test_jsonl_drops_unknown_value_types():
    import claimio

    bad = json.dumps({"claim_id": "x", "client_id": "s", "referent": "r",
                      "value": "red", "value_type": "rgb",
                      "spans": [{"doc_id": "d", "start": 0, "end": 3}]})
    assert claimio.read_claims([bad]) == []


def test_jsonl_output_never_echoes_span_text():
    """Quotes may arrive on input; they must not come back out. The aggregator
    works on pointers, and echoing text would quietly undo that."""
    import claimio

    line = json.dumps({
        "claim_id": "a", "client_id": "s1", "referent": "r", "value": "30",
        "value_type": "numeric", "unit": "s", "load_bearing": True,
        "spans": [{"doc_id": "d", "start": 0, "end": 9,
                   "quote": "SECRET DOCUMENT TEXT"}],
    })
    other = json.dumps({
        "claim_id": "b", "client_id": "s2", "referent": "r", "value": "60",
        "value_type": "numeric", "unit": "s", "load_bearing": True,
        "spans": [{"doc_id": "d2", "start": 0, "end": 9}],
    })
    result = claimguard_merge(claimio.read_claims([line, other]))
    blob = json.dumps([e.to_dict() for e in result.escalations])
    assert "SECRET DOCUMENT TEXT" not in blob
    assert '"quote": null' in blob


def test_hard_corpus_holds_the_facts_fixed_and_the_prose_hard():
    """`corpus_hard.json` exists to separate two things that look alike: an
    extractor failing, and a corpus being easy.

    Same referents, same planted conflicts, same three sites — only the
    registers differ (hedges, bullets, numbers as words, non-ISO dates). So a
    score difference against `corpus.json` is attributable to prose difficulty
    and nothing else.
    """
    import re

    easy = extract.load_corpus(CORPUS)
    hard = extract.load_corpus(
        str(Path(CORPUS).parent / "corpus_hard.json")
    )

    assert easy["gold"] == hard["gold"], "the answer key must be identical"
    assert set(easy["clients"]) == set(hard["clients"])
    for cid in easy["clients"]:
        assert len(extract.passages(hard, cid)) == 10

    # Offsets computed, not hand-written: end must equal start + len(text).
    for cid in extract.client_ids(hard):
        for p in extract.passages(hard, cid):
            assert p["end"] - p["start"] == len(p["text"]), (cid, p["doc_id"])

    text = " ".join(p["text"] for cid in extract.client_ids(hard)
                    for p in extract.passages(hard, cid))
    assert re.search(r"\n\s*-", text), "no bullets — not actually hard"
    assert re.search(r"\bfour hours\b", text, re.I), "no numbers-as-words"
    assert re.search(r"\b15 January 2026\b", text), "no non-ISO date"


def test_rules_extractor_degrades_on_hard_prose():
    """The claim 'my extractor will match nothing on your data', as a number.

    Eight regexes tuned to one corpus reach 8/8 there and 3/8 on the same facts
    in different registers. Asserting the direction rather than the exact value,
    so tightening a rule is not a test failure.
    """
    hard = extract.load_corpus(str(Path(CORPUS).parent / "corpus_hard.json"))
    easy = extract.load_corpus(CORPUS)

    def referent_sites(corpus):
        got = {}
        for cid in extract.client_ids(corpus):
            for c in extract.extract_rules(extract.passages(corpus, cid), cid):
                got.setdefault(c.referent, set()).add(cid)
        return got

    e, h = referent_sites(easy), referent_sites(hard)
    assert len(e["monitoring_interval"]) == 3
    assert len(h.get("monitoring_interval", set())) < 3, "numbers-as-words"
    assert "protocol_effective_date" not in h, "non-ISO date should not parse"
    assert "nurse_ratio_day" not in h, "rephrased ratio should not match"
    # Role names and enum values survive rewording — they are lexical anchors.
    assert len(h["escalation_authority"]) == 3


def test_model_loading_survives_the_transformers_dtype_rename():
    """`torch_dtype` was renamed to `dtype`. Newer transformers warns on the
    old name; older ones reject the new one outright with
    `TypeError: got an unexpected keyword argument 'dtype'`.

    Colab ships new, this laptop ships 4.54, and pinning a version one of them
    cannot satisfy is not an option. Found because `make llm-hard` had no
    cached ledger to replay and therefore actually loaded a model — every other
    path was replaying and never exercised the loader.
    """
    import inspect

    import local_llm

    src = inspect.getsource(local_llm.load)
    assert "torch_dtype" in src, "the fallback for older transformers is gone"
    assert 'kwargs.pop("dtype")' in src




def test_ledger_replay_survives_the_corpus_being_re_offset():
    """A ledger keyed on (client, doc, start) empties silently when a corpus is
    restructured — and reports it as the model having extracted nothing.

    That happened: embedding the hard corpus's passages inside full documents
    moved every `start`, and four configurations flipped from real numbers to
    `0 of 0` with no error. Ledgers now carry their own passage text.
    """
    import local_llm

    cached = {
        "raw": {"site-a/doc:999": '[{"referent":"r"}]'},
        "texts": {"site-a/doc:999": "A passage whose offset has since moved."},
    }
    call = local_llm.replay(cached)
    assert call("Passage:\nA passage whose offset has since moved.\n\nJSON:") \
        == '[{"referent":"r"}]'

    # Ledgers predating `texts` must still work via the offset key.
    legacy = {"raw": {"site-a/protocol_v3:119": "[]"}}
    assert local_llm.replay(legacy, CORPUS) is not None


# ------------------------------------------------------------ silo side


def test_one_silo_alone_produces_no_pairs():
    """Why relation typing cannot live in a silo.

    `block()` yields a pair only across different `client_id`s — cross-source
    disagreement is the whole signal. So a single silo's claims produce zero
    pairs and zero edges, and an NLI head loaded there would have nothing to
    compare. DeBERTa belongs at the aggregator, once, over claims that have
    already arrived from several sources.
    """
    hard = str(Path(CORPUS).parent / "corpus_hard.json")
    corpus = extract.load_corpus(hard)

    alone = extract.extract_rules(extract.passages(corpus, "site-a"), "site-a")
    assert alone, "site-a should extract something"
    assert list(block(alone)) == []
    assert claimguard_merge(alone).edges == ()

    everyone = [c for cid in extract.client_ids(corpus)
                for c in extract.extract_rules(extract.passages(corpus, cid), cid)]
    assert list(block(everyone)), "three silos together must produce pairs"


def test_silo_emits_pointers_and_no_document_text():
    """What crosses the boundary, asserted by fragment rather than by eye."""
    import silo

    hard = str(Path(CORPUS).parent / "corpus_hard.json")
    corpus = extract.load_corpus(hard)
    outbound = []
    for site in extract.client_ids(corpus):
        claims, _ = silo.emit(hard, site)
        outbound.extend(json.dumps(c.to_wire(0)) for c in claims)
    blob = "\n".join(outbound)

    for cid in extract.client_ids(corpus):
        for p in extract.passages(corpus, cid):
            words = p["text"].split()
            for i in range(max(0, len(words) - 7)):
                assert " ".join(words[i : i + 8]) not in blob

    for line in outbound:
        for span in json.loads(line)["spans"]:
            assert span["quote"] is None
            assert span["end"] > span["start"]


def test_silo_refuses_to_emit_an_unresolvable_pointer():
    """The provenance check runs where the documents are, before sending.

    The aggregator has no documents and cannot verify a locator; the silo can,
    and it is the last place anyone can. A pointer that does not resolve is a
    fabricated citation.
    """
    import silo

    hard = str(Path(CORPUS).parent / "corpus_hard.json")
    docdir = str(Path(CORPUS).parent / "documents_hard")

    claims, _ = silo.emit(hard, "site-a")
    assert silo.check_pointers(claims, hard, "site-a", docdir) == []

    # Move one span past the end of its document.
    claims[0].spans[0] = SpanRef(claims[0].spans[0].doc_id, 10_000, 10_050)
    problems = silo.check_pointers(claims, hard, "site-a", docdir)
    assert problems and "runs past end" in problems[0]


# ------------------------------------------------- SiloSafe gate adapter


def test_abstention_on_a_load_bearing_claim_is_not_agreement():
    """The precise failure this project exists to prevent, in the adapter.

    A gate that maps "no escalation" to READY reports `4 hours` vs
    `240 minutes` as alignment. It is not alignment — Layer 1 declined to
    judge. Rendering "I could not decide" as "they agree" is the silent failure,
    and it is one line of adapter logic away at all times.
    """
    import gate

    four_h = Claim("a", "s1", "interval", "interval is", "4",
                   ValueType.NUMERIC, [SpanRef("d", 0, 9)], unit="hours",
                   load_bearing=True)
    two40_m = Claim("b", "s2", "interval", "interval is", "240",
                    ValueType.NUMERIC, [SpanRef("d", 0, 9)], unit="minutes",
                    load_bearing=True)

    report = gate.evaluate([four_h, two40_m])
    assert report.verdict == gate.INCOMPLETE, report.reason
    assert "declined" in report.reason
    assert report.abstentions

    # Not load-bearing: an abstention nothing acts on does not block.
    idle = [Claim("c", "s1", "r", "p", "4", ValueType.NUMERIC,
                  [SpanRef("d", 0, 9)], unit="hours"),
            Claim("d", "s2", "r", "p", "240", ValueType.NUMERIC,
                  [SpanRef("d", 0, 9)], unit="minutes")]
    assert gate.evaluate(idle).verdict == gate.READY


def test_no_pair_formed_is_incomplete_not_ready():
    """An extractor that names the same concept differently at two sites
    produces zero pairs. A gate without this check returns READY having
    compared nothing — measured at 0.5B, where referent drift left 0 of 8 gold
    cases reachable."""
    import gate

    drifted = [
        Claim("a", "s1", "concurrent_anticoagulation", "p", "yes",
              ValueType.ENUM, [SpanRef("d", 0, 9)], load_bearing=True),
        Claim("b", "s2", "concurrent_anticoagulant", "p", "no",
              ValueType.ENUM, [SpanRef("d", 0, 9)], load_bearing=True),
    ]
    report = gate.evaluate(drifted)
    assert report.verdict == gate.INCOMPLETE
    assert "nothing was compared" in report.reason


def test_case_report_renders_polarity_and_carries_no_document_text():
    """`structural:negation` means both values are the SAME string. Printed
    raw, the conflict reads as two identical values and the tool looks broken.
    """
    import gate

    report = gate.evaluate([
        Claim("a", "s1", "anticoag", "therapy is", "permitted",
              ValueType.BOOLEAN, [SpanRef("d1", 0, 9)], load_bearing=True),
        Claim("b", "s2", "anticoag", "therapy is", "permitted",
              ValueType.BOOLEAN, [SpanRef("d2", 0, 9)], polarity=False,
              load_bearing=True),
    ])
    text = report.for_model()
    assert report.verdict == gate.HOLD
    assert "NOT 'permitted'" in text
    for esc in report.escalations:
        assert all(s["quote"] is None for s in esc["spans"])


# ------------------------------------------------- SiloSafe contract


def test_silosafe_contract_is_not_modified():
    """The adapter must work with the closed schema, not around it.

    `_exact_keys` rejects extra fields and `test_silo_report_rejects_quote_field`
    asserts `quote` is refused. Both are deliberate. Anything here that needed
    an eighth field would be a proposal to break their tests, not an adapter.
    """
    import silosafe

    record = {"claim_id": "b1", "referent": "monitoring_interval", "value": "6",
              "value_type": "numeric", "unit": "hours",
              "evidence_token": "ev_abc123", "confidence": 0.9}
    assert set(record) == silosafe.CONTRACT_KEYS

    claim = silosafe.to_claim(record, "site-b")
    assert claim.client_id == "site-b"
    assert claim.value == "6"

    with pytest.raises(ValueError, match="unexpected keys"):
        silosafe.to_claim({**record, "quote": "leaked"}, "site-b")


def test_evidence_token_is_a_commitment_not_a_locator():
    """The distinction the whole integration turns on.

    A token proves the silo did not change its story. It cannot be resolved to
    a location, so it is carried as provenance and rendered as a commitment —
    never as `doc:start-end`, which would claim more than it can deliver.
    """
    import silosafe

    claim = silosafe.to_claim(
        {"claim_id": "b1", "referent": "r", "value": "6", "value_type": "numeric",
         "unit": "h", "evidence_token": "ev_abc123", "confidence": 0.9},
        "site-b",
    )
    span = claim.spans[0]
    assert silosafe.is_commitment(span)
    assert silosafe.token_of(span) == "ev_abc123"
    assert span.start == span.end == 0, "a commitment has no extent"
    assert span.quote is None


def test_a_fabricated_quote_mints_a_valid_token_and_checked_minting_refuses_it():
    """The gap, and the fix, in one test.

    Nothing in the digest checks `quote ⊆ record(record_id)`, so an invented
    quote produces a well-formed, verifiable-looking token. Only the silo holds
    the record, so only the silo can catch it — after emission nobody can.
    """
    import silosafe

    record = "The minimum monitoring interval is 6 hours. Deviations logged."

    fabricated = silosafe.mint("c1", "site-b", "rec-7", "interval is 90 minutes")
    assert fabricated.startswith("ev_") and len(fabricated) == 19, \
        "a fabricated quote mints a perfectly well-formed token"

    assert silosafe.mint_checked(
        "c1", "site-b", "rec-7", "interval is 6 hours", record
    ).startswith("ev_")
    with pytest.raises(ValueError, match="quote not found"):
        silosafe.mint_checked(
            "c1", "site-b", "rec-7", "interval is 90 minutes", record
        )


def test_reveal_lets_a_reviewer_verify_without_holding_silo_data():
    """Selective disclosure: the preimage, per escalation, on request."""
    import silosafe

    token = silosafe.mint("c1", "site-b", "rec-7", "interval is 6 hours")
    honest = silosafe.Reveal(token, "c1", "site-b", "rec-7", "interval is 6 hours")
    altered = silosafe.Reveal(token, "c1", "site-b", "rec-7", "interval is 8 hours")

    assert silosafe.check_reveal(token, honest)
    assert not silosafe.check_reveal(token, altered)


def test_missing_polarity_turns_a_contradiction_into_consensus():
    """The contract's one real blind spot, pinned so it cannot be forgotten.

    With no `polarity` field, `permitted` and `not permitted` arrive as the
    same value string and the gate reads agreement. The information was
    destroyed upstream; no aggregator can recover it.
    """
    import silosafe

    def rec(cid, value):
        return {"claim_id": cid, "referent": "concurrent_anticoagulant",
                "value": value, "value_type": "boolean", "unit": None,
                "evidence_token": f"ev_{cid}", "confidence": 0.9}

    both = (silosafe.to_claims([rec("b1", "permitted")], "site-b",
                               load_bearing=["concurrent_anticoagulant"])
            + silosafe.to_claims([rec("c1", "permitted")], "site-c",
                                 load_bearing=["concurrent_anticoagulant"]))
    import gate
    assert gate.evaluate(both).verdict == gate.READY, "reads as agreement"

    # Folding the negation into `value` restores the conflict, no schema change.
    fixed = (silosafe.to_claims([rec("b1", "not_permitted")], "site-b",
                                load_bearing=["concurrent_anticoagulant"])
             + silosafe.to_claims([rec("c1", "permitted")], "site-c",
                                  load_bearing=["concurrent_anticoagulant"]))
    assert gate.evaluate(fixed).verdict == gate.HOLD
    assert "polarity" in silosafe.polarity_note()
