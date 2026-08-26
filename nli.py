"""Task C - the Layer 2 adapter. Self-contained.

`claimguard.resolve()` declares the seam:

    NliFn = Callable[[str, str], tuple[str, float]]

This module supplies one, over a cached cross-encoder, with no dependency on
anything outside this folder.

## What was taken from the sibling repo, and what was left

The label-mapping approach here follows `contra/src/contra/judges.py`
(`NLIJudge`, `_LABEL_ALIASES`) from the `fl-artifact` project, along with its
measured model choice. That logic is reproduced rather than imported so this
repo is one shippable folder.

What was deliberately **not** brought across is `NLIJudge`'s span machinery -
candidate-span chunking, per-span scoring, joint scoring of the top-k. That
exists to judge a hypothesis against a *document*, and the server here never
holds a document. Claims arrive as pointers with quotes withheld, so the only
thing Layer 2 can see is the claim string. Dropping the chunking drops exactly
the part this design cannot use.

Cost of the short premise, measured on gold case K4:

    "escalation goes to on-call consultant" vs "... duty registrar"
        stitched claim strings   contradiction 0.995
        full passage text        contradiction 0.999

Both far clear of the 0.70 threshold, so nothing is lost by the boundary that
forces the short form.

## Why the label map is the part worth keeping

Label order is not a constant. Checkpoints do not agree on which logit index
means "contradiction", and none is obliged to match MNLI's canonical order.
Hard-coding `["contradiction", "entailment", "neutral"]` yields a permutation
that never raises, never logs, and reports entailment as contradiction. So the
mapping is read from `config.id2label` at load time, and a checkpoint whose
labels cannot be mapped raises instead of guessing.

## Thresholding happens once

`resolve()` owns the threshold and needs the real score to record
`nli:low-confidence` honestly. This module therefore returns the raw argmax
probability and applies no cutoff of its own. One thresholding decision, taken
by the layer that owns the policy.
"""

from __future__ import annotations

import os

from claimguard import NliFn

# Pinned. An unpinned checkpoint is a silent behaviour change on the one
# morning you cannot afford one.
#
# Model choice follows fl-artifact's measured bake-off over technical prose
# (numpy heads 0.43 / nli-deberta-v3-small 0.57 / this 0.71). The gap is
# training data rather than size: MNLI alone is everyday prose, while FEVER
# adds fact-verification against evidence and ANLI adds adversarial pairs.
MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"

# Alternate, ~280 MB, measurably weaker. For the machine that cannot spare
# the disk.
MODEL_SMALL = "cross-encoder/nli-deberta-v3-small"
REVISION_SMALL = "fa2804872c3b4bd748f38c0185cc85775361e735"

# Whatever a checkpoint calls its labels -> what `claimguard.resolve()` speaks.
_LABEL_ALIASES = {
    "entailment": "entailment",
    "entail": "entailment",
    "entails": "entailment",
    "contradiction": "contradiction",
    "contradict": "contradiction",
    "contradicts": "contradiction",
    "neutral": "neutral",
    "not_mentioned": "neutral",
    "not_entailment": "neutral",
}

_ENGINE = None  # loaded once, on first use - never per call


def _is_cached(model: str, revision: str | None) -> bool:
    """Are this checkpoint's files already on local disk?"""
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(model, revision=revision, local_files_only=True)
        return True
    except Exception:
        return False


class NliUnavailable(RuntimeError):
    """Raised with an actionable message, never swallowed into a wrong answer."""


class _CrossEncoder:
    """Tokeniser + model + the label map, loaded once."""

    def __init__(self, model: str, revision: str | None, offline: bool = True) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=offline
        )
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            model, revision=revision, local_files_only=offline
        )
        self.mdl.eval()
        torch.set_grad_enabled(False)

        id2label = getattr(self.mdl.config, "id2label", {}) or {}
        self.labels: dict[int, str] = {}
        for idx, name in id2label.items():
            mapped = _LABEL_ALIASES.get(str(name).strip().lower())
            if mapped is not None:
                self.labels[int(idx)] = mapped
        if not self.labels:
            raise NliUnavailable(
                f"{model} reports labels {list(id2label.values())}, none of which "
                f"map to entailment/contradiction/neutral. Pick a 3-way NLI "
                f"checkpoint, e.g. {MODEL} or {MODEL_SMALL}."
            )

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Full three-way distribution for one pair."""
        enc = self.tok(
            [premise], [hypothesis],
            return_tensors="pt", truncation=True, padding=True, max_length=256,
        )
        probs = self._torch.softmax(self.mdl(**enc).logits, dim=-1)[0]
        out = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
        for idx, label in self.labels.items():
            out[label] = max(out[label], float(probs[idx]))
        return out


def _load(model: str = MODEL, revision: str | None = REVISION) -> _CrossEncoder:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    # Flower forks worker processes; a tokenizer touched before the fork
    # deadlocks the child unless parallelism is off.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Offline is decided from whether the checkpoint is already cached, not
    # forced through the global environment. Cached means no network at all,
    # so bad wifi cannot hang the demo; uncached means fetch it once. Forcing
    # `HF_HUB_OFFLINE=1` from inside a library breaks every caller that has
    # not warmed its cache yet, and reports it as a Hugging Face access error.
    offline = _is_cached(model, revision)

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise NliUnavailable(
            "torch/transformers are not installed, so Layer 2 is unavailable.\n"
            "  fix: pip install torch transformers\n"
            "  without it the demo still runs; gold cases K4 and K8 degrade "
            "to UNDECIDED."
        ) from exc

    try:
        _ENGINE = _CrossEncoder(model, revision, offline=offline)
    except NliUnavailable:
        raise
    except Exception as exc:
        raise NliUnavailable(
            f"could not load {model} at revision {revision}: {exc}\n"
            f"  fix: python scripts/preflight.py --warm"
        ) from exc
    return _ENGINE


def cross_encoder(model: str = MODEL, revision: str | None = REVISION) -> NliFn:
    """Build the `NliFn` that `resolve(claims, nli=...)` expects.

    Returns `(label, score)` where label is one of
    `entailment | contradiction | neutral` and score is that label's
    probability. Thresholding is the caller's business.
    """
    engine = _load(model, revision)

    def nli(premise: str, hypothesis: str) -> tuple[str, float]:
        probs = engine.score(premise, hypothesis)
        best = max(probs, key=lambda k: probs[k])
        return (best, probs[best])

    return nli


def available() -> bool:
    """Can Layer 2 run at all? Used by the CLI to degrade rather than crash."""
    try:
        _load()
        return True
    except NliUnavailable:
        return False


def load_or_none() -> NliFn | None:
    """The CLI's preferred entry point: an `NliFn`, or None with a warning."""
    try:
        return cross_encoder()
    except NliUnavailable as exc:
        print(f"warning: {exc}")
        return None
