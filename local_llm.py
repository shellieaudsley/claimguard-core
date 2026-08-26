"""Task B's model half - a local generative extractor, no API and no network.

`extract.extract()` takes any `call_llm: str -> str`, so this module only has
to supply one. It uses whatever is in the Hugging Face cache. The spec names
`flwrlabs/Lizzy-7B-GGUF` and using the host's own model would be a nice
alignment win, but a 7B GGUF needs llama.cpp and ~4.4 GB of weights, which is
uncomfortable on an 8 GB machine. This runs on what fits.

## Two models, two prompting styles

    Qwen2.5-0.5B            base      494M, fp32, ~58 tok/s   few-shot completion
    Qwen2.5-0.5B-Instruct   instruct  494M, fp32, ~58 tok/s   chat template

Same size, same speed, and the difference in output is the whole point of
having both. A **base** model handed `extract.PROMPT` - which is written as an
instruction - emits end-of-text immediately and returns nothing at all. It only
works as a completion with worked examples. An **instruct** model can be given
the instruction directly, which is what the prompt was written for.

Keeping both selectable is what makes the comparison in
`scripts/measure_llm.py` a controlled one: parameter count is held fixed, so
whatever changes is attributable to instruction tuning rather than capacity.

## The finding the comparison exists to test: referent drift

Open-world, the base model names the same concept differently at different
sites:

    concurrent_anticoagulation   vs the rules' concurrent_anticoagulant
    version                      vs the rules' protocol_effective_date

`claimguard.block()` buckets on exact referent equality. Two sites spelling a
referent differently produce **zero candidate pairs, zero conflicts, and a
clean green run that detected nothing** - a false PASS from the layer whose
entire job is not to produce one.

So the referent is not a free-text field the model may choose. It is the
blocking key, and it should come from a controlled vocabulary. Whether a model
will *obey* that vocabulary is exactly what instruction tuning ought to buy,
and is measured rather than assumed.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# Bumped whenever the module's public surface changes. `scripts/colab.md` has
# the notebook assert it, because the failure it guards against is silent:
# Colab's files.upload() does NOT overwrite an existing file, it saves the new
# one as "local_llm (1).py" and leaves the import resolving to the stale copy.
# The symptom is an AttributeError for a function that is plainly in the file
# you just uploaded.
__version__ = "0.4.0"


@dataclass(frozen=True)
class ModelSpec:
    """A cached checkpoint and how it must be prompted."""

    name: str
    revision: str | None
    chat: bool  # True -> apply_chat_template and give it the instruction
    params_b: float = 0.5   # billions, for the VRAM guard
    load_4bit: bool = False  # quantise on load; needed for 7B on a 16 GB card

    @property
    def key(self) -> str:
        return f"{self.name}:{'4bit' if self.load_4bit else 'full'}"

    def weights_gb(self) -> float:
        return self.params_b * (0.55 if self.load_4bit else 2.0)


MODELS: dict[str, ModelSpec] = {
    # Cached locally. 494M each, so base-vs-instruct holds capacity fixed.
    "base": ModelSpec("Qwen/Qwen2.5-0.5B", None, chat=False),
    "instruct": ModelSpec(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "7ae557604adf67be50417f59c2c2f167def9a775",
        chat=True,
    ),
    # Not cached. Sizes are fp16 weights, i.e. what a GPU needs; on CPU in
    # fp32 they are double that, which is why 7B does not fit an 8 GB laptop.
    # `scripts/colab.md` runs these on a free T4.
    "1.5b": ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", None, True, params_b=1.54),
    "3b": ModelSpec("Qwen/Qwen2.5-3B-Instruct", None, True, params_b=3.09),
    # 7B in fp16 is 15.2 GB of weights against a T4's ~14.7 GB usable, so it
    # OOMs *after* a 15 GB download - the worst possible ordering. 4-bit is
    # the only configuration that fits, and `3b` is the better first
    # experiment either way: a third of the download for most of the jump.
    "7b": ModelSpec("Qwen/Qwen2.5-7B-Instruct", None, True, params_b=7.62,
                    load_4bit=True),
    "7b-fp16": ModelSpec("Qwen/Qwen2.5-7B-Instruct", None, True, params_b=7.62),
}
DEFAULT = "instruct"

_ENGINES: dict[str, Callable[[str], str]] = {}

# Two worked examples for the base model, covering the two fields it is most
# likely to drop: `polarity` on a negation, and `unit` on a bare number.
_EXAMPLES = '''Passage: The notice period for termination is 30 days.
JSON: [{"referent":"notice_period","predicate":"the notice period is","value":"30","value_type":"numeric","unit":"days","polarity":true,"quote":"notice period for termination is 30 days"}]

Passage: Remote work is not permitted for probationary staff.
JSON: [{"referent":"remote_work","predicate":"remote work is","value":"permitted","value_type":"boolean","unit":null,"polarity":false,"quote":"Remote work is not permitted"}]
'''

_TAIL = "\nPassage: @@PASSAGE@@\nJSON:"


# --------------------------------------------------------- token budget

# One JSON record for this schema costs roughly 70 tokens. A passage yields at
# most about one assertion per sentence, so the budget scales with sentence
# count and is clamped at both ends.
#
# Why bother: a fixed cap is wrong in both directions. Too low truncates the
# JSON mid-record on a dense passage - which does not read as an error, it
# reads as the model producing fewer claims, and silently depresses recall on
# exactly the passages that carry the most. Too high spends generation time on
# short passages that had one assertion in them, and on CPU that is the whole
# runtime.
TOKENS_PER_RECORD = 70
BUDGET_FLOOR = 120
BUDGET_CEILING = 700


def token_budget(prompt) -> int:
    """Bounded, deterministic, derived from the passage's own shape."""
    text = prompt[-1]["content"] if isinstance(prompt, list) else prompt
    passage = _passage_from(text) if "Passage:" in text else text
    sentences = max(1, len(re.findall(r"[.!?](?:\s|$)", passage)) or 1)
    return max(BUDGET_FLOOR, min(BUDGET_CEILING, 60 + TOKENS_PER_RECORD * sentences))


def looks_truncated(raw: str) -> bool:
    """Did generation stop before the JSON array closed?

    Reported rather than silently absorbed. A truncated generation and a model
    that genuinely found nothing both parse to zero records, and conflating
    them would make the budget look free when it was costing recall.
    """
    opened = raw.find("[")
    if opened < 0:
        return False
    return raw.rfind("]") < opened


def _vocab_preamble(referents: Sequence[str] | dict[str, str]) -> str:
    """Render the controlled vocabulary, with glosses when they are available.

    A bare identifier is not a definition. Measured: the instruct model obeyed
    an 8-item bare list almost perfectly (1 off-vocabulary referent out of 7)
    while using `consent_window` for three unrelated passages - it complied
    with the list without understanding it. The gloss and the explicit `other`
    escape both exist to attack that, rather than the compliance the model had
    already got right.

    `other` is a real value rather than an instruction to stay silent, because
    "omit the record entirely" gives a weak model no token to emit and it
    reaches for the nearest list item instead. Records tagged `other` are
    dropped downstream, so the escape costs nothing and gives it somewhere to
    put what it cannot classify.
    """
    if isinstance(referents, dict):
        lines = "\n".join(f"  {r:<26} {g}" for r, g in sorted(referents.items()))
    else:
        lines = "\n".join(f"  {r}" for r in sorted(referents))
    return (
        f'Choose "referent" from this list, matching on MEANING:\n{lines}\n'
        f'  {"other":<26} anything not described above\n'
        f'Use "other" whenever the passage does not match one of the named '
        f'referents. Do not force a poor match.\n\n'
    )


def build_prompt(
    spec: ModelSpec, passage: str, referents: Sequence[str] | None
) -> str | list[dict]:
    """The prompt this checkpoint actually needs.

    Instruct: the real instruction from `extract.PROMPT`, as a chat message.
    Base: the same task rewritten as a completion with worked examples.
    """
    import extract as ex

    preamble = _vocab_preamble(referents) if referents else ""

    if spec.chat:
        return [{"role": "user", "content": preamble + ex.PROMPT.format(passage=passage)}]
    return preamble + _EXAMPLES + _TAIL.replace("@@PASSAGE@@", passage)


def is_cached(spec: ModelSpec | str = DEFAULT) -> bool:
    """Are this checkpoint's files already on local disk?"""
    if isinstance(spec, str):
        spec = MODELS[spec]
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(
            spec.name, revision=spec.revision, local_files_only=True
        )
        return True
    except Exception:
        return False


def load(spec: ModelSpec | str = DEFAULT):
    """Load once per checkpoint. Returns a callable over a built prompt.

    Offline is decided per checkpoint from whether it is already cached, and
    is NOT forced through the global environment.

        cached      -> local_files_only=True: no network at all, so bad
                       conference wifi cannot hang the demo
        not cached  -> download it

    An earlier version did `os.environ.setdefault("HF_HUB_OFFLINE", "1")` here.
    That is right on a laptop with warm caches and wrong everywhere else: on a
    fresh Colab runtime it disables all outgoing traffic and every load dies
    with `LocalEntryNotFoundError`, which reads like a Hugging Face access
    problem and is not one. A library should not reach into the process
    environment to make that choice for its caller.
    """
    if isinstance(spec, str):
        spec = MODELS[spec]
    if spec.key in _ENGINES:
        return _ENGINES[spec.key]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _check_reachable(spec)
    _check_fits(spec)
    offline = is_cached(spec)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # fp16 on a GPU, fp32 on CPU. CPU fp16 is emulated and slower, not faster,
    # so this is a correctness-of-speed choice rather than a precision one.
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(
        spec.name, revision=spec.revision, local_files_only=offline
    )
    kwargs = {"revision": spec.revision, "local_files_only": offline}
    if spec.load_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"   # bitsandbytes places it; do not .to()
    else:
        kwargs["dtype"] = dtype

    # `torch_dtype` was renamed to `dtype`. Newer transformers warns on the old
    # name; older ones reject the new one with
    # `TypeError: __init__() got an unexpected keyword argument 'dtype'`.
    # Colab ships new, this laptop ships 4.54 - so support both rather than
    # pinning a version the other environment cannot satisfy.
    try:
        mdl = AutoModelForCausalLM.from_pretrained(spec.name, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc) or "dtype" not in kwargs:
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        mdl = AutoModelForCausalLM.from_pretrained(spec.name, **kwargs)
    if "device_map" not in kwargs:
        mdl = mdl.to(device)
    mdl.eval()
    torch.set_grad_enabled(False)
    print(f"  loaded {spec.name} on {device} "
          f"({'4-bit' if spec.load_4bit and device == 'cuda' else dtype}, "
          f"~{spec.weights_gb():.1f} GB, "
          f"{'from cache' if offline else 'downloaded'}).")

    def run(prompt, max_new_tokens: int | None = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = token_budget(prompt)
        if spec.chat:
            text = tok.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        ids = tok(text, return_tensors="pt").to(device)
        out = mdl.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: the run must be reproducible
            pad_token_id=tok.eos_token_id,
        )
        decoded = tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)
        # The completion format lets a base model run on into a new example.
        # Cut at the next one rather than letting it into the JSON parser.
        return decoded if spec.chat else decoded.split("\nPassage:")[0]

    _ENGINES[spec.key] = run
    return run


def available(spec: ModelSpec | str = DEFAULT) -> bool:
    """Can this checkpoint be run - either cached, or fetchable?

    The previous version answered only "is it cached", with offline forced on.
    On a fresh runtime that made every uncached model report unavailable, and
    `measure_llm.py` skipped all of them with a message saying the cache was
    empty - technically true, and useless, because the fix was to download
    them and nothing offered to.
    """
    if is_cached(spec):
        return True
    return not _hub_offline()


def _hub_offline() -> bool:
    """Is the hub in offline mode right now?

    Checks the LIVE constant, not just the environment. `huggingface_hub` reads
    `HF_HUB_OFFLINE` exactly once, at import, and caches it in
    `constants.HF_HUB_OFFLINE`. After that, `os.environ.pop("HF_HUB_OFFLINE")`
    changes nothing at all - the library never looks at the environment again.

    That is why re-uploading a fixed file into a live notebook does not help:
    the flag was already latched by the first import.
    """
    if os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    try:
        import huggingface_hub.constants as C

        return bool(C.HF_HUB_OFFLINE)
    except Exception:
        return False


def force_online() -> bool:
    """Undo a latched offline flag, in-process. Returns True if it was set.

    Clears both the environment variable and the cached constant, in that
    order. Verified to re-enable downloads without restarting the interpreter.

    Restarting the runtime is still the better move in a notebook, because a
    stale `HF_HUB_OFFLINE` is rarely the only stale thing - `sys.modules` will
    also still hold the old version of whatever module set it.
    """
    was = _hub_offline()
    os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        import huggingface_hub.constants as C

        C.HF_HUB_OFFLINE = False
    except Exception:
        pass
    return was


class ModelUnavailable(RuntimeError):
    """Raised instead of letting transformers produce a 24-frame traceback."""


def _check_fits(spec: ModelSpec) -> None:
    """Refuse a download that cannot possibly load. Checked BEFORE fetching.

    Getting this order wrong is expensive in a way that reads as a mystery:
    15 GB of weights arrive over half an hour and only then does CUDA report
    out-of-memory, by which point the session may also have timed out.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return  # CPU: slow, but bounded by RAM+swap rather than VRAM
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
    except Exception:
        return

    need = spec.weights_gb()
    usable = total * 0.92  # CUDA context and activations
    if need <= usable:
        return
    alt = "7b" if spec.load_4bit is False and spec.params_b > 7 else "3b"
    raise ModelUnavailable(
        f"{spec.name} needs ~{need:.1f} GB of weights and this GPU has "
        f"{total:.1f} GB (~{usable:.1f} usable).\n"
        f"  It would download for many minutes and then fail with CUDA OOM.\n"
        f"  Use MODELS['{alt}'] instead - 4-bit 7B is ~4.2 GB, 3B fp16 ~6.2 GB."
    )


def _check_reachable(spec: ModelSpec) -> None:
    """Fail fast on the one contradiction that produces a misleading error.

    Not cached AND offline is unsatisfiable. Left to `transformers`, it surfaces
    as `LocalEntryNotFoundError` wrapped in `OSError: We couldn't connect to
    'https://huggingface.co'` - which reads as a network or permissions problem
    and is neither. It is a flag somebody set, usually this package's own
    earlier version.
    """
    if is_cached(spec) or not _hub_offline():
        return
    raise ModelUnavailable(
        f"{spec.name} is not cached and the Hugging Face hub is in OFFLINE mode, "
        f"so it cannot be downloaded.\n"
        f"  This is NOT an access problem - Qwen2.5 is public and needs no token.\n"
        f"  In a notebook: Runtime -> Restart session, then re-run. Re-uploading a\n"
        f"    file is not enough; huggingface_hub latches HF_HUB_OFFLINE at import\n"
        f"    and sys.modules still holds the old module.\n"
        f"  Without restarting: local_llm.force_online() clears both, then retry.\n"
        f"  On a laptop this is expected - the Makefile exports HF_HUB_OFFLINE=1."
    )


def make_call_llm(
    spec: ModelSpec | str = DEFAULT,
    referents: Sequence[str] | None = None,
) -> Callable[[str], str]:
    """The callable `extract.extract(call_llm=...)` wants.

    `extract.PROMPT.format(passage=...)` is what gets handed in, so the passage
    is recovered from it and re-wrapped in whatever form this checkpoint needs.
    """
    if isinstance(spec, str):
        spec = MODELS[spec]
    engine = load(spec)

    def call(prompt: str) -> str:
        passage = _passage_from(prompt)
        return engine(build_prompt(spec, passage, referents))

    return call


def _passage_from(prompt: str) -> str:
    """Pull the passage back out of `extract.PROMPT`."""
    marker = "Passage:\n"
    start = prompt.find(marker)
    if start < 0:
        return prompt
    end = prompt.find("\n\nJSON:", start)
    return prompt[start + len(marker) : end if end > 0 else None].strip()


# ------------------------------------------------------------------ cache


def extract_corpus(
    spec: ModelSpec | str = DEFAULT,
    corpus_path: str = "corpus.json",
    cache_path: str = "llm_ledger.json",
    referents: Sequence[str] | None = None,
    refresh: bool = False,
) -> dict:
    """Run the model over every passage once and cache the raw output.

    ~10 minutes per configuration on this machine. Cached because iterating on
    the validator should not cost another ten.
    """
    import extract as ex

    if isinstance(spec, str):
        spec = MODELS[spec]

    if os.path.exists(cache_path) and not refresh:
        with open(cache_path) as fh:
            return json.load(fh)

    call = make_call_llm(spec, referents)
    corpus = ex.load_corpus(corpus_path)
    out: dict = {
        "model": spec.name,
        "chat": spec.chat,
        "corpus": corpus_path,
        "closed_world": referents is not None,
        "glossed": isinstance(referents, dict),
        "truncated": [],
        # Passage text alongside the generation. The key is (client, doc,
        # start), and `start` is exactly what changes when a corpus is
        # restructured - which silently emptied the replay and reported it as
        # the model extracting nothing. Keyed content survives that.
        "texts": {},
        "raw": {},
    }

    t0 = time.time()
    for client_id in ex.client_ids(corpus):
        for p in ex.passages(corpus, client_id):
            key = f"{client_id}/{p['doc_id']}:{p['start']}"
            raw = call(ex.PROMPT.format(passage=p["text"]))
            out["raw"][key] = raw
            out["texts"][key] = p["text"]
            if looks_truncated(raw):
                out["truncated"].append(key)
    out["seconds"] = round(time.time() - t0, 1)

    with open(cache_path, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def replay(cached: dict, corpus_path: str = "corpus.json") -> Callable[[str], str]:
    """A `call_llm` that serves the cache, so the pipeline runs in seconds."""
    import extract as ex

    by_passage = {}

    # Preferred: the ledger carries its own passage text, so replay is immune
    # to the corpus being re-offset between generation and analysis.
    texts = cached.get("texts") or {}
    for key, text in texts.items():
        if key in cached["raw"]:
            by_passage[text] = cached["raw"][key]

    # Fallback for ledgers generated before `texts` existed - match on
    # (client, doc, start) as before.
    if not by_passage:
        corpus = ex.load_corpus(corpus_path)
        for client_id in ex.client_ids(corpus):
            for p in ex.passages(corpus, client_id):
                key = f"{client_id}/{p['doc_id']}:{p['start']}"
                if key in cached["raw"]:
                    by_passage[p["text"]] = cached["raw"][key]

    def call(prompt: str) -> str:
        return by_passage.get(_passage_from(prompt), "[]")

    return call
