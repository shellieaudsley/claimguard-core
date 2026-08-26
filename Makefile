# Layer 2 needs a cached cross-encoder; everything else is standard library.
export HF_HUB_OFFLINE := 1
export TOKENIZERS_PARALLELISM := false
PY ?= python3

.PHONY: help accept accept-hard table verify silo test lint clean

help:
	@echo "verify       the reviewer-verification demo: pointers -> text, per site"
	@echo "accept       the nine gold cases (add NLI=local for 9/9)"
	@echo "accept-hard  the same table over prose written to break extractors"
	@echo "table        the headline table (add NLI=local for the README numbers)"
	@echo "silo         3 silo runs -> aggregation; only JSONL crosses"
	@echo "test         the suite"

accept:
	$(PY) acceptance.py --nli $(or $(NLI),none)

accept-hard:
	$(PY) acceptance.py --corpus corpus_hard.json --nli $(or $(NLI),none)

table:
	$(PY) table.py --nli $(or $(NLI),none) $(ARGS)

silo:
	@for s in site-a site-b site-c; do \
	  $(PY) silo.py --site $$s --corpus corpus_hard.json --check documents_hard \
	    --out claims/$$s.claims.jsonl; done
	@cat claims/site-*.claims.jsonl > claims/all.claims.jsonl
	@$(PY) claimio.py --in claims/all.claims.jsonl --edges claims/edges.jsonl \
	   --escalations claims/escalations.jsonl

verify: silo
	@echo
	@for s in site-b site-c site-a; do \
	  $(PY) silo.py --site $$s --resolve claims/escalations.jsonl \
	    --check documents_hard; done

test:
	$(PY) -m pytest

lint:
	ruff check $(filter-out claimguard.py,$(wildcard *.py)) tests

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ tests/__pycache__
