# 0001 — HuggingFace BERT-based NER

**Date:** 2026-05-12
**Status:** Accepted
**Branch:** main

---

## Context

The project requires named entity recognition across English and German Wikinews articles. The assignment lists three viable approaches:

- spaCy's built-in NER (`en_core_web_sm`, `de_core_news_sm`)
- DeepPavlov
- A multilingual transformer model

The relevant constraints:

| Constraint | Implication |
|---|---|
| Two languages (en + de) | Need either one multilingual model or two language-specific models. |
| Accuracy matters — Task 3 explicitly asks for error analysis | A weaker model would produce a non-representative error distribution. |
| Runs on CPU and consumer GPU | Model size matters; 7B+ parameter models are out. |
| Standardised entity labels (PER / ORG / LOC / MISC) | Whatever is chosen must produce a comparable label schema. |

spaCy's German model (`de_core_news_sm`) was known to underperform on the Wikinews corpus in informal pre-testing — frequent label confusion between `PER` and `LOC` for German political figures.

---

## Decision

Use HuggingFace transformer-based NER pipelines via `transformers.pipeline("ner", ...)`:

- **English:** `dslim/bert-base-NER` (~110M parameters, CoNLL-2003 fine-tuned, F1 ≈ 0.91 on the test set).
- **German:** `Davlan/bert-base-multilingual-cased-ner-hrl` (~177M parameters, multilingual BERT fine-tuned on HRL languages including German, MISC/PER/ORG/LOC labels).

Both pipelines use `aggregation_strategy="simple"` to merge subword tokens back into entity spans, producing one entity per logical mention rather than one per subword piece.

---

## Consequences

**Positive**
- Both models share the standard `PER / ORG / LOC / MISC` label schema, so downstream code (entity DataFrame, error analysis) treats them uniformly.
- The English model is small enough to run on CPU in reasonable time (~2 minutes for 300 articles); the German multilingual model is larger but still CPU-feasible.
- HuggingFace's `pipeline` abstraction lets the same `run_ner()` function consume both — see [`src/ner.py:195`](../../src/ner.py#L195).
- The chunking layer ([`_chunk_text`](../../src/ner.py#L46)) keeps each input under the model's 512-token context window even for long articles.

**Negative / Limitations**
- ~1.1 GB combined model footprint on disk; ~1.2 GB peak RAM during inference. Requires the sequential-loading pattern documented in [ADR 0004](0004-sequential-gpu-loading.md).
- Label schema is fixed by the upstream models — no fine-grained "political entity" or "event" labels. Those fall under `ORG` or `MISC` per the standard CoNLL conventions.
- The German model is multilingual rather than German-specific. A dedicated German NER model (e.g. `flair/ner-german`) could plausibly outperform it; this was not benchmarked.

---

## Alternatives Considered

**spaCy for both languages.** Rejected. The German model (`de_core_news_sm`) showed notable PER/LOC confusion on early test articles, and its label schema (`PERSON / NORP / ORG / LOC / GPE / ...`) differs from the BERT models' schema — mixing them would complicate the comparison Task 3 asks for.

**DeepPavlov.** Rejected as heavier on setup (custom configs, additional download mechanism) for marginal accuracy gain over the HuggingFace BERT baseline on news-domain text.

**One multilingual model for both languages.** The German model *is* multilingual, but running English text through `Davlan/bert-base-multilingual-cased-ner-hrl` yields lower F1 than the dedicated English model. Two language-specific pipelines outperform one universal one in this size class.

---

## References

- spaCy NER documentation: https://spacy.io/usage/linguistic-features#named-entities
- `dslim/bert-base-NER` model card: https://huggingface.co/dslim/bert-base-NER
- `Davlan/bert-base-multilingual-cased-ner-hrl` model card: https://huggingface.co/Davlan/bert-base-multilingual-cased-ner-hrl
- Implementation: [`src/ner.py`](../../src/ner.py); configuration: [`config/config.yaml`](../../config/config.yaml) lines 25–26.
