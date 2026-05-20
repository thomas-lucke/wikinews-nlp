# 0004 — Sequential model loading

**Date:** 2026-05-12
**Status:** Accepted
**Branch:** main

---

## Context

The pipeline uses five transformer models across four stages:

| Stage | Model | Approx. size |
|---|---|---|
| NER (English) | `dslim/bert-base-NER` | ~430 MB |
| NER (German) | `Davlan/bert-base-multilingual-cased-ner-hrl` | ~700 MB |
| Summarization | `facebook/bart-large-cnn` | ~1.6 GB |
| Similarity | `sentence-transformers/all-MiniLM-L6-v2` | ~90 MB |
| Topic prediction | `facebook/bart-large-mnli` | ~1.6 GB |

Loaded simultaneously, that's ~4.5 GB of model weights in RAM (or GPU VRAM), before any activations or KV caches. Consumer GPUs commonly have 6–8 GB; even a 12 GB card has only modest headroom once activations are accounted for.

Loading everything up-front is the simplest pattern. It also reliably OOMs on most target hardware.

---

## Decision

Load one model at a time. Use it. Free its memory. Load the next one. Concretely:

- Each pipeline stage's cell ([Cell 11](../../notebooks/analysis.ipynb) English NER, [Cell 12](../../notebooks/analysis.ipynb) German NER, [Cell 14](../../notebooks/analysis.ipynb) summarization, etc.) loads its model at the top, runs inference, then explicitly evicts:

  ```python
  del en_ner_pipeline
  release_model()
  ```

- [`src/utils.py:46`](../../src/utils.py#L46) `release_model()` runs `gc.collect()` and `torch.cuda.empty_cache()` to force a real memory reclaim, not just Python's lazy garbage collection.
- The notebook cells are organised in execution order so each model is needed exactly once before being evicted.

---

## Consequences

**Positive**
- The pipeline runs end-to-end on machines with ~4 GB free RAM (and ~6 GB VRAM on GPU), which is the realistic floor for the target audience.
- Peak memory is bounded by the *single* largest model (~1.6 GB for BART), not by the sum of all five (~4.5 GB).
- The pattern is explicit at the call site — readers see `del pipeline; release_model()` and understand the memory contract immediately, no hidden lifecycle magic.

**Negative / Limitations**
- Re-running a cell that's already executed reloads the model from disk → RAM. On an SSD that's a few seconds per model; on slower disks it can be ~10 seconds. This is what users perceive as "the model is loading every time" — it's the disk → RAM cost, not a redownload (see [Cell 14 / FIX-related discussion](../code_implementation_issues.md)).
- The pattern requires discipline: forgetting `release_model()` after one stage causes the next stage to OOM on constrained hardware. Worth noting in code review.
- During active iterative development on a single stage, the explicit eviction is occasionally inconvenient — you might want to keep a pipeline resident for repeated experiments. The fix in that situation is to comment out `del` and `release_model()` locally (don't commit the change).

---

## Alternatives Considered

**Keep all models resident.** Simplest, fastest re-execution. Rejected — OOMs on the target hardware floor.

**Use only smaller models.** E.g. `distilbert-base-NER`, distilled BART variants. Rejected for the NER/summarization quality reasons documented in [ADR 0001](0001-huggingface-for-ner.md) and [ADR 0002](0002-english-only-summarization.md).

**Process articles in mini-batches across models in a tight loop** (load NER, score N articles, unload, load summarizer, summarize N, unload, ...). Rejected as over-complex for a notebook-driven analysis where each stage's full output feeds the next stage's dataframe construction.

**Use HuggingFace's `accelerate` / `device_map="auto"` to swap layers between CPU and GPU.** Rejected — adds dependency and complexity; the manual sequential pattern works fine for five small-to-medium models.

---

## References

- `release_model()` implementation: [`src/utils.py:46`](../../src/utils.py#L46)
- Cells that follow the pattern: 11, 12, 14, 15, 17 in [`notebooks/analysis.ipynb`](../../notebooks/analysis.ipynb)
- Related discussion of disk → RAM reload cost: FIX-4 area in [`docs/code_implementation_issues.md`](../code_implementation_issues.md)
