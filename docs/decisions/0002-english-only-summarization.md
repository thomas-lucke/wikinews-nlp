# 0002 — English-only summarization

**Date:** 2026-05-12
**Status:** Accepted
**Branch:** main

---

## Context

The project does cross-lingual analysis: NER and topic prediction run on both English and German articles. The summarization stage (Task 4) could plausibly do the same — but the available abstractive summarization models are not equal across languages.

The relevant options for German were:

| Option | Note |
|---|---|
| `facebook/bart-large-cnn` | English only. CNN/DailyMail fine-tune. Industry standard for news summarization. |
| `Einmalumdiewelt/T5-Base_GNAD` and similar German-specific T5 models | Smaller training corpora; output quality on news is uneven in spot-checks. |
| `facebook/mbart-large-50-many-to-many-mmt` | Multilingual but trained for translation, not summarization. Repurposing it requires task-specific fine-tuning we are not doing here. |
| `csebuetnlp/mT5_multilingual_XLSum` | Multilingual summarization; smaller than BART, trained on a wider corpus but with lower per-language quality. |

The assignment requires summarisation of 10–20 articles per category across at least 3 categories. The bar is *quality*, not *language coverage breadth*.

---

## Decision

Run the summarization stage on English articles only, using `facebook/bart-large-cnn`. The NER, similarity, and topic prediction stages remain multilingual (en + de). Config: [`languages.summarization: ["en"]`](../../config/config.yaml#L22).

The decision is enforced inside `summarize_articles` ([`src/summarizer.py:85`](../../src/summarizer.py#L85)) by checking `article.get("language") in config["languages"]["summarization"]` and skipping non-English articles. Articles in other languages reach the function but exit without a `summary` field.

---

## Consequences

**Positive**
- Summary quality is high and consistent: BART-large-cnn was fine-tuned on a large English news corpus that matches the input domain well.
- Downstream similarity analysis ([`src/similarity.py`](../../src/similarity.py)) has fewer cross-lingual artefacts to interpret.
- The dual-pass normalisation (NER pool vs. summarization pool, [`notebooks/analysis.ipynb`](../../notebooks/analysis.ipynb) Cells 7–8) is justified by this choice — different language sets mean different valid article pools.

**Negative / Limitations**
- The "compare countries / languages" angle does not extend to summarisation. We cannot say anything about whether English vs. German news summarises better — only that the chosen English summariser works well.
- One full pipeline stage (Cell 14) processes only ~60 articles where it could theoretically process ~120 if German were included.
- Adding German summarization later would require both a model swap and a re-run of validation; not a free extension.

---

## Alternatives Considered

**Use `mT5_multilingual_XLSum` for both languages.** Rejected — smaller model, lower per-language quality on news. The point of summarisation in this project is to demonstrate quality, not breadth.

**Use a German-specific T5 in parallel.** Rejected — running two different models with different output styles would complicate the grammar/style evaluation in [`build_summary_quality_dataframe`](../../src/summarizer.py#L149); the heuristics there assume English punctuation conventions.

**Skip summarization entirely for the second language.** This is essentially the chosen approach, just framed differently. The config makes the constraint explicit (`languages.summarization: ["en"]`) rather than implicit.

---

## References

- BART paper: https://arxiv.org/abs/1910.13461
- `facebook/bart-large-cnn` model card: https://huggingface.co/facebook/bart-large-cnn
- Implementation: [`src/summarizer.py`](../../src/summarizer.py); language gate at [`src/summarizer.py:85`](../../src/summarizer.py#L85).
- Configuration: [`config/config.yaml`](../../config/config.yaml) line 22 (`languages.summarization`).
