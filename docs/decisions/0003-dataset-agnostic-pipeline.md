# 0003 — Dataset-agnostic ingestion

**Date:** 2026-05-12
**Status:** Accepted
**Branch:** main

---

## Context

The assignment is scoped to the Wikinews multilingual dataset (one JSONL file, ~15,200 records). The simplest implementation hardcodes `pd.read_json(..., lines=True)` and Wikinews-specific field names (`categories`, `event_id`, `title`, ...).

That approach has two downsides:

1. **Reusability.** Future iterations or related projects might want the same NER / summarisation / similarity pipeline on a different news corpus. Hardcoding the loader couples the pipeline to one source.
2. **Validation.** If the dataset format changes (e.g. Wikinews ships a CSV update), every ingestion call site breaks at once.

Against that, full abstraction is over-engineering for a coursework project. The cost of supporting *every* possible format is real; the cost of supporting a small generic subset is small.

---

## Decision

Make ingestion format-agnostic at a narrow, explicit boundary:

- [`src/data_inspector.py:141`](../../src/data_inspector.py#L141) `detect_format()` inspects the raw path and returns one of:
  `"jsonl"`, `"json"`, `"csv"`, `"tsv"`, `"directory-of-txt"`, or `"unknown"`.
- [`src/data_inspector.py:257`](../../src/data_inspector.py#L257) `load_raw_records()` dispatches on the detected format to a per-format parser.
- [`src/data_normalizer.py`](../../src/data_normalizer.py) maps source-specific field names to a canonical schema via `FIELD_MAPPINGS` and `_select_from_categories` / `_extract_country_from_categories` helpers.
- [`config/config.yaml`](../../config/config.yaml) holds dataset-specific values (source URL, raw path, length thresholds, topic/country lists) — not the loader code.

Wikinews-specific logic (resolving topic and country from the `categories` list, recognising MediaWiki markup) is contained to two functions in [`src/data_normalizer.py`](../../src/data_normalizer.py) and one in [`src/preprocessing.py`](../../src/preprocessing.py). The rest of the pipeline sees only the canonical schema.

---

## Consequences

**Positive**
- Swapping in a different newslike corpus is a config-and-mapping change, not a code rewrite. Most of the pipeline (NER, summarization, similarity, topic prediction) is dataset-agnostic by construction.
- `detect_format()` has its own unit tests ([`tests/test_data_inspector.py`](../../tests/test_data_inspector.py)) and validates by attempting to parse one record before committing to a format — bad files fail fast.
- The dual-pass normalisation pattern (NER pool vs. summarization pool, see [ADR 0002](0002-english-only-summarization.md)) works without modification because both call the same `normalise_articles` entry point.

**Negative / Limitations**
- The Wikinews-specific `_KNOWN_COUNTRIES` vocabulary and category-parsing logic still live inside `data_normalizer`. A truly source-neutral pipeline would push these into a plugin. This is left intentionally — over-engineering for one project.
- Format detection is heuristic (file extension + first-record parse attempt). Pathological inputs (e.g. a `.csv` file that's actually JSON) can be misclassified; the `_confirm_parse` step in [`src/data_inspector.py:104`](../../src/data_inspector.py#L104) catches most but not all cases.
- "Dataset-agnostic" is aspirational, not absolute — see FIX-1 in [`docs/code_implementation_issues.md`](../code_implementation_issues.md), where the Wikinews-specific quirk of `text` as a list of strings required a join step in `_infer_text_field`.

---

## Alternatives Considered

**Hardcode `pd.read_json(..., lines=True)` for Wikinews only.** Simpler. Rejected because it couples every module's tests to one fixture file format and bakes the source URL into Python rather than YAML.

**Build a full plugin architecture (per-dataset adapters).** Over-engineered. The current `detect_format` + `FIELD_MAPPINGS` approach handles the realistic near-future use cases (other newslike JSONL/CSV corpora) without the cost of a plugin registry.

**Use a generic ETL framework (Prefect / Dagster / Airflow).** Inappropriate for a single-run analysis notebook. Adds complexity and a runtime dependency that earns nothing here.

---

## References

- Implementation: [`src/data_inspector.py`](../../src/data_inspector.py), [`src/data_normalizer.py`](../../src/data_normalizer.py)
- Test suite: [`tests/test_data_inspector.py`](../../tests/test_data_inspector.py) (21 tests), [`tests/test_data_normalizer.py`](../../tests/test_data_normalizer.py) (34 tests)
- Wikinews-specific format quirk: FIX-1 in [`docs/code_implementation_issues.md`](../code_implementation_issues.md)
