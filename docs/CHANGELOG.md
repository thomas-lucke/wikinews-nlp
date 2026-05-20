# Changelog

All notable changes are documented here. One-liners for minor changes; a short paragraph for significant ones. Most recent at the top.

---

## Analysis run & bug fixes — 2026-05-20

### Fixed
- **NER aggregation bug (FIX-10).** Switched `aggregation_strategy` from `"simple"` to `"average"` in both NER pipelines to prevent BERT subword fragments (`##word`) surfacing as entity tokens. Entity text is now extracted from character offsets (`cleaned_text[start:end]`) instead of the lossy `.word` field, recovering entities that were previously silently dropped (e.g. *WikiLeaks*, *Thierry Henry*).
- **Language-only NER grouping (ADR-0005).** The NER normalisation pass now groups by language only; country is retained as metadata. The previous country+language grouping created near-empty strata for most country/topic combinations, distorting the per-group article caps.
- `plot_topic_confusion_matrix`: `ax.text` now receives `str(int(...))` — resolves Pylance `reportArgumentType` error.

### Changed
- Similarity distribution histogram and topic error breakdown now render as vertical stacks (3 rows × 1 col) instead of side-by-side (1 row × 3 cols).
- Cell 17 accuracy note is now dynamic: prints the evaluated sample count and the normalised article pool size rather than a static "small sample" message.
- Cell 18 summary written from actual run outputs, covering dataset scope, NER findings, similarity statistics, and topic prediction results.

### Added
- Five Architecture Decision Records in `docs/decisions/`: HuggingFace for NER (0001), English-only summarisation (0002), dataset-agnostic pipeline (0003), sequential GPU loading (0004), language-only grouping (0005).
- `README.md` with project overview, setup instructions, and notebook execution guide.

---

## CODE v1.0 — 2026-05-12

First complete implementation against SPEC v3.0. All 22 tasks delivered; `uv run pytest tests/` → **146 passed**.

### Added
- All nine `src/` modules implemented and tested: `utils`, `data_loader`, `data_inspector`, `data_normalizer`, `preprocessing`, `ner`, `summarizer`, `similarity`, `topic_predictor`. spaCy, HuggingFace, and SentenceTransformer calls are fully mocked — no model downloads in CI.
- `notebooks/analysis.ipynb` with the 17 cells specified in SPEC §"Notebook orchestration".
- `scripts/review_spec.py` wraps the Anthropic SDK with extended thinking (model `claude-opus-4-7`, 10 k token budget) for spec review.

### Fixed
- **Production bug:** `similarity.calculate_similarity` batched both inputs into one `model.encode()` call. The mock fixture returns `(1, dim)` per call, so the second slice was empty and `cos_sim[0][0]` raised `IndexError`. Fixed by encoding original and summary in separate calls.

### Notes
- All deviations, judgment calls, and fixes are logged in `docs/code_implementation_issues.md` (24 entries across tasks 3–22).
- TLS workaround: `--native-tls` (uv) and `truststore.inject_into_ssl()` required on this machine due to a corporate MITM CA.

---

## SPEC v3.0 — 2026-05-11 to 2026-05-13 (patches 1–5)

Five iterative passes to tighten the spec before and during implementation. No pipeline behaviour or module interfaces were redesigned — all changes are clarifications and gap-fills.

- **Patch 5 (2026-05-13):** Implementation history folded back. Three places where the code contradicted the spec (§5.4 download tightening, §7.3 per-group sort key, §20.1 `evaluate_topic_predictions` empty-title substitution) corrected. 37 under-specified cases now appear in docstrings.
- **Patch 4 (2026-05-11):** Four gaps surfaced during test authoring: `detect_format` 80%-dominance boundary pinned; `summarize_article` pipeline kwarg names fixed; `evaluate_topic_predictions` returns `0.0` accuracy (not `NaN`) when `evaluated == 0`; plotting/loader functions explicitly excluded from unit test scope.
- **Patch 3 (2026-05-11):** Aligned normalisation with the real Wikinews dataset — list-valued `categories` for topics, `pageid` → `event_id` mapping. Added `data_inspector.category_profile()` as a non-interactive selection gate before the user edits `config.yaml`.
- **Patch 2 (2026-05-11):** Removed all forward-looking language. Deleted unused `processed_path` config key.
- **Patch 1 / initial (2026-05-11):** Three review passes — module boundary interfaces, hidden environmental assumptions, and a 35-issue hostile implementation review. Key decisions: FIELD_MAPPINGS precedence order, `_chunk_text` overlap window bounds, `_resolve_overlapping_entities` execution order (dedup before partial-overlap), Cell 7 validation gate raises `RuntimeError` on failure.
