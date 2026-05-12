# CHANGELOG

---

## SPEC v3.0 — 2026-05-11 (patch 4)

Spec gaps surfaced during test authoring (Phase 4 of the AI engineering workflow).
The hostile reviews in Phase 3 caught most ambiguity; these four were the residue
that only became visible when concrete test assertions were attempted.

1. **`detect_format` 80% dominance edge case pinned.** "≥80%" now reads "at least
   80% (count / total >= 0.8 — exactly 80% counts as dominant)" so the boundary
   case (e.g. 4 of 5 files) has a defined outcome.

2. **`summarize_article` pipeline kwarg names pinned.** Added an explicit
   "Pipeline call signature" paragraph requiring the HuggingFace-documented kwarg
   names `truncation`, `min_length`, `max_length` — no renaming, no positional
   calls. Tests assert against these exact names.

3. **`evaluate_topic_predictions` empty-evaluation rule defined.** When
   `evaluated == 0`, accuracy is `0.0` — not `NaN` and not `None`. The notebook
   prints `f"{accuracy:.1%}"` without NaN-handling, so a finite float is required.

4. **"Functions intentionally not covered by unit tests" subsection added.** Four
   categories of functions (matplotlib plotting, environment-dependent fallbacks,
   log-output helpers, thin model-loader wrappers) are now explicitly out of unit
   test scope, with rationale. This converts implicit gaps into documented scope
   decisions and tells test authors not to add placeholder assertions for them.

---

## SPEC v3.0 — 2026-05-11 (patch 3)

Implementation-risk cleanup after a code-generation review.

1. **Aligned normalisation with the real Wikinews dataset.** Added support for
   `categories` as list-valued topics, changed the default topic from `Politics`
   to dataset-native `Politics and conflicts`, and mapped `pageid` to `event_id`
   instead of using it as article identity.

2. **Made format detection work for the target GitHub repo.** Directory detection now
   ignores non-data files such as `README.md`, so one large JSONL data file is enough
   to detect `"jsonl"`.

3. **Relaxed download skip logic for single-file datasets.** Existing raw data now
   skips download when at least one recognised data file over 100KB is present.

4. **Corrected inconsistent contracts.** Clarified two-pass normalisation sampling,
   topic prediction mutation behavior, per-article error fallbacks, and similarity
   score range.

5. **Fixed CI mocking guidance.** Tests now patch module-local aliases
   (`src.ner.hf_pipeline`, `src.ner.get_device`, etc.) instead of patching names that
   are no longer read after import.

6. **Added lightweight summary grammar/style review.** The spec now includes a
   `build_summary_quality_dataframe` helper and notebook display step to satisfy the
   assignment requirement without adding a heavy grammar-checking dependency.

7. **Added required country filtering.** Introduced `countries.selected`, normalised
   `country` article metadata, country validation, country-aware sampling, and
   country-scoped notebook analysis. Country filtering is separate from language
   filtering and uses Wikinews `categories` tags.

8. **Added non-interactive category selection gate.** The spec now includes
   `data_inspector.category_profile()` and `print_category_profile()` so the
   notebook can display available Wikinews category labels/counts before the user
   edits `config.yaml` and continues.

---

## SPEC v3.0 — 2026-05-11 (patch 2)

Scope alignment: removed all forward-looking language. This project is a fixed-scope,
single-delivery implementation. No extension or redesign is planned after submission.

1. **`processed_path` removed from config.yaml schema.** The key was unused by any
   module and annotated "reserved for future use." Removed entirely — no module
   references it.

2. **Known Limitations "No persistent cache" reworded.** Removed "not implemented in
   this version" (implies a future version). Now states this is a deliberate scope
   decision with a concrete manual workaround.

3. **Resolved Open Questions item 5 reworded.** Removed "add a persistence step in a
   future version." Now states "deliberate scope decision for this project."

---

## SPEC v3.0 — 2026-05-11

Three review passes applied to the v3.0 draft. All changes are clarifications or
tightenings of existing decisions; no pipeline behaviour or module interfaces were
redesigned.

---

### Pass 1 — Module boundary interface review (`spec_interview_review.md`)

Resolved ambiguities at every point where one function's output becomes another's input.

1. **Date field contract unified.** Article schema comment changed from "ISO date string
   YYYY-MM-DD" to "raw source date string as-is, not parsed or validated." Data flow
   section cross-references the schema. The two conflicting statements are now one.

2. **ID field contract made explicit.** Article schema `id` comment now states: "This is
   the only ID field in the article dict. All references to 'source id field' elsewhere
   in this spec refer to this key."

3. **Topic casing rule inlined into `validate_normalised`.** Docstring now includes the
   instruction to normalise both config topics and article topics with
   `_normalise_topic_string` before comparing. Previously this was only stated in the
   "Resolved open questions" section.

4. **`summarize_articles` field mapping made explicit.** Docstring now shows the exact
   call `summarize_article(article["cleaned_text"], ...)` and states that non-qualifying
   articles do not receive a `"summary"` key; downstream code must use `.get("summary")`.

5. **`preprocess_articles` unrecognised-language branch clarified.** Added note that in
   the standard notebook pipeline this branch is never taken — it is a safety guard for
   ad-hoc use only.

6. **`evaluate_topic_predictions` return schema pinned.** `results` entries now
   explicitly state that `topic` is lowercase (copied from article) and `predicted_topic`
   is original-cased (from config).

7. **`summary` schema comment corrected.** Article schema comment no longer says "None
   for German articles" — German articles do not receive the key at all.

---

### Pass 2 — Hidden assumptions review (`spec_interview_review2.md`)

Converted implicit environmental assumptions into explicit documented behaviour.

1. **Language code comparison is exact-match.** `normalise_articles` step 3 now states
   that comparison is exact lowercase ISO 639-1. Regional variants (`"en-US"`, `"de-DE"`)
   and full names (`"English"`) will not match and all such records will be dropped.
   Mitigation note added for datasets that use non-standard codes.

2. **`directory-of-txt` sort is locale-independent.** `load_raw_records` now mandates
   Python's default Unicode code-point sort (`sorted()`). Explicitly prohibits
   `locale.strcoll` or any locale-aware sort, with rationale (cross-machine determinism).

3. **`encoding="utf-8"` required for all structured formats.** JSON, JSONL, and CSV/TSV
   now each carry an explicit `encoding="utf-8"` requirement with rationale: silent
   character replacement in structured formats would corrupt field values. The
   `directory-of-txt` format uses `errors="replace"` (unchanged) with a note explaining
   why replacement is acceptable there but not in structured formats.

4. **Cell 7 validation gate hardened.** Notebook now raises `RuntimeError` immediately
   if either `validation_passed` is False, preventing silent continuation on bad data.
   Human review note updated to cover warnings (which do not raise).

---

### Pass 3 — Hostile implementation review (35 issues)

Found and resolved every place where a coding agent would have to guess. Decisions
requiring design input were confirmed before changes were applied.

**Design decisions confirmed:**
- FIELD_MAPPINGS precedence: iterate FIELD_MAPPINGS keys against raw record (not raw
  record keys against FIELD_MAPPINGS).
- Language inference: drop immediately if no language field maps; no inference attempted.
- `preprocess_articles` ordering: clean all texts first, then feed cleaned strings to
  `nlp.pipe()` as a separate pass.
- NER chunk threshold: remains character-based; token-limit overage is a known limitation
  handled by HuggingFace pipeline truncation.
- `plot_top_entities` grouping: by `entity_text` alone (intentional; chart shows name
  frequency, not label distribution).
- Redistribution quota in `predict_all_topics`: fill remaining slots evenly across
  topics with unsampled articles.

**Spec changes applied:**

1. **`_infer_text_field` return type clarified.** Docstring now states it returns the
   field NAME (not the value). Caller extracts value with `raw_record[returned_name]`.
   Exclusion rule added: any key already in FIELD_MAPPINGS is excluded from inference
   candidates regardless of value length.

2. **FIELD_MAPPINGS precedence rule added.** Comment now mandates iterating FIELD_MAPPINGS
   keys against the raw record. Warns against the reverse (iterate raw record keys against
   FIELD_MAPPINGS), which produces source-order-dependent results.

3. **`detect_format` directory detection rewritten.** Detection order now explicit:
   collect dominant extension, then check `.txt` → `directory-of-txt` before other
   extensions. `"zip"` return value documented as unreachable in normal pipeline use.
   Minority-extension files are skipped by `load_raw_records` with a logged warning.

4. **`DroppedRecord.article_index` semantics pinned.** Now explicitly the zero-based
   index in the list returned by `load_raw_records`, stable before any filtering.

5. **Deduplication hash source clarified.** Step 4 of `normalise_articles` now says
   "hash the mapped `article["text"]` value" rather than "raw text field."

6. **ID assignment tracking mechanism specified.** Step 6 now says to track a
   `has_source_id` boolean flag during step 2 field mapping, so step 6 can determine
   whether to keep the source value or generate a hash without re-inspecting raw keys.

7. **`preprocess_articles` processing order specified.** Six-step sequence documented:
   filter by language → `clean_text()` on all articles → collect cleaned strings →
   `nlp.pipe()` → extract Doc fields → write to article dicts.

8. **`_chunk_text` overlap search window bounded.** Backward search in step 5 is now
   bounded to `(end - chunk_size)` as the minimum position, making it O(overlap).

9. **`_resolve_overlapping_entities` "longer" defined.** Now `(end - start)` is
   authoritative, not `len(entity["text"])`.

10. **Character-based chunking limitation documented in `run_ner`.** Note added that
    400-character chunks typically stay under BERT's 512-token limit for standard news
    text; HuggingFace pipeline truncates if the limit is exceeded.

11. **`build_entity_dataframe` date column type corrected.** Column is typed
    `object (str or float NaN)`. Added note: `pd.isna()` must be used to check absence,
    not `date is None`.

12. **`plot_top_entities` grouping made explicit.** Groups by `entity_text` alone.
    Rationale stated: chart shows name frequency; same surface string with different
    labels is intentionally merged.

13. **`plot_entity_dynamics` year-month format specified.** Uses `dt.to_period("M")`,
    rendered as `"%Y-%m"` strings on the x-axis.

14. **`summarize_article` special token handling specified.** `encode()` called with
    `add_special_tokens=False` to count only content tokens.

15. **`summarize_articles` config key paths made explicit.** Docstring now shows
    `config["summarization"]["min_summary_length"]` and `max_summary_length`.

16. **`score_all_articles` uses `.get()` for both keys.** Skip condition now uses
    `article.get("cleaned_text")` and `article.get("summary")` to avoid KeyError on
    articles that never received those keys.

17. **`explain_similarity_extremes` tie-breaking uses `str()`.** `str(article_id)`
    cast before lexicographic sort, to handle datasets where IDs are integers.

18. **`predict_all_topics` redistribution formula specified.** Remaining slots computed
    as `sample_size - len(selected)`, distributed as `ceil(remaining / n_remainders)`.
    Pool safety: use `rng.sample(pool, len(pool))` rather than raising ValueError on
    undersized pool.

19. **`predict_all_topics` dead branch removed.** Sort is always `str(article["id"])`
    — the fallback to list index is removed because `id` is guaranteed after
    normalisation.

20. **`evaluate_topic_predictions` empty title substitution added.** Uses same
    `[id: ...]` substitution as `build_entity_dataframe` for consistency.

21. **`mock_ner_pipeline` fixture intent clarified.** Docstring now states it produces
    raw HuggingFace pre-rename output (with `entity_group`, not `label`), and that
    tests should assert the renamed `label` key in `run_ner` output.

22. **`pyproject.toml` added to directory structure.**

23. **`.pre-commit-config.yaml` added to directory structure** with full file content
    (black + ruff hooks).

24. **`processed_path` config key documented.** Marked as reserved for future use; no
    current module reads or writes it.

25. **Python 3.10 `match` syntax note softened.** Changed from "required for match
    syntax" to "match syntax may be used in implementation but is not mandated by this
    spec."
