# 0005 — Language-Only Grouping for Article Normalisation

**Date:** 2026-05-18  
**Status:** Accepted  
**Branch:** fix/code_test

---

## Context

The assignment requires analysis "by specific country(-ies)" but emphasises
"experiment with multiple language texts." The raw Wikinews dataset has no
dedicated country column — country is signalled only via the `categories` list
alongside topic labels.

Country-filtered runs (`countries=["United States", "Germany"]`) produced:

| Group | Articles (NER pass) |
|---|---|
| Germany / Politics and conflicts | 8 |
| Germany / Science and technology | 1 |
| Germany / Sports | 4 |
| United States / (all topics) | sufficient |

Germany groups fall below the `articles_per_topic_min: 10` threshold, making
cross-country comparison unreliable. Only the US had sufficient samples across
all three topics.

---

## Decision

Pass `countries=None` to `normalise_articles()` in both the NER pass (Cell 7)
and the summarisation pass (Cell 8). This disables country filtering so articles
are selected from the full English + German pool. Country is retained as
best-effort metadata extracted from the categories list for use in downstream
visualisations.

---

## Consequences

**Positive**
- Both language groups (en, de) produce 20 articles per topic (60 per language,
  120 total for NER), well above the minimum.
- Country metadata is still available for secondary analysis (e.g. NER entity
  plots filtered by country label).
- The pipeline design is not compromised — `countries` filtering remains fully
  functional for future runs.

**Negative / Limitations**
- Country metadata is best-effort only. It is extracted from a fixed
  `_KNOWN_COUNTRIES` vocabulary; articles from regions not in that list receive
  `country=""`. The distribution of country labels within each language group is
  not controlled.
- The `articles_per_topic_max` cap applies per `(language, topic)` cell, not per
  `(country, language, topic)`. Country composition within a sampled group is
  therefore random, not balanced.

---

## Implementation

### `src/data_normalizer.py`

- **Signature:** `countries: list[str]` → `countries: Optional[list[str]]`
- **Validation:** `countries=[]` still raises `ValueError`; `countries=None` is
  accepted and disables filtering.
- **New helper `_extract_country_from_categories(categories)`:** searches the
  raw categories list against `_KNOWN_COUNTRIES` by exact (normalised) match.
  Returns the first match or `None`.
- **New constant `_KNOWN_COUNTRIES`:** `frozenset[str]` of lowercase country
  names. Originally a hand-maintained list of ~50 names; now generated from
  `pycountry` (ISO 3166) plus a small `_COUNTRY_ALIASES` set for colloquial
  forms the ISO dataset omits — see FIX-8. Used only for metadata extraction.
- **Country assignment when `countries=None`:** calls
  `_extract_country_from_categories()`; stores result or `""` if no match.
  Never drops a record on country grounds.
- **Sampling group key:** `(language, topic)` when `countries is None`;
  `(country, language, topic)` when countries filtering is active. This prevents
  country metadata from fragmenting the sampling pool.
- **Text-as-list fix (also in this change):** Wikinews `text` field is a list of
  paragraph strings. Added a join step before the text-length check so the field
  is always a single string after normalisation.

### `notebooks/analysis.ipynb`

- Cell 7 (NER pass): `countries=None`
- Cell 8 (summarisation pass): `countries=None` (was previously still passing
  `config["countries"]["selected"]` — now consistent with Cell 7)

### `config/config.yaml`

- `countries.selected` is unchanged and remains the source of truth for
  explicit country-filtered runs.
- Config comment updated to reflect `countries=None` as the default notebook
  behaviour.

### `tests/test_data_normalizer.py`

Six new tests covering the `countries=None` path:

| Test | What it verifies |
|---|---|
| `test_countries_none_does_not_raise` | `None` is accepted, not treated as empty |
| `test_countries_none_extracts_country_from_categories` | known country extracted as metadata |
| `test_countries_none_country_empty_string_when_no_known_country` | unknown region → `""` |
| `test_countries_none_does_not_drop_by_country` | no `country_not_in_config` drops |
| `test_countries_none_groups_by_language_topic_not_country` | `max_per_topic` applies to merged pool |
| `test_countries_filter_still_groups_by_country_language_topic` | existing behaviour unchanged |

### `docs/SPEC_v3.md`

- Function signature updated.
- Config comment updated.
- Step 3 (country drop rule) amended: drop only when `countries is not None`.
- Step 5 (grouping) amended: key is `(language, topic)` or
  `(country, language, topic)` depending on mode.
- Args docstring updated.

---

## Alternatives Considered

**Keep country filtering, lower the minimum.** Rejected — Germany / Science and
technology has 1 article; no minimum threshold makes that viable.

**Increase dataset scope.** The dataset is fixed; no additional data source was
available.

**Use language as the primary grouping axis with country as a secondary filter
applied only to plots.** This is essentially what the implementation does:
language is the primary sampling axis, country is secondary metadata used only
in visualisation.

---

## Update 2026-05-18 — Sample size raised for NER pass

The "120 total for NER" figure in **Consequences** is superseded. Cell 7 now uses
`articles_per_topic_ner: 100` (600 NER articles total: 300 en + 300 de). See
FIX-4 in `docs/code_implementation_issues.md`. The language-only grouping
decision itself is unchanged.

---

## Update 2026-05-19 — Country dropped from output DataFrames

This ADR originally retained country as best-effort metadata "for use in
downstream visualisations." After NER plots (Cell 13) and similarity plots
(Cell 16) moved to language-only and topic-only axes, no consumer reads the
`country` column from any output DataFrame. The column was inconsistent
(often empty, since `_KNOWN_COUNTRIES` only matches articles whose categories
list a recognised country name) and misleading (an article *about* Germany
might still have `country=""` if its categories don't say so explicitly).

**What changed**

- `build_entity_dataframe`, `investigate_ner_errors`, `build_similarity_dataframe`,
  `explain_similarity_extremes` no longer surface `country` as a column.
- `plot_top_entities`, `plot_entity_dynamics`, `investigate_ner_errors` lost
  their now-unused `country: Optional[str] = None` parameters.

**What is unchanged**

- The article-level `country` field is still populated by
  `_extract_country_from_categories` during normalisation.
- The sampling logic in `normalise_articles` is unchanged.
- The `countries` block in `config/config.yaml` is unchanged (still the
  source of truth for explicit country-filtered runs).

See FIX-6 in `docs/code_implementation_issues.md`.
