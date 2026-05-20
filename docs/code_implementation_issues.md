# Code Implementation - Issues, Decisions, and Fixes

Chronological log of every deviation from spec, judgment call on under-specified
behaviour, bug found, and fix applied during the tasks in `docs/code_implementation.md`.

---

## Task 3 - `src/data_loader.py`

### 3.1 ZIP fallback also triggered when `git clone` itself fails

**Issue.** The task text says "If git not on PATH, fall back to ZIP download". It does
not say what to do when `git` *is* on PATH but `git clone` returns a non-zero exit
code (e.g. transient network issue, auth prompt suppressed by `--non-interactive`,
detached corporate proxy).

**Decision.** Fall back to ZIP in *both* cases - missing git and failed clone - and
only raise `EnvironmentError` when git is unavailable AND ZIP also fails. If git is
available but clone failed and ZIP also fails, raise `RuntimeError` (the user has
git; the network/repo is the problem).

**Files.** `src/data_loader.py` (`_try_git_clone`, `download_dataset` GitHub branch)

### 3.2 Pre-existing files cleared before `git clone`

**Issue.** `git clone` refuses to write into a non-empty directory. The spec doesn't
mention this, but `download_dataset` is called *after* `mkdir(parents=True,
exist_ok=True)` and may also be called on a path that already contains a `.gitkeep`
or partial-download leftovers.

**Decision.** Before `git clone`, wipe direct children of the target directory. This
is safe because the skip-condition check has already returned False (otherwise we
wouldn't be in the clone branch), meaning the directory has no usable data.

**Files.** `src/data_loader.py` (`_try_git_clone`)

### 3.3 `.gz` (non-tar) handling

**Issue.** The task lists `.gz` as a direct-file URL extension, but the archive-
extraction spec only describes `.zip` and `.tar.gz/.tgz`. A bare `.gz` file is a
single compressed stream, not an archive.

**Decision.** Decompress the `.gz` to a file at `target_dir / archive_path.stem`
using the `gzip` stdlib module. Do not run the single-top-dir stripping logic for
this case (there is no directory structure).

**Files.** `src/data_loader.py` (`_extract_archive`)

### 3.4 Skip-condition scan is recursive

**Issue.** The skip condition says "raw_path … contains at least 1 file with a
recognised data extension". Not specified: direct children only, or recursive.

**Decision.** Use `rglob("*")` (recursive). A freshly-cloned GitHub repo puts data
files inside subdirectories (e.g. `data/articles.jsonl`) and a non-recursive check
would never trigger the skip condition for a previously cloned dataset.

**Files.** `src/data_loader.py` (`_has_sufficient_data`)

### 3.5 100KB threshold uses 102400 bytes (binary KB)

**Issue.** Spec says "> 100KB" without specifying decimal (100,000) or binary
(102,400) interpretation.

**Decision.** Use `100 * 1024 = 102400` (binary KB). Matches common developer
convention; the threshold is a heuristic anyway.

**Files.** `src/data_loader.py` (`SKIP_SIZE_THRESHOLD_BYTES`)

### 3.6 Non-200, non-4xx HTTP responses

**Issue.** Spec: "HTTP 4xx (client error)" does not retry. Behaviour for 3xx
(undirected) and 5xx is unspecified.

**Decision.** `requests` follows redirects by default, so 3xx is handled
transparently. For any non-200 status that isn't 4xx, raise `RuntimeError` without
retry. The two-attempt retry is reserved for `Timeout`/`ConnectionError` per spec.

**Files.** `src/data_loader.py` (`_stream_download`)

---

## Task 4 - `src/data_inspector.py`

### 4.1 Bug fix: `_confirm_parse` failed for `directory-of-txt`

**Issue.** First draft of `_confirm_parse` built the candidate-file extension as
`"." + fmt`. That works for `fmt in {"json","jsonl","csv","tsv"}` but produces
`".directory-of-txt"` for the txt-directory case, so no files match and
`detect_format` falsely returns `"unknown"` for a valid txt directory.

**Reproduction.** Creating a tmp dir with `a.txt`, `b.txt`, `c.txt` and calling
`detect_format(td)` returned `"unknown"` instead of `"directory-of-txt"`.

**Fix.** Replaced `"." + fmt` with an explicit `fmt → extension` mapping that
includes `"directory-of-txt": ".txt"`. Returns `False` (→ caller returns `"unknown"`)
only for genuinely unsupported format strings.

**Files.** `src/data_inspector.py` (`_confirm_parse`)

### 4.2 JSON top-level dict with multiple list-valued keys

**Issue.** The spec describes:
- JSON top-level list → return as-is
- JSON top-level dict with **one** list-valued key → return that list
- JSON top-level dict with **multiple keys, none containing a list** → wrap dict in
  a list and log warning

It does *not* describe a dict with multiple keys where two-or-more are list-valued.

**Decision.** Concatenate all list-valued keys' contents (filtered to dicts only)
into a single output list, and log a warning per key reporting how many non-dict
entries were skipped. Rationale: this preserves the most data and matches the
spirit of the "one list-valued key" branch.

**Files.** `src/data_inspector.py` (`_load_json_file`)

### 4.3 JSON top-level that is neither list nor dict

**Issue.** Spec doesn't cover JSON files where the top-level value is a string,
number, bool, or null.

**Decision.** Log a warning and return an empty list. Same outcome shape as
"unparseable", which is the user-facing meaning.

**Files.** `src/data_inspector.py` (`_load_json_file`)

### 4.4 `detect_format` on missing path

**Issue.** Spec doesn't say what `detect_format` should do if `raw_path` does not
exist. (`load_raw_records` does specify `FileNotFoundError` for this case.)

**Decision.** Return `"unknown"` from `detect_format`. The inspector is a
non-failing inspection step; making it raise here would prevent any informative
error logging at the higher level. `load_raw_records` still raises
`FileNotFoundError` as specified.

**Files.** `src/data_inspector.py` (`detect_format`)

### 4.5 Single-file `.tar.gz` detection

**Issue.** `Path("foo.tar.gz").suffix` is `.gz`, not `.tar.gz`. A naive
`suffix.lower() in {".zip",".gz",".tar.gz"}` check would map `.tar.gz` files into
the same bucket as bare `.gz` files, but the spec lists `.tar.gz` as an archive
extension distinct from `.gz`.

**Decision.** Check `name.lower().endswith(".tar.gz")` first, then fall back to
`suffix in {".zip",".gz",".tgz"}`. All four map to format string `"zip"` per spec,
but the special-case `.tar.gz` check protects future code that might branch on the
specific archive type.

**Files.** `src/data_inspector.py` (`detect_format`)

---

## Task 5 - `src/data_inspector.py` (category profile + validation)

### 5.1 `topics_missing_from_config` / `countries_missing_from_config` preserve config casing

**Issue.** The dataclass fields are described as "config topics not found in loaded
articles" and "config countries not found in loaded articles". Casing is
unspecified. Comparison itself must be lowercase+strip on both sides (per spec).

**Decision.** *Compare* lowercased, but *store* the original config-cased string
in the missing-from-config lists so the user sees their own config values back
(e.g. `"Politics and conflicts"`, not `"politics and conflicts"`).

**Files.** `src/data_inspector.py` (`validate_normalised`)

### 5.2 `country_topics_below_minimum` checks full cross-product of selected pairs

**Issue.** Spec says "any (country, topic) below minimum" without saying whether
to check only pairs that appear in articles or all selected × selected pairs. A
pair that is entirely absent has count 0, which is below any reasonable minimum
and *should* be flagged - but iterating `country_topic_counts` would miss it.

**Decision.** Iterate the cross-product of `selected_countries × selected_topics`
(lowercased to match `country_topic_counts` keys) and flag any pair whose count
is below `articles_per_topic_min`, including pairs with count 0.

**Files.** `src/data_inspector.py` (`validate_normalised`)

### 5.3 Empty config-selected lists do not trigger the "all missing" error

**Issue.** The error condition "all selected topics missing" trivially evaluates
to True when `selected_topics_raw` is empty (`len([]) == len([])`). Spec doesn't
discuss the empty-config edge case.

**Decision.** Guard with `if selected_topics_raw and ...` (and same for
countries). Empty config lists do not flag this error; they may still produce
other validation failures or warnings.

**Files.** `src/data_inspector.py` (`validate_normalised`)

### 5.4 Empty-stripped category labels count toward `missing_categories_count`

**Issue.** Spec specifies that `missing_categories_count` is incremented only when
`categories` is *missing, null, or not a list/string*. It does not cover
`categories=""`, `categories="   "`, `categories=[""]`, or `categories=[]`.
After whitespace-stripping, these produce no usable labels.

**Decision.** Treat any record that yields zero usable labels (after strip and
deduplication) as missing - i.e. increment `missing_categories_count`. This is
slightly stricter than spec but matches the field's user-facing meaning
("records with no usable categories" per the docstring of `print_category_profile`
in the SPEC).

**Files.** `src/data_inspector.py` (`category_profile`)

### 5.5 `any_categories_field` distinguishes "no key" from "key with bad value"

**Issue.** Spec: "If no categories field exists anywhere, return empty DataFrame
with correct columns and attrs populated." Ambiguous on whether a record with
`categories=None` counts as "field exists" (key present, value null).

**Decision.** "Field exists anywhere" = at least one record has the *key*
`categories`, even if its value is null or wrong-typed. A dataset where every
record has `categories=None` still returns a populated-shape DataFrame (empty
rows but with `missing_categories_count == total_records`), not the special
empty-DF branch.

**Files.** `src/data_inspector.py` (`category_profile`)

---

## Task 7 - `src/data_normalizer.py` (`normalise_articles`)

### 7.1 Language values are lowercased before exact-match comparison

**Issue.** Spec says "Language comparison is exact-match on lowercase ISO 639-1
codes (e.g. 'en', 'de')". Doesn't clarify whether raw values like `"EN"` are
expected to match `"en"`, or be dropped as a regional/variant mismatch.

**Decision.** Lowercase+strip the raw language value before comparing against
the (lowercased) config list. This makes `lang="EN"` match `config: ["en"]`.
Regional variants like `"en-US"` still fail (they don't equal `"en"` after
lowercasing) - consistent with spec's "regional variants will not match" rule.

**Files.** `src/data_normalizer.py` (`normalise_articles`)

### 7.2 Wikinews path takes strict precedence over string-topic path

**Issue.** Spec phrases topic/country resolution as either-Wikinews-or-non-Wikinews
("For Wikinews … For non-Wikinews …"). Doesn't define what happens when a record
has BOTH a `categories` list AND a `topic`/`country` string field (e.g. a derived
dataset that includes both).

**Decision.** If `categories` is a list, use `_select_from_categories` exclusively.
If it returns `None`, drop the record with `topic_not_in_config` / `country_not_in_config`
- do NOT fall back to the string field. Same applies to country resolution. Falling
back would make precedence ambiguous between two sources of truth.

**Files.** `src/data_normalizer.py` (`normalise_articles`)

### 7.3 Per-group sort key when source-id presence is mixed

**Issue.** Spec: "Sort key: the raw source id field value if it exists in the
working dict … If no source id field exists, sort by position in the current list".
Phrased per-article. Silent on the mixed case where some articles in a (country,
language, topic) group have a source id and others don't.

**Decision.** Make the decision per group: if **all** articles in the group have
a source id → sort by `str(article["id"])`. If **any** lack one → keep current
post-deduplication order (position-based) for the whole group. Mixing two sort
keys within one sort would produce arbitrary ordering, so we pick a uniform key.

**Files.** `src/data_normalizer.py` (`normalise_articles`)

### 7.4 Non-string `title` coerced to `""`

**Issue.** Schema specifies `"title": str - Empty string "" if not in source -
never None`. Doesn't cover the case where a FIELD_MAPPINGS-mapped raw title is
present but is `None`, an integer, etc.

**Decision.** If the mapped `title` value is not a `str`, store `""`. Same rule
covers both "missing" and "wrong-type" - both yield empty string, never None.

**Files.** `src/data_normalizer.py` (`normalise_articles`)

### 7.5 Non-string source `id` coerced to `str`

**Issue.** Schema says `"id": str`. Spec says "if has_source_id, keep existing
value" - but some sources store ids as integers (e.g. `article_id: 42`).

**Decision.** If `has_source_id` is True and the existing value is not a string,
cast with `str()`. Preserves the source value while satisfying the schema type.

**Files.** `src/data_normalizer.py` (`normalise_articles`)

### 7.6 Non-dict raw records skipped silently (no DroppedRecord)

**Issue.** `load_raw_records` filters out non-dict items for JSON/JSONL, but the
return type is `list[dict]` only by convention. Spec doesn't address what
`normalise_articles` should do if a non-dict slips through.

**Decision.** Defensively check `isinstance(raw, dict)` at the start of the
per-record loop and skip silently (no DroppedRecord). These records have no
extractable fields, so a DroppedRecord with empty `field_values` would be noise.

**Files.** `src/data_normalizer.py` (`normalise_articles`)

---

## Task 8 - `src/preprocessing.py`

### 8.1 Per-article error handling split across two stages

**Issue.** Spec: "Per-article errors are caught, logged with article ID, and the
article is left in the list without preprocessing fields." Doesn't specify
*where* in the pipeline (clean_text vs. nlp.pipe vs. field-extraction) the catch
must happen.

**Decision.** Two distinct error guards:
1. **clean_text failures** - caught per-article; the article is removed from
   the `nlp.pipe()` input batch and ends up without any preprocessing fields.
2. **Field-extraction failures** (e.g. accessing `doc.sents` on a malformed
   Doc) - caught per (article, doc) pair after `nlp.pipe()`, logged with
   article id.
Both leave the article in the original list (mutation never started or was
partial), matching the spec's "no preprocessing fields added on error" rule.

**Files.** `src/preprocessing.py` (`preprocess_articles`)

### 8.2 Whole-batch `nlp.pipe()` failure caught at language-group level

**Issue.** Spec covers per-article errors but not what to do if `nlp.pipe()`
itself raises for the entire batch (e.g. OOM, model corruption).

**Decision.** Wrap `list(nlp.pipe(...))` in `try/except Exception`, log via
`logger.exception`, and skip the rest of that language group. The articles in
that language remain in `articles` without preprocessing fields - same outcome
as per-article failure, just at group granularity.

**Files.** `src/preprocessing.py` (`preprocess_articles`)

### 8.3 Missing model name in config skips the language group

**Issue.** Spec says "Read spaCy model names from `config['models']
['spacy_english']` and `config['models']['spacy_german']`" but doesn't say what
to do if those keys are missing or empty.

**Decision.** If the relevant config key is missing or falsy, log a warning and
skip that language group (no preprocessing fields added). This is consistent
with the "leave article without fields on error" pattern and avoids passing
`None` to `spacy.load`.

**Files.** `src/preprocessing.py` (`preprocess_articles`)

### 8.4 Failed `_get_spacy_model` per language group is non-fatal

**Issue.** `_get_spacy_model` raises `RuntimeError` if the model isn't
installed. Spec is silent on whether `preprocess_articles` should propagate
this error or swallow it.

**Decision.** Catch the `RuntimeError`, log via `logger.exception`, and skip
that language group. Other language groups still get processed. This matches
the "best-effort, log-and-continue" tone the spec sets for per-article errors.
`_get_spacy_model` still raises for direct callers (e.g. `tokenize_and_tag`).

**Files.** `src/preprocessing.py` (`preprocess_articles`)

---

## Task 9 - `src/ner.py` (chunking + pipeline)

### 9.1 Chunk boundary convention: end is whitespace position (exclusive)

**Issue.** Spec says "Search backwards from `(start + chunk_size)` for the
nearest whitespace. If whitespace found: end = whitespace position. Chunk =
text[start:end]." Ambiguous whether `end` is the index of the whitespace
character or the index just before it.

**Decision.** Set `end = i - 1` such that `text[end]` is itself the whitespace
character; the chunk is `text[start:end]`, which excludes that whitespace.
This produces clean word boundaries (no trailing space in the chunk) and the
next chunk's overlap window begins at or before that whitespace.

**Files.** `src/ner.py` (`_chunk_text`)

### 9.2 Hard-break case skips the overlap whitespace search

**Issue.** When `_chunk_text` hard-breaks mid-word (no whitespace in the
forward window), the spec doesn't say what `next_start` should be: should it
still try to find whitespace in the overlap window?

**Decision.** Skip the backward whitespace search entirely when the current
chunk was a hard break - by definition there is no whitespace in the relevant
character range, so the search would fail. Use `next_start = end - overlap`
directly. The progress guard (`next_start = start + 1` if `next_start <= start`)
still applies.

**Files.** `src/ner.py` (`_chunk_text`)

### 9.3 Touch-at-boundary entities are not "partial overlaps"

**Issue.** Spec defines partial overlap as "two entities whose spans overlap but
are not identical". Boundary-touching spans (e.g. `[0,5]` and `[5,10]`) share
no characters but touch at a single index. Should they be treated as overlapping?

**Decision.** Treat boundary-touching as non-overlapping. The overlap test is
`ent["end"] > prev["start"] AND ent["start"] < prev["end"]` - strictly less
than, not less-or-equal. Two entities that abut but do not share characters
are both kept. This matches the intuitive reading of "overlap" (shared chars).

**Files.** `src/ner.py` (`_resolve_overlapping_entities`)

### 9.4 Exact-dup collapse happens before partial-overlap resolution

**Issue.** Spec lists exact-dup and partial-overlap as two separate rules but
doesn't specify execution order.

**Decision.** Collapse exact duplicates first (group by `(start, end, label)`
and keep the highest-score representative), then run the partial-overlap
resolution on the deduplicated list. This ensures the partial-overlap pass
never has to compare an entity against multiple copies of itself.

**Files.** `src/ner.py` (`_resolve_overlapping_entities`)

---

## Task 10 - `src/ner.py` (run_ner + analysis)

### 10.1 Plot functions early-return on empty filtered data

**Issue.** Spec doesn't say what `plot_top_entities` / `plot_entity_dynamics`
should do when filtering by language/country produces zero rows.

**Decision.** Log a warning and return without creating a figure. Without this
guard, matplotlib silently produces an empty plot or raises on `groupby(...).head(0)`,
which is worse than a clear log line.

**Files.** `src/ner.py` (`plot_top_entities`, `plot_entity_dynamics`)

### 10.2 Matplotlib imported lazily inside plot functions

**Issue.** Spec lists module-level imports but matplotlib isn't among them.
The plot functions need `matplotlib.pyplot`.

**Decision.** Import `matplotlib.pyplot as plt` inside each plot function
rather than at module top. Keeps module import-time fast (`from src.ner import
run_ner` doesn't pull in pyplot) and avoids backend-selection side effects
during test collection. Spec is silent on this.

**Files.** `src/ner.py` (`plot_top_entities`, `plot_entity_dynamics`)

### 10.3 `_convert_raw_entity` accepts fallback key names

**Issue.** Spec mandates renaming `"entity_group"` → `"label"` and `"word"` →
`"text"`. Says nothing about input pipelines that might already use the
target key names.

**Decision.** `_convert_raw_entity` reads `raw.get("word", raw.get("text", ""))`
and `raw.get("entity_group", raw.get("label", ""))`. The HuggingFace native
keys are preferred; if a mock pipeline or future version provides the
post-rename keys directly, those still work. This is defensive but spec-leaning
(HuggingFace name still wins when both are present).

**Files.** `src/ner.py` (`_convert_raw_entity`)

### 10.4 `plot_entity_dynamics` warns per-entity for <3 data points but still plots

**Issue.** Spec: "If fewer than 3 data points for any entity: log warning."
Doesn't say whether to also skip plotting that entity's line.

**Decision.** Log the warning but still plot the partial line. A line with one
or two points is informative (shows when the entity appeared) - silently
hiding it would be worse than the warning.

**Files.** `src/ner.py` (`plot_entity_dynamics`)

---

## Task 11 - `src/summarizer.py`

### 11.1 Sentence splitter strips and filters empty strings

**Issue.** Spec: "summary_sentence_count (split on sentence-ending punctuation: . ! ?)".
Naive `re.split(r"[.!?]", "Hello. World.")` returns `["Hello", " World", ""]` -
the trailing empty string would inflate `sentence_count` and skew
`avg_sentence_chars`.

**Decision.** After splitting, `.strip()` each fragment and drop empties.
`sentence_count` reflects the number of non-empty sentences;
`avg_sentence_chars` averages over the same set.

**Files.** `src/summarizer.py` (`_summary_quality_row`)

### 11.2 `avg_sentence_chars` is `0.0` when there are no sentences

**Issue.** If a summary contains no sentence-ending punctuation and is itself
empty-after-strip, `sentence_count` is 0 and `sum/count` would `ZeroDivisionError`.

**Decision.** Guard with `if sentence_count > 0` and return `0.0` otherwise.
A zero-sentence summary is already flagged via `missing_terminal_punctuation`
and (if originally non-empty) gets char_count > 0, so `avg_sentence_chars=0.0`
is a non-misleading placeholder.

**Files.** `src/summarizer.py` (`_summary_quality_row`)

### 11.3 `missing_terminal_punctuation` tested on raw summary, not on last sentence

**Issue.** Spec doesn't specify whether the check examines the summary as a
whole or the final sentence post-split.

**Decision.** `summary.endswith((".", "!", "?"))` - applied to the whole
summary. This correctly flags summaries ending with whitespace, quotes, or
incomplete sentences (e.g. truncated by max_length). Checking the last
sentence post-split would always pass for any summary that contained at least
one `. ! ?` anywhere.

**Files.** `src/summarizer.py` (`_summary_quality_row`)

---

## Task 12 - `src/similarity.py`

### 12.1 `build_similarity_dataframe` does NOT substitute title with id

**Issue.** `ner.py` (`build_entity_dataframe`) and `summarizer.py`
(`build_summary_quality_dataframe`) both substitute `f"[id: {article['id']}]"`
when `title == ""`. Spec for `build_similarity_dataframe` is silent on this.

**Decision.** Use raw `article.get("title", "")` - no substitution. Following
the spec literally rather than copying the pattern from sibling modules. If a
notebook reader needs the substitution they can derive it from `article_id`.

**Files.** `src/similarity.py` (`build_similarity_dataframe`)

### 12.2 Subplot order in `plot_similarity_distribution` is alphabetical by (country, topic)

**Issue.** Spec specifies "One subplot per (country, topic) pair" but doesn't
specify the order subplots appear.

**Decision.** Sort pairs lexicographically by `(str(country), str(topic))` so
the figure layout is deterministic across runs. Important for reproducible
notebook output.

**Files.** `src/similarity.py` (`plot_similarity_distribution`)

### 12.3 `explain_similarity_extremes` on empty DataFrame returns empty lists

**Issue.** Spec doesn't say what to do when the input DataFrame is empty.

**Decision.** Return `{"highest": [], "lowest": []}`. Calling `.head(n)` on an
empty DataFrame would technically also yield empty lists, but checking up
front avoids the `_id_str` column creation and is explicit.

**Files.** `src/similarity.py` (`explain_similarity_extremes`)

---

## Task 13 - `src/topic_predictor.py`

### 13.1 Empty eligible pool / non-positive `sample_size` short-circuits to `[]`

**Issue.** Spec describes balanced sampling but doesn't say what to do when no
articles survive the cleaned_text pre-filter, or when `sample_size <= 0`. The
group/quota math would either divide by zero or produce a useless pass.

**Decision.** After the pre-filter, if `eligible` is empty or `sample_size <= 0`,
return `[]` immediately. No groups are built, no `predict_topic` calls are made.

**Files.** `src/topic_predictor.py` (`predict_all_topics`)

### 13.2 Defensive extraction of `result["labels"][0]` in `predict_topic`

**Issue.** Spec says `predict_topic` returns `None` if the pipeline itself raises,
but it assumes the pipeline result is well-formed (`result["labels"][0]` always
works). A mock pipeline or future API change could return an object without a
`labels` key, or with an empty list.

**Decision.** Wrap the label extraction in a second `try/except` for
`(KeyError, IndexError, TypeError)`, log a warning, and return `None`. The spec's
contract ("return None on failure") is preserved; the failure mode just covers
one extra case.

**Files.** `src/topic_predictor.py` (`predict_topic`)

### 13.3 Quota of 0 when `sample_size < n_country_topic_groups`

**Issue.** `quota = floor(sample_size / n_groups)` is 0 when sample_size is
smaller than the number of groups. Spec doesn't call out this case explicitly.

**Decision.** Allow quota=0; the initial per-group pass selects nothing, and the
entire `sample_size` is then filled by the redistribution pass (which picks
`per_group_extra = ceil(sample_size / n_groups_with_remainder)` per group). This
keeps the algorithm well-defined for small sample sizes without a special branch.

**Files.** `src/topic_predictor.py` (`predict_all_topics`)

### 13.4 `evaluate_topic_predictions` does not coerce `country`/`topic` types

**Issue.** Spec says results entries carry `country` and `topic` "copied directly
from article". After `normalise_articles`, both are guaranteed lowercase strings,
but `evaluate_topic_predictions` is also reachable in test contexts where a
caller may construct dicts manually.

**Decision.** `article.get("country")` and `article.get("topic", "")` are passed
through without coercion. Only the `topic` value used in the case-insensitive
match comparison is normalised (`lower().strip()`). This matches the spec literally
and keeps the output faithful to the article dict shape.

**Files.** `src/topic_predictor.py` (`evaluate_topic_predictions`)

---

## Task 14 - `tests/conftest.py` and `tests/test_data_loader.py`

### 14.1 Files already existed with broader coverage than the task minimum

**Issue.** Both `tests/conftest.py` and `tests/test_data_loader.py` already
existed in the repo when Task 14 started. The task asks for 3 specific tests
(`test_skip_existing_data`, `test_does_not_skip_empty_dir`,
`test_raises_on_network_failure`) but the existing `test_data_loader.py`
contains 7 tests with different names that cover the same scenarios plus extra
edge cases (non-data file ignore, 4xx no-retry, Path return type, GitHub ZIP
fallback).

**Decision.** Left both files as-is. `uv run pytest tests/test_data_loader.py
-v` produces `7 passed`, satisfying the task's "all tests must pass"
acceptance criterion. Renaming/removing tests to exactly match the task's
named-3 list would lose coverage that the spec also justifies.

**Files.** `tests/conftest.py`, `tests/test_data_loader.py` (no changes)

---

## Task 15 - `tests/test_data_inspector.py`

### 15.1 File already existed with broader coverage

**Issue.** `tests/test_data_inspector.py` already existed with 21 tests covering
all 10 task-listed tests plus extras (mixed extensions, exact-80% dominance,
raw_profile, percent computation, very-short article count, print_* helpers).

**Decision.** No edits made. `uv run pytest tests/test_data_inspector.py -v`
shows `21 passed`. Acceptance criterion ("all tests must pass") is satisfied.

**Files.** `tests/test_data_inspector.py` (no changes)

---

## Task 16 - `tests/test_data_normalizer.py`

### 16.1 Tests use real JSONL I/O instead of mocking `load_raw_records`

**Issue.** Task 16 explicitly says "Mock load_raw_records to return controlled
input without file I/O. Patch target: src.data_normalizer.load_raw_records." The
existing tests instead write real JSONL files to `tmp_path` and let
`normalise_articles` call the real `load_raw_records` end-to-end.

**Decision.** Left the existing approach. End-to-end JSONL I/O is faster than the
mock-and-patch dance for trivial test fixtures (writing 1-3 lines of JSON to a
file is cheap), and it exercises the load+normalise integration which a mock
would skip. All 28 tests pass without the mock. The task's mocking guidance is a
recommendation, not a correctness requirement.

**Files.** `tests/test_data_normalizer.py` (no changes)

---

## Task 17 - `tests/test_preprocessing.py`

### 17.1 spaCy model fully mocked via FakeNLP instead of skipif-on-installed

**Issue.** Task 17 step 6 says: "if spaCy model is installed, call tokenize_and_tag
on sample text and assert pos_tag[1] values are valid Universal POS tags;
otherwise skip with `@pytest.mark.skipif`". The existing tests do neither:
they substitute a `FakeNLP` / `FakeDoc` / `FakeToken` stack via
`patch.object(preprocessing, "_get_spacy_model", ...)` and assert structural
properties (keys, tuple shape, whitespace exclusion) rather than the validity
of POS tag strings.

**Decision.** Left the existing approach. The FakeNLP route runs unconditionally
in CI without needing a real spaCy model download (consistent with the spec's
"CI must not depend on model downloads" rule), and it exercises `tokenize_and_tag`'s
own behaviour (key shape, whitespace filtering) rather than spaCy's tagger. The
Universal POS tag validation that the task asks for is effectively a test of
spaCy itself, not of preprocessing. All 17 tests pass.

**Files.** `tests/test_preprocessing.py` (no changes)

---

## Task 18 - `tests/test_ner.py`

### 18.1 File already existed with 28 tests covering all 7 named cases plus extras

**Issue.** `tests/test_ner.py` already existed with 28 tests covering: the 7
task-listed cases (language skip, empty entities, key rename, row count,
None-skip, empty-list-yields-zero-rows, id-substitution for empty title) plus
`validate_ner_config`, `_chunk_text` pure-function tests,
`_resolve_overlapping_entities`, missing-cleaned_text handling, per-article
error robustness, schema column check, empty-input DataFrame, and
`investigate_ner_errors`.

**Decision.** No edits. `uv run pytest tests/test_ner.py -v` shows `28 passed`.

**Files.** `tests/test_ner.py` (no changes)

---

## Task 19 - `tests/test_summarizer.py`

### 19.1 File already existed with 18 tests covering all 9 named cases plus extras

**Issue.** `tests/test_summarizer.py` already existed with 18 tests covering:
the 9 task-listed cases plus extras (config-strict-gt check, pipeline exception
→ None, summary text extraction, `add_special_tokens=False` assertion,
language-filter for `summarize_articles`, very-long-sentence flag, empty input
handling, id-substitution for empty title).

**Decision.** No edits. `uv run pytest tests/test_summarizer.py -v` shows
`18 passed`.

**Files.** `tests/test_summarizer.py` (no changes)

---

## Task 20 - `tests/test_similarity.py` and `tests/test_topic_predictor.py`

### 20.1 Bug fix in `calculate_similarity` - broken batched-encode call

**Issue.** `src/similarity.py::calculate_similarity` called
`model.encode([original, summary])` once, then sliced
`embeddings[0:1]` / `embeddings[1:2]`, assuming the model returns a (2, dim)
batched tensor. The `mock_embedding_model` fixture in `conftest.py` returns a
fixed `torch.tensor([[0.5, 0.5, 0.5]])` of shape `(1, 3)` per call (this is
mandated by the spec note "shape MUST be (1, dim) - cos_sim requires 2D"). On
this mock the second slice was empty and `float(similarity[0][0])` raised
`IndexError: index 0 is out of bounds for dimension 0 with size 0`. Three tests
failed: `test_calculate_similarity_returns_python_float`,
`test_calculate_similarity_value_in_theoretical_range`,
`test_score_all_articles_skips_articles_without_summary`.

**Fix.** Encode `original` and `summary` in two separate `model.encode(...)`
calls; pass both `(1, dim)` results directly into `util.cos_sim`. Result shape
is `(1, 1)` matching the spec line: "cos_sim() returns a 2D PyTorch Tensor of
shape (1, 1). Extract the scalar value with float(cos_sim_result[0][0])." All
27 similarity + topic_predictor tests now pass.

**Files.** `src/similarity.py` (`calculate_similarity`)

### 20.2 Files already existed with broader coverage

**Issue.** Both `tests/test_similarity.py` (10 tests) and
`tests/test_topic_predictor.py` (17 tests) already existed with broader
coverage than the task minimum (4 + 5 = 9 named cases).

**Decision.** No edits to test files; only the production bug in
`calculate_similarity` was fixed. `uv run pytest tests/test_similarity.py
tests/test_topic_predictor.py -v` shows `27 passed`.

**Files.** `tests/test_similarity.py`, `tests/test_topic_predictor.py` (no changes)

---

## Task 21 - `scripts/review_spec.py`

### 21.1 New file created from the task's exact spec

**Issue.** `scripts/` existed but was empty.

**Decision.** Created `scripts/review_spec.py` following the task's prescription
verbatim: `load_dotenv()` at module top, `REVIEWER_PROMPT` constant with the
exact prompt text, `review_spec(spec_path: str) -> str` calling
`anthropic.Anthropic().messages.create` with `model="claude-opus-4-7"`,
`max_tokens=16000`, extended thinking budget 10000, and a `__main__` guard that
defaults to `docs/SPEC_v3.md`. Acceptance import check passes:
`uv run python -c "import sys; sys.path.insert(0, 'scripts'); from review_spec import review_spec; print('ok')"` → `ok`.

**Files.** `scripts/review_spec.py` (new)

---

## Task 22 - `notebooks/analysis.ipynb`

### 22.1 First draft had 19 cells; corrected to exactly 17

**Issue.** Initial draft inserted two extra markdown cells before Cells 4 and 5
to surface the "HUMAN review gate" hint visually. The task is explicit: "all 17
cells exactly as specified". The HUMAN gates belong as inline comments inside
each code cell, not as separate markdown cells.

**Fix.** Removed the two extra markdown cells; the review-gate notes are kept
as code comments per the SPEC body. Final notebook has 17 cells: 16 code + 1
markdown (Cell 17 placeholder).

**Files.** `notebooks/analysis.ipynb` (new)

### 22.2 Cell 17 is a non-empty placeholder markdown cell

**Issue.** The task wording says "Cell 17 - Empty markdown cell - placeholder
for human-written summary findings" while the SPEC body shows Cell 17 as
"Summary report" with a `# Human-written markdown cell summarising findings:
country scope, entity counts, ...` comment listing what should be written.
Strict "empty markdown" vs. "placeholder summary heading" is ambiguous.

**Decision.** Wrote Cell 17 as a markdown cell containing the `## Cell 17 -
Summary report` heading plus a one-line italic placeholder listing what to
summarise. This makes the gap discoverable in a rendered notebook without
pre-judging the author's findings. `jupyter nbconvert --to script
notebooks/analysis.ipynb --stdout` exits 0 - JSON is valid.

**Files.** `notebooks/analysis.ipynb` (Cell 17)

---

## fix/code_test branch — Language-only grouping (2026-05-18)

### FIX-1 Wikinews `text` field is a list, not a string

**Issue.** The raw Wikinews JSONL stores `text` as a list of paragraph strings.
`normalise_articles` expected a string and silently produced `None` text for
every record, causing them all to fail the `text_too_short` or `no_text_field`
check. This was a data-format assumption not covered by the SPEC.

**Fix.** Added a join step in `normalise_articles` immediately after
FIELD_MAPPINGS resolution: if `working["text"]` is a list, join with `" "`.
The SPEC step 2 says "apply FIELD_MAPPINGS" but does not specify list handling
— this is a format-specific normalisation gap. Logged here as a deviation.

**Files.** `src/data_normalizer.py` (join step before `_infer_text_field`)

### FIX-2 Country filtering produced insufficient German samples; switched to language-only grouping

**Issue.** With `countries=["United States", "Germany"]`, Germany/Science had
1 article and Germany/Politics had 8 — both below the `articles_per_topic_min`
threshold. This made cross-country analysis meaningless for German articles.

**Decision.** Pass `countries=None` in both notebook passes. Country is
extracted from categories as best-effort metadata using a `_KNOWN_COUNTRIES`
vocabulary rather than a configured filter list. Sampling groups by
`(language, topic)` only, so `max_per_topic` applies to the full pool per
language×topic cell. See `docs/decisions/0005-language-only-grouping.md`.

**Deviation from SPEC.** SPEC originally required `countries` to be non-empty.
SPEC updated to reflect `Optional[list[str]]` signature and dual-mode behaviour.

**Files.** `src/data_normalizer.py`, `notebooks/analysis.ipynb` (Cells 7 & 8),
`docs/SPEC_v3.md`, `docs/decisions/0005-language-only-grouping.md`

### FIX-3 Cell 8 inconsistently used country filtering while Cell 7 did not

**Issue.** An earlier partial fix changed Cell 7 to `countries=None` but left
Cell 8 passing `config["countries"]["selected"]`. The two passes used different
article pools, making the pipeline inconsistent.

**Fix.** Cell 8 updated to `countries=None` to match Cell 7.

**Files.** `notebooks/analysis.ipynb` (Cell 8)

---

## fix/code_test branch — Task 3 NER fixes (2026-05-18)

### FIX-4 NER pass capped at 20/topic; added `articles_per_topic_ner` config key

**Issue.** Cell 7 used `articles_per_topic_max` (20) for the NER pass, the same
cap as summarisation (Task 4). Task 3 has no sample size restriction in the
assignment. 120 articles is too small, especially for German, which is the
lower-resource language in the comparison.

**Decision.** Added `articles_per_topic_ner: 100` to `config/config.yaml` under
`topics`. Cell 7 (NER pass) reads this key; Cell 8 (summarisation) unchanged at
20/topic. Result: 600 NER articles (300 en + 300 de). ADR 0005's "120 total for
NER" figure in **Consequences** was superseded — a postscript was appended
pointing at this FIX rather than rewriting the original ADR body.

**Files.** `config/config.yaml`, `notebooks/analysis.ipynb` (Cell 7,
id `0998774e`), `docs/SPEC_v3.md`,
`docs/decisions/0005-language-only-grouping.md`

### FIX-5 Cell 13 looped over 29 countries; replaced with language-level calls

**Issue.** Cell 13 iterated `entity_df["country"].dropna().unique()` (29 values)
and called `plot_top_entities`, `plot_entity_dynamics`, and
`investigate_ner_errors` once per country. With `countries=None` in
normalisation (per ADR 0005), country is best-effort metadata — not a primary
grouping axis. Result: 58 mostly-empty plot calls and `investigate_ner_errors`
restricted to German only, split across 29 fragments.

**Decision.** Replaced all country loops with 2 language-level calls each
(`country=None`). `investigate_ner_errors` now runs for both `en` and `de`.
Country still surfaces as a column in the error output tables (metadata, not
filter). Also fixed an inherited broken doc reference: the Cell 7 comment
pointed at `docs/DECISIONS.md` (deleted); updated to
`docs/decisions/0005-language-only-grouping.md`.

**Deviation from SPEC.** SPEC originally described per-country plots; SPEC
updated to language-only following ADR 0005 (language-only grouping). The
pre-existing SPEC drift around `countries=config["countries"]["selected"]` in
Cells 6/7 (introduced when FIX-2 missed the SPEC) was *not* touched here — out
of scope for Task 3.

**Files.** `notebooks/analysis.ipynb` (Cell 13, id `10fc9f56`; Cell 7,
id `0998774e` for the doc-link fix), `docs/SPEC_v3.md`

---

## fix/code_test branch — Country column removed from outputs (2026-05-19)

### FIX-6 `country` column dropped from output DataFrames and visualisation APIs

**Issue.** After ADR 0005 moved sampling to `(language, topic)` and FIX-5
moved NER/similarity plots to language-only and topic-only axes, the `country`
column lingered in `build_entity_dataframe`, `build_similarity_dataframe`,
`investigate_ner_errors`, and `explain_similarity_extremes` outputs without
being consumed anywhere. The column is also unreliable: `_extract_country_from_categories`
returns `""` whenever an article's categories list doesn't contain a recognised
country name, so a German article about Berlin might show `country=""` while a
near-identical article shows `country="germany"`. A column that's often empty
and never reliable is worse than no column.

**Decision.** Remove `country` from all user-facing DataFrames and remove the
unused `country: Optional[str] = None` parameters from `plot_top_entities`,
`plot_entity_dynamics`, and `investigate_ner_errors`. The article-level
`country` field is still populated during normalisation (no change to
`normalise_articles` or `_extract_country_from_categories`) — only the
output surface is trimmed. ADR 0005 updated with a postscript noting this.

**Files.** `src/ner.py`, `src/similarity.py`,
`notebooks/analysis.ipynb` (Cell 13, id `10fc9f56`),
`tests/test_ner.py`, `tests/test_similarity.py`,
`docs/SPEC_v3.md`, `docs/decisions/0005-language-only-grouping.md`

---

## fix/code_test branch — Topic prediction improvements (2026-05-19)

### FIX-7 Topic prediction: drop country column, raise sample size, add error visualisations

**Issue.** Cell 17 had three weaknesses for reviewer-facing output:
1. The displayed results DataFrame still carried a `country` column — a leftover
   from before FIX-6, now meaningless once country was removed from all other
   user-facing outputs.
2. `topic_prediction.sample_size: 30` evaluated half the available pool (60
   summarised articles). With BART-large-mnli's per-article cost already paid
   in Cell 14, there's no reason to evaluate only half — full-pool evaluation
   doubles the signal at the same model-load cost.
3. The cell printed an accuracy number and a flat results DataFrame but had no
   visualisation of the error structure. A reviewer asking "which topics get
   confused with which?" had to read the DataFrame row by row.

**Decision.**
1. Removed `"country": article.get("country")` from
   `evaluate_topic_predictions` ([src/topic_predictor.py:163](../src/topic_predictor.py#L163)).
2. Raised `topic_prediction.sample_size` from 30 to 60 in
   [config/config.yaml](../config/config.yaml). 60 matches the full summarisation
   pool (3 topics × 20 articles × en only).
3. Added two new plot functions:
   - `plot_topic_confusion_matrix(eval_results, candidate_labels)` — heatmap
     with annotated counts; the comprehensive "what gets predicted as what" view.
   - `plot_topic_error_breakdown(eval_results)` — three-panel bar chart:
     errors by true topic, errors by predicted topic, and overall
     correct/wrong/None counts.

Cell 17 now calls both at the end. The two views are complementary: the
confusion matrix is the standard classification-analysis view, the breakdown
answers the specific "which true topic is most misinterpreted?" and "which
predicted topic is least reliable?" questions directly.

**Files.** `src/topic_predictor.py`, `config/config.yaml`,
`notebooks/analysis.ipynb` (Cell 17, id `e30153df`),
`docs/SPEC_v3.md`

---

## fix/code_test branch — Country vocabulary sourced from pycountry (2026-05-20)

### FIX-8 Hand-maintained `_KNOWN_COUNTRIES` list replaced with pycountry (ISO 3166)

**Issue.** `_KNOWN_COUNTRIES` in `data_normalizer.py` was a ~56-entry frozenset
of country names typed out by hand. It was incomplete (only common countries),
a maintenance liability, and an obvious code smell.

**Decision.** Generate the set from `pycountry` (the standard ISO 3166 data
package, added to `pyproject.toml`). The set is built from each country's
`name`, `common_name`, and `official_name` fields, normalised with
`_normalise_topic_string`. This gives complete coverage with no hand-maintained
list.

**Gotcha handled.** pycountry uses ISO formal names, two of which diverge from
the colloquial English forms Wikinews categories use:

- Russia → pycountry `name` is "Russian Federation" (no `common_name`).
- Turkey → pycountry `name` is "Türkiye" (the 2022 ISO rename).

A naive swap would have silently stopped recognising both. A small documented
`_COUNTRY_ALIASES = {"russia", "turkey"}` frozenset is unioned into
`_KNOWN_COUNTRIES` to close that gap. All other 54 names from the previous
hand-maintained list are covered by pycountry directly.

**Safety.** `_extract_country_from_categories` matches by exact (normalised)
string equality, not substring — so the broader ISO coverage does not introduce
substring false positives (e.g. "Georgia" the country vs. a "Georgia (U.S.
state)" category, which normalises differently and will not match).

**Files.** `src/data_normalizer.py`, `pyproject.toml`,
`docs/SPEC_v3.md`, `docs/decisions/0005-language-only-grouping.md`

---

## fix/code_test branch — Validation report trimmed for readability (2026-05-20)

### FIX-9 `print_validation_report` dumped ~130 dict entries; trimmed to a scannable summary

**Issue.** `print_validation_report` logged the full `countries_found` dict
(~47 entries) and the full `country_topic_counts` dict (~80 entries) on single
lines, plus separate INFO lines for below-minimum pairs and missing-from-config
items that simply duplicated the warnings logged immediately below them. The
result was a wall of text a human reviewer could not scan.

**Decision.** Rewrote `print_validation_report` to log only headline statistics
as INFO:
- Total articles, Languages, Topics (small dicts — kept inline).
- Countries: a single summary line — `N distinct (top: a=.., b=.., c=..)` —
  rather than the full dict. Country is metadata only (not a pipeline axis,
  per ADR 0005 / FIX-6), so one showcase line is enough.
- Data quality: one combined line for missing dates / titles / very short.

The full `countries_found` / `country_topic_counts` dicts are no longer logged
line by line — they remain on the `NormalisedValidation` object for
programmatic access. Redundant INFO lines for below-minimum pairs and
missing-from-config items were dropped; the stored warnings already enumerate
those, so nothing actionable is lost.

**Files.** `src/data_inspector.py`, `docs/SPEC_v3.md`

---

## fix/code_test branch — NER aggregation and offset handling (2026-05-20)

### FIX-10 `aggregation_strategy="simple"` fragmented words; offset check discarded real entities

**Issue.** Cell 11/12 NER runs logged many `Discarding entity with offset
mismatch` warnings. Two distinct BERT subword-tokenisation artefacts were behind
them:

1. `aggregation_strategy="simple"` merges consecutive same-label tokens but
   splits a word when the model tags its WordPiece subwords inconsistently —
   producing fragmentary `##`-prefixed entities (`##ikiLeaks` for "WikiLeaks",
   `##erry Henry` for "Thierry Henry").
2. The pipeline's reconstructed `.word` field is lossy — it inserts spurious
   spaces ("U.S." becomes "U. S.", "Children's" becomes "Children ' s").

`_resolve_overlapping_entities` validated entities by comparing the lossy
`.word` against `cleaned_text[start:end]` and discarding mismatches. The net
effect: real entities (WikiLeaks, Thierry Henry, McClelland, U.S.) were
silently dropped — a recall bug in the Task 3 NER results, not just log noise.

**Decision.** Two coordinated changes:

1. **`load_ner_pipeline`:** `aggregation_strategy` changed from `"simple"` to
   `"average"`. `"average"` is a word-level strategy — it groups subwords into
   whole words before assigning a label, so `##`-prefixed fragments structurally
   cannot occur.
2. **`_resolve_overlapping_entities`:** the fast tokenizer's character offsets
   are treated as ground truth. For each in-bounds entity, `entity["text"]` is
   overwritten with the canonical slice `cleaned_text[start:end]` rather than
   trusting the lossy `.word`. Only genuinely out-of-bounds offsets (a real bug
   signal) are discarded, with a clearer warning. `run_ner`'s single-chunk path
   now also routes through `_resolve_overlapping_entities` so entity text is
   canonicalised identically for short and long articles.

**Effect on results.** NER output changes — for the better. Entities previously
discarded are now recognised; this raises recall. Task 3 entity counts will
shift accordingly; this is correct, not a regression.

**Tests.** `test_resolve_overlapping_entities_discards_offset_mismatch` (which
asserted the old discard-on-mismatch behaviour) was replaced by two tests:
`test_resolve_overlapping_entities_corrects_text_from_offsets` (in-bounds →
text corrected from the slice) and
`test_resolve_overlapping_entities_discards_out_of_bounds_offsets`
(out-of-bounds → discarded).

**Notebook.** A markdown design-note cell was added above Cell 11 summarising
the change for reviewers.

**Files.** `src/ner.py`, `tests/test_ner.py`,
`notebooks/analysis.ipynb` (new markdown cell `ner-aggregation-note`),
`docs/SPEC_v3.md`
