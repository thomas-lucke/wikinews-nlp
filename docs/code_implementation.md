# Phase 5 — Code Implementation Guide
### Wikinews NLP Analysis Pipeline — SPEC_v3.md

Copy each prompt into a new Claude Code session. Complete each task fully and run the tests before moving to the next.

---

## Task List

- [Phase 5 — Code Implementation Guide](#phase-5--code-implementation-guide)
    - [Wikinews NLP Analysis Pipeline — SPEC\_v3.md](#wikinews-nlp-analysis-pipeline--spec_v3md)
  - [Task List](#task-list)
  - [Task 1 — Project scaffold and config](#task-1--project-scaffold-and-config)
  - [Task 2 — src/utils.py](#task-2--srcutilspy)
  - [Task 3 — src/data\_loader.py](#task-3--srcdata_loaderpy)
  - [Task 4 — src/data\_inspector.py — format detection and raw profiling](#task-4--srcdata_inspectorpy--format-detection-and-raw-profiling)
  - [Task 5 — src/data\_inspector.py — category profile and validation](#task-5--srcdata_inspectorpy--category-profile-and-validation)
  - [Task 6 — src/data\_normalizer.py — helpers and load\_raw\_records](#task-6--srcdata_normalizerpy--helpers-and-load_raw_records)
  - [Task 7 — src/data\_normalizer.py — normalise\_articles](#task-7--srcdata_normalizerpy--normalise_articles)
  - [Task 8 — src/preprocessing.py](#task-8--srcpreprocessingpy)
  - [Task 9 — src/ner.py — chunking and pipeline](#task-9--srcnerpy--chunking-and-pipeline)
  - [Task 10 — src/ner.py — run\_ner and analysis functions](#task-10--srcnerpy--run_ner-and-analysis-functions)
  - [Task 11 — src/summarizer.py](#task-11--srcsummarizerpy)
  - [Task 12 — src/similarity.py](#task-12--srcsimilaritypy)
  - [Task 13 — src/topic\_predictor.py](#task-13--srctopic_predictorpy)
  - [Task 14 — tests/conftest.py and test\_data\_loader.py](#task-14--testsconftestpy-and-test_data_loaderpy)
  - [Task 15 — tests/test\_data\_inspector.py](#task-15--teststest_data_inspectorpy)
  - [Task 16 — tests/test\_data\_normalizer.py](#task-16--teststest_data_normalizerpy)
  - [Task 17 — tests/test\_preprocessing.py](#task-17--teststest_preprocessingpy)
  - [Task 18 — tests/test\_ner.py](#task-18--teststest_nerpy)
  - [Task 19 — tests/test\_summarizer.py](#task-19--teststest_summarizerpy)
  - [Task 20 — tests/test\_similarity.py and test\_topic\_predictor.py](#task-20--teststest_similaritypy-and-test_topic_predictorpy)
  - [Task 21 — scripts/review\_spec.py](#task-21--scriptsreview_specpy)
  - [Task 22 — notebooks/analysis.ipynb](#task-22--notebooksanalysisipynb)
  - [Final verification](#final-verification)

---

## Task 1 — Project scaffold and config

```
Create the full directory and file scaffold for the Wikinews NLP project, then write config/config.yaml and the project root files. This project uses `uv` for dependency management — NOT pip + requirements.txt.

Project root: use the current working directory.

Directory structure to create:
  config/
  src/
  notebooks/
  docs/decisions/
  tests/
  scripts/
  data/raw/
  data/processed/
  logs/

Files to create (content specified below):

--- config/config.yaml ---
Exact content:

data:
  source_url: "https://github.com/PrimerAI/WikiNews-multilingual"
  raw_path: "data/raw"
  min_article_length: 100

topics:
  selected:
    - "Politics and conflicts"
    - "Science and technology"
    - "Sports"
  articles_per_topic_min: 10
  articles_per_topic_max: 20

countries:
  selected:
    - "United States"
    - "Germany"

languages:
  ner: ["en", "de"]
  summarization: ["en"]

models:
  ner_english: "dslim/bert-base-NER"
  ner_german: "Davlan/bert-base-multilingual-cased-ner-hrl"
  spacy_english: "en_core_web_sm"
  spacy_german: "de_core_news_sm"
  summarization: "facebook/bart-large-cnn"
  similarity: "sentence-transformers/all-MiniLM-L6-v2"
  topic_prediction: "facebook/bart-large-mnli"

summarization:
  min_summary_length: 50
  max_summary_length: 200

ner:
  chunk_size: 400
  chunk_overlap: 50
  error_score_threshold: 0.6

topic_prediction:
  sample_size: 30
  hypothesis_template: "This news article is about {}."

similarity:
  threshold: 0.8

logging:
  log_file: "logs/pipeline.log"

random_seed: 42

--- .gitignore ---
data/
logs/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/

--- .env.example ---
ANTHROPIC_API_KEY=your_key_here

--- pyproject.toml ---
Exact content (uv-compatible, PEP 621 metadata):

[project]
name = "wikinews-nlp"
version = "0.1.0"
description = "Wikinews multilingual NLP analysis pipeline"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1.0",
    "transformers>=4.40.0,<5.0.0",
    "sentence-transformers>=2.7.0,<3.0.0",
    "spacy>=3.7.0,<4.0.0",
    "pandas>=2.0.0",
    "matplotlib>=3.8.0",
    "pyyaml>=6.0",
    "requests>=2.31.0",
    "mwparserfromhell>=0.6.4",
    "jupyter>=1.0.0",
    "python-dotenv>=1.0.0",
    "anthropic>=0.40.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.12.0",
    "black>=24.0.0",
    "ruff>=0.4.0",
]

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "I"]

[tool.black]
line-length = 88

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

--- .pre-commit-config.yaml ---
repos:
  - repo: https://github.com/psf/black
    rev: 24.3.0
    hooks:
      - id: black
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.0
    hooks:
      - id: ruff
        args: [--fix]

--- src/__init__.py ---
(empty file)

--- tests/__init__.py ---
(empty file)

Also create empty placeholder files (just a module-level docstring is fine) for:
  src/utils.py
  src/data_loader.py
  src/data_inspector.py
  src/data_normalizer.py
  src/preprocessing.py
  src/ner.py
  src/summarizer.py
  src/similarity.py
  src/topic_predictor.py

Do not implement any logic yet — just stubs so imports resolve.

Dependency installation: after creating pyproject.toml, run:
  uv sync
This creates .venv/ and installs all dependencies (including the dev group).

Then install the spaCy language models:
  uv run python -m spacy download en_core_web_sm
  uv run python -m spacy download de_core_news_sm

Do NOT create requirements.txt. uv uses pyproject.toml + uv.lock as the source of truth.

Acceptance criterion: all directories and files exist, `uv sync` completes successfully, and `uv run python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"` exits without error.
```

---

## Task 2 — src/utils.py

```
Implement src/utils.py exactly as specified below.

The file must contain:

1. Module-level imports: gc, logging, sys, pathlib.Path, torch

2. Module-level logger:
   logger = logging.getLogger(__name__)

3. Constant:
   RANDOM_SEED: int = 42

4. Function setup_logging(log_file: str = "logs/pipeline.log", level: int = logging.INFO) -> None
   - Creates the log directory if it does not exist (using pathlib.Path)
   - Clears all existing handlers on the root logger before adding new ones
     (prevents duplicate output on Jupyter notebook re-run)
   - Adds two handlers: StreamHandler(sys.stdout) and FileHandler(log_path, mode="a", encoding="utf-8")
   - Format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
   - Date format: "%Y-%m-%d %H:%M:%S"
   - Sets the root logger level to `level`

5. Function get_device() -> int
   - Returns 0 if torch.cuda.is_available() else -1
   - No logging inside this function

6. Function release_model() -> None
   - Calls gc.collect()
   - If torch.cuda.is_available(): calls torch.cuda.empty_cache()
   - Logs: logger.info("GPU cache cleared.")
   - DOES NOT take a model as an argument — the caller must del their own reference first

No other functions or classes.

Acceptance criterion: `uv run python -c "from src.utils import setup_logging, get_device, release_model, RANDOM_SEED; print('ok')"` exits without error.
```

---

## Task 3 — src/data_loader.py

```
Implement src/data_loader.py with the single public function download_dataset.

Read docs/SPEC_v3.md section "src/data_loader.py" for the full specification. Key rules:

1. Module-level logger: logger = logging.getLogger(__name__)

2. Function signature:
   def download_dataset(source_url: str, raw_path: str) -> Path

3. Skip condition: raw_path already exists AND contains at least 1 file with a recognised
   data extension (.json, .jsonl, .csv, .tsv, .txt) AND total size of recognised data files
   is > 100KB. If all three are true, log a skip message and return Path(raw_path).resolve().

4. GitHub URLs (contain "github.com"):
   - Try git clone via subprocess first
   - If git not on PATH, fall back to ZIP download
   - Try branch names in order: main, master
   - If neither returns HTTP 200, raise RuntimeError

5. Direct file URLs (.zip, .tar.gz, .gz, .csv, .json, .jsonl):
   - Download with requests(timeout=60, stream=True)
   - Extract if compressed

6. Archive extraction:
   - .zip → zipfile.ZipFile
   - .tar.gz / .tgz → tarfile
   - If archive contains exactly one top-level directory, strip it and place files directly in raw_path

7. Retry: one retry on requests.Timeout or requests.ConnectionError, with a 5-second delay.
   HTTP 4xx and extraction failures do NOT retry.

8. Returns: Path(raw_path).resolve() (absolute path)

9. Raises:
   - RuntimeError on download failure after retry or ZIP fallback exhausted
   - EnvironmentError if git required but unavailable and ZIP also fails

All paths use pathlib.Path. All file opens use encoding="utf-8" where applicable.

Acceptance criterion: the function exists, is importable, and the skip-condition branch
works correctly (test by creating a temp dir with a dummy .jsonl file > 100KB and calling
the function — it should return immediately without network access).
```

---

## Task 4 — src/data_inspector.py — format detection and raw profiling

```
Implement the first half of src/data_inspector.py: the dataclasses, detect_format, raw_profile,
and load_raw_records (load_raw_records is shared and needed by both inspector and normalizer).

Read docs/SPEC_v3.md section "src/data_inspector.py" for the full specification.

Implement these in order:

1. Imports: logging, dataclasses (dataclass, field), pathlib.Path, typing.Optional, pandas as pd

2. Module-level logger: logger = logging.getLogger(__name__)

3. Dataclass RawProfile with fields:
   total_records: int
   detected_format: str
   detected_fields: list[str]
   file_count: int
   total_size_bytes: int
   sample_record: dict

4. Dataclass NormalisedValidation with fields (exactly as in SPEC):
   total_articles, languages_found, countries_found, topics_found,
   country_topic_counts, missing_date_count, missing_title_count,
   very_short_article_count, country_topics_below_minimum,
   topics_missing_from_config, countries_missing_from_config,
   validation_passed, warnings (default_factory=list), errors (default_factory=list)

5. Function detect_format(raw_path: str) -> str
   Detection order:
   a. Single file: check extension → "json", "jsonl", "csv", "tsv", "zip", or "unknown"
   b. Directory: collect direct-child files only (non-recursive). Compute dominance using
      recognised data files only (.txt, .json, .jsonl, .csv, .tsv). Ignore README.md and
      non-data files for dominance. If ≥80% of recognised data files share one extension,
      return that format string. Otherwise return "unknown".
   c. Attempt to parse one record from the detected format to confirm. If parsing fails,
      return "unknown".

6. Function load_raw_records(raw_path: str, detected_format: str) -> list[dict]
   - For JSON: load top-level. If list → return as-is. If dict with one list-valued key →
     return that list. If dict with multiple keys, none containing a list → wrap in list,
     log warning. Skip keys whose list contains non-dicts.
   - For JSONL: one dict per line. Skip blank lines and unparseable lines (log warning).
   - For CSV/TSV: csv.DictReader. All values are strings.
   - For directory-of-txt: each .txt file → {"text": contents, "id": stem, "language": None,
     "topic": None, "date": None}. Read with encoding="utf-8", errors="replace". Sort by stem.
   - For JSON/JSONL/CSV: read files with encoding="utf-8" (no error replacement — let
     UnicodeDecodeError propagate).
   - If directory for JSON/JSONL/CSV: read only direct-child files with dominant extension,
     sorted by filename. Log warning for recognised data files with non-dominant extensions.
   - For "unknown": raise ValueError.
   - Raises FileNotFoundError if raw_path does not exist.

7. Function raw_profile(raw_path: str, detected_format: str) -> RawProfile
   - Uses load_raw_records to count records and collect field names (union across all records).
   - For directory-of-txt: detected_fields = []
   - sample_record = first valid record
   - total_size_bytes = sum of file sizes

8. Function print_raw_profile(profile: RawProfile) -> None
   - Logs via logger.info(): format, file count, total size, record count, field names,
     sample record keys. Truncates text fields in sample to 100 chars. Never prints full text.

Do NOT implement category_profile or validate_normalised yet — those are Task 5.

Acceptance criterion: `uv run python -c "from src.data_inspector import detect_format, raw_profile, load_raw_records; print('ok')"` exits without error.
```

---

## Task 5 — src/data_inspector.py — category profile and validation

```
Add the remaining functions to src/data_inspector.py. Do NOT rewrite existing functions.

Read docs/SPEC_v3.md section "src/data_inspector.py" for the full specification.

Implement these three functions:

1. category_profile(raw_path: str, detected_format: str) -> pd.DataFrame
   - Uses load_raw_records internally.
   - For each parseable record: if "categories" is a list, count each unique string item
     once (deduplicate within record). If "categories" is a string, count it. If missing/null
     or wrong type: increment internal missing_categories_count, continue.
   - Strip whitespace from each category label.
   - Return DataFrame sorted by count descending, then category ascending.
   - Columns: category (str), count (int), percent (float).
   - Set df.attrs["total_records"] and df.attrs["missing_categories_count"].
   - If no categories field exists anywhere, return empty DataFrame with correct columns
     and attrs populated.

2. validate_normalised(articles: list[dict], config: dict) -> NormalisedValidation
   - Compute all fields of NormalisedValidation from the article list.
   - Normalise topic/country comparisons with lowercase+strip on both sides.
   - very_short_article_count: count articles where len(text) < config["data"]["min_article_length"] * 5
   - Errors (set validation_passed=False): total_articles==0, all selected topics missing,
     all selected countries missing.
   - Warnings (do not fail): missing_date_count > 5% of total, very_short_article_count > 10%,
     any (country,topic) below minimum, any topic/country from config missing from articles.

3. print_category_profile(df: pd.DataFrame, top_n: int = 50) -> None
   - Logs via logger.info(): total records, missing categories count, top_n rows.
   - Ends with: "Review these category labels, update config.yaml topics.selected and
     countries.selected if needed, then rerun Cell 1 or reload config before continuing."
   - Does not prompt for input, does not modify config.yaml.

4. print_validation_report(report: NormalisedValidation) -> None
   - Logs normal fields with logger.info(), warnings with logger.warning(), errors with logger.error().
   - Ends with "Validation passed." or "Validation FAILED — review errors above."
   - No ANSI colour codes.

Acceptance criterion: `uv run python -c "from src.data_inspector import category_profile, validate_normalised; print('ok')"` exits without error.
```

---

## Task 6 — src/data_normalizer.py — helpers and load_raw_records

```
Implement src/data_normalizer.py — the helper functions, constants, and DroppedRecord dataclass.
Do NOT implement normalise_articles yet — that is Task 7.

Read docs/SPEC_v3.md section "src/data_normalizer.py" for the full specification.

Implement:

1. Imports: hashlib, logging, random, dataclasses.dataclass, pathlib.Path, typing.Optional.
   Also add a module-level import: `from src.data_inspector import load_raw_records`
   (the spec's import block at line 855 of SPEC_v3.md omits this, but the test
   harness patches `src.data_normalizer.load_raw_records` — see Task 16 — so the
   name must exist on the data_normalizer module).

2. Module-level logger: logger = logging.getLogger(__name__)

3. FIELD_MAPPINGS dict — copy exactly from SPEC. The exact key order matters:
   the first matching key wins per internal field. Do NOT reorder.
   Key mapping includes "pageid" → "event_id" and excludes "url".

4. Dataclass DroppedRecord:
   article_index: int
   reason: str   # one of: "text_too_short" | "language_not_in_config" |
                 #         "topic_not_in_config" | "country_not_in_config" |
                 #         "duplicate" | "no_text_field"
   field_values: dict

5. Function _generate_stable_id(text: str) -> str
   Returns first 16 hex chars of sha256(text.encode("utf-8")).hexdigest()

6. Function _normalise_topic_string(s: str) -> str
   Returns s.lower().strip()

7. Function _select_from_categories(categories: list[str], selected_labels: list[str]) -> Optional[str]
   - Iterate selected_labels in order (config order, not source order).
   - For each label, check if _normalise_topic_string(label) matches any
     _normalise_topic_string(cat) in categories.
   - Return _normalise_topic_string of the first matching label, or None.

8. Function _infer_text_field(raw_record: dict, min_article_length: int) -> Optional[str]
   - Find a field NOT in FIELD_MAPPINGS whose value is a str with
     len(value.strip()) >= min_article_length.
   - If multiple candidates, return the one with the longest stripped value.
   - Return None if no candidate found.
   - Return the FIELD NAME (key), not the value.

9. Function print_normalisation_summary(valid: list[dict], dropped: list[DroppedRecord]) -> None
   - Log via logger.info(): total valid, breakdown by (country, language, topic),
     total dropped with count per reason.

Note: load_raw_records lives in src/data_inspector.py and is imported from there when needed.
data_normalizer.py does NOT re-implement load_raw_records.

Acceptance criterion: `uv run python -c "from src.data_normalizer import FIELD_MAPPINGS, DroppedRecord, _generate_stable_id, _select_from_categories; print('ok')"` exits without error.
```

---

## Task 7 — src/data_normalizer.py — normalise_articles

```
Add normalise_articles to src/data_normalizer.py. Do NOT rewrite existing functions.

Read docs/SPEC_v3.md section "src/data_normalizer.py" for the full specification.
Pay special attention to the RNG rules and processing order — these are critical for correctness.

Function signature:
def normalise_articles(
    raw_path: str,
    detected_format: str,
    languages: list[str],
    topics: list[str],
    countries: list[str],
    max_per_topic: int,
    min_article_length: int,
    random_seed: int = 42,
) -> tuple[list[dict], list[DroppedRecord]]

Processing order (follow exactly):

1. Pipeline-level validation: raise ValueError if languages, topics, or countries is empty.

2. Create ONE stateful RNG: rng = random.Random(random_seed)
   Do NOT create another random.Random inside the function.

3. Load records via load_raw_records (already imported at module level per Task 6).

4. For each record (track original index for DroppedRecord):
   a. Apply FIELD_MAPPINGS in insertion order — iterate FIELD_MAPPINGS keys, check if
      each key exists in the raw record. First matching key per internal field wins.
      Track has_source_id: True if any of "id", "article_id", "uid" was present.
   b. For the "text" slot: if no FIELD_MAPPINGS key matched, try _infer_text_field().
      Log inferred fields with logger.info.
   c. Drop if: no text found OR len(text.strip()) < min_article_length
      → reason "no_text_field" or "text_too_short"
   d. Drop if: language missing or not in languages list → reason "language_not_in_config"
      Language comparison: exact lowercase ISO 639-1 match. No inference.
   e. Drop if: topic missing or no config topic matches → reason "topic_not_in_config"
      For Wikinews (has "categories" list): use _select_from_categories(categories, topics).
      For others with string topic field: use _normalise_topic_string on both sides.
   f. Drop if: country missing or no config country matches → reason "country_not_in_config"
      Same pattern as topic.

5. Deduplicate by text hash: _generate_stable_id(article["text"]). First occurrence wins.
   Drop with reason "duplicate". Build in O(n) — no sorting.

6. For each (country, language, topic) group, if count > max_per_topic:
   - Sort by str(source_id_value) if source id exists, else by position in post-dedup list.
   - Sample max_per_topic using rng.sample() (the single stateful rng from step 2).

7. Set missing optional fields:
   - date=None if absent
   - title="" if absent
   - event_id=None if absent
   - id: if has_source_id, keep existing value. If not, generate via _generate_stable_id(text).
   - event_id: if pageid was present, store str(raw_value).

8. Normalise topic and country: store lowercased stripped strings (_normalise_topic_string).
   Note: for the Wikinews path, _select_from_categories already returns the value lowercased
   and stripped, so this step is idempotent there. For non-Wikinews sources with a string
   topic/category or country field, this is the step that performs the normalisation
   before storing into the article dict.

Return (valid_articles, dropped_records).

Acceptance criterion: make these tests pass:
- Function returns a tuple of (list, list)
- Passing empty languages=[] raises ValueError
- A Wikinews-style record with pageid stores it in event_id, not id
- Two identical text records: second is dropped with reason "duplicate"
```

---

## Task 8 — src/preprocessing.py

```
Implement src/preprocessing.py exactly as specified.

Read docs/SPEC_v3.md section "src/preprocessing.py" for the full specification.

Implement:

1. Imports: logging, re, typing.Optional, spacy

2. Module-level logger and model cache:
   logger = logging.getLogger(__name__)
   _SPACY_MODELS: dict[str, spacy.Language] = {}

3. Function _get_spacy_model(language: str, model_name: str) -> spacy.Language
   - Returns cached model if already loaded (keyed by language).
   - On first call: spacy.load(model_name), set nlp.max_length = 2_000_000, cache it.
   - If OSError: raise RuntimeError with installation instructions:
     f"spaCy model '{model_name}' not found. Run: python -m spacy download {model_name}"

4. Function clean_text(text: str) -> str
   Apply in this exact order:
   a. MediaWiki templates: try mwparserfromhell.parse(text).strip_code().
      If mwparserfromhell not installed (ImportError): use regex r'\{\{[^}]*\}\}' → ""
      and log a warning about nested templates.
   b. MediaWiki links: r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]' → r'\1'
   c. HTML tags: r'<[^>]+>' → ""
   d. URLs: r'https?://\S+' → ""
   e. Multiple whitespace/newlines: → single space
   f. Strip leading/trailing whitespace.
   Punctuation is NOT removed.

5. Function tokenize_and_tag(text: str, language: str, model_name: str) -> dict
   - Calls _get_spacy_model(language, model_name) to get the cached model.
   - Runs the spaCy model on text.
   - Returns dict with:
     "sentences": list[str]  — sentence strings from spaCy senter
     "tokens": list[str]     — non-whitespace token strings (token.is_space == False)
     "pos_tags": list[tuple[str,str]]  — [(token.text, token.pos_) for non-space tokens]
   - POS tags use Universal POS tagset (token.pos_, NOT token.tag_).

6. Function preprocess_articles(articles: list[dict], config: dict) -> list[dict]
   - Group articles by language.
   - For each language group:
     a. Run clean_text on each article's "text" field.
     b. Feed cleaned strings to nlp.pipe() for batch processing.
     c. Extract sentences, tokens, pos_tags from each Doc.
     d. Write "cleaned_text", "sentences", "tokens", "pos_tags" to the article dict in-place.
   - Articles with unrecognised language: skip with logger.warning, do not add fields.
   - Per-article errors: catch Exception, log with article id, leave article in list without fields.
   - Read spaCy model names from config["models"]["spacy_english"] and config["models"]["spacy_german"].
   - Returns same list (mutated in-place).

Acceptance criterion: run `uv run python -c "from src.preprocessing import clean_text; print(clean_text('[[Berlin|Berlin, Germany]] {{cite}} text'))"` — should output "Berlin, Germany text" (single space — step e collapses runs of whitespace and step f strips).
```

---

## Task 9 — src/ner.py — chunking and pipeline

```
Implement the first half of src/ner.py: the config validator, pipeline loader, and chunking logic.
Do NOT implement run_ner or analysis functions yet — those are Task 10.

Read docs/SPEC_v3.md section "src/ner.py" for the full specification.

Implement:

1. Imports: logging, collections.Counter, typing.Optional, pandas as pd,
   transformers.pipeline as hf_pipeline, src.utils.get_device

2. Module-level logger: logger = logging.getLogger(__name__)

3. Function validate_ner_config(config: dict) -> None
   Raise ValueError if:
   - config["ner"]["chunk_size"] <= 0
   - config["ner"]["chunk_overlap"] >= config["ner"]["chunk_size"]
   - config["ner"]["chunk_overlap"] < 0

4. Function load_ner_pipeline(model_name: str, language: str) -> object
   - Calls hf_pipeline("ner", model=model_name, device=get_device(),
     aggregation_strategy="simple")
   - Logs model load.
   - Raises RuntimeError on failure.

5. Function _chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]
   Algorithm (follow exactly):
   - If text is empty: return [].
   - Validate: chunk_size <= 0 → ValueError. overlap >= chunk_size → ValueError.
   - start = 0
   - Loop:
     - If len(text) - start <= chunk_size: add (text[start:], start) as final chunk. Break.
     - Search backwards from (start + chunk_size) for the nearest whitespace.
     - If whitespace found: end = whitespace position. Chunk = text[start:end].
     - If no whitespace: end = start + chunk_size. Hard-break. Log warning.
     - Append (text[start:end], start).
     - Compute next_start = end - overlap. Search backwards from there within the bounded
       window (no further back than end - chunk_size) for whitespace. If found, adjust
       next_start to that position. If not found, use (end - overlap) as-is.
     - Progress guard: if next_start <= start, set next_start = start + 1
       (defensive — guarantees forward progress when the whitespace search lands at or
       before the current start, e.g. on degenerate inputs where overlap is near chunk_size).
     - start = next_start.
   - Returns list of (chunk_str, start_offset) tuples.

6. Function _resolve_overlapping_entities(entities: list[dict], cleaned_text: str) -> list[dict]
   - Exact duplicates (same start, end, label): keep higher score.
   - Partial overlaps: keep the entity with larger (end - start). Tie → higher score.
   - Offset validation: verify cleaned_text[start:end] == entity["text"].
     If mismatch: log warning and discard.
   - Sort final list by start offset ascending.

Acceptance criterion: write a small inline test:
  chunks = _chunk_text("word " * 100, chunk_size=50, overlap=10)
  assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)
  assert chunks[0][1] == 0
Run `uv run python -c "from src.ner import _chunk_text; chunks = _chunk_text('word ' * 100, 50, 10); print(len(chunks), 'chunks ok')"`.
```

---

## Task 10 — src/ner.py — run_ner and analysis functions

```
Add the remaining functions to src/ner.py. Do NOT rewrite existing functions.

Read docs/SPEC_v3.md section "src/ner.py" for the full specification.

Implement:

1. Function run_ner(articles, ner_pipeline, language, chunk_size, chunk_overlap) -> list[dict]
   - Only process articles where article["language"] == language.
   - For articles without "cleaned_text": set entities=[], log warning.
   - For articles where len(cleaned_text) <= chunk_size: run pipeline directly on cleaned_text.
   - For longer articles: chunk with _chunk_text, run pipeline on each chunk, adjust start/end
     offsets by chunk's start_offset, then call _resolve_overlapping_entities.
   - KEY RENAME (mandatory): HuggingFace returns "entity_group" — rename to "label".
     Also keep: "text" (from "word"), "start", "end", "score".
     Entity dict: {"text": ..., "label": ..., "start": ..., "end": ..., "score": ...}
   - Set entities=[] (not None) if NER ran but found nothing, or cleaned_text empty.
   - Set entities=None for articles where language doesn't match (NER not run).
   - Per-article errors: catch Exception, log with article id, set entities=[], continue.
   - Returns same list with "entities" field added.

2. Function build_entity_dataframe(articles: list[dict]) -> pd.DataFrame
   - Flatten all entities from articles that have the "entities" key and entities is not None.
   - Skip articles where "entities" key is absent or entities is None.
   - Include articles where entities == [] (they produce 0 rows — do not add a guard that skips them).
   - Title handling: if article["title"] == "", use f"[id: {article['id']}]"
   - Columns: article_id, event_id, title, date, language, country, topic,
              entity_text, entity_label, score
   - Return empty DataFrame with correct columns if no entities found.

3. Function plot_top_entities(df, top_n, language, country=None) -> None
   - Filter df to language (and country if provided).
   - Count distinct article_id per entity_text (not total rows).
   - Group by entity_text only (merge different labels for same surface string).
   - Plot horizontal bar chart of top_n.
   - Title: f"Top {top_n} entities — {country} — {language.upper()}" or without country.
   - Use matplotlib. Do NOT call plt.show().

4. Function plot_entity_dynamics(df, entity_names, language, country=None) -> None
   - Filter to language and optional country.
   - Filter rows to df["entity_text"].isin(entity_names) — only the named entities are plotted.
   - Parse date with pd.to_datetime(errors="coerce"). Drop NaT rows.
   - Create year_month = parsed_dates.dt.to_period("M").
   - Group by (entity_text, year_month). Count unique article_id per group.
   - Plot one line per entity. X-axis: period strings ("%Y-%m"). Y-axis: article count.
   - If fewer than 3 data points for any entity: log warning.
   - Use matplotlib. Do NOT call plt.show().

5. Function investigate_ner_errors(articles, language, error_score_threshold, country=None) -> pd.DataFrame
   - Collect entities where score < error_score_threshold.
   - Filter to language and optional country.
   - MISC label is NOT treated as an error — only confidence matters.
   - Columns: article_id, event_id, country, title, entity_text, entity_label, score.
   - Sort by score ascending.

Acceptance criterion: `uv run python -c "from src.ner import run_ner, build_entity_dataframe; print('ok')"` exits without error.
```

---

## Task 11 — src/summarizer.py

```
Implement src/summarizer.py exactly as specified.

Read docs/SPEC_v3.md section "src/summarizer.py" for the full specification.

Implement:

1. Imports: logging, pandas as pd, transformers.pipeline as hf_pipeline, src.utils.get_device
   Also: from typing import Optional

2. Module-level logger: logger = logging.getLogger(__name__)

3. validate_summarization_config(config: dict) -> None
   Raise ValueError if config["summarization"]["min_summary_length"] >= config["summarization"]["max_summary_length"]

4. load_summarization_pipeline(model_name: str) -> object
   - hf_pipeline("summarization", model=model_name, device=get_device())
   - Log model load.

5. summarize_article(text: str, summ_pipeline: object, min_length: int, max_length: int) -> Optional[str]
   - If text is empty string: log warning, return None.
   - Short-article guard:
     token_count = len(summ_pipeline.tokenizer.encode(text, add_special_tokens=False))
     If token_count < min_length: log warning, return None.
   - Call: summ_pipeline(text, truncation=True, min_length=min_length, max_length=max_length)
     Use these exact kwarg names.
   - Return result[0]["summary_text"].
   - If pipeline raises: log error, return None.

6. summarize_articles(articles: list[dict], summ_pipeline: object, config: dict) -> list[dict]
   - Qualifying articles: those where article["language"] is in config["languages"]["summarization"].
   - Do NOT hardcode "en" — read from config.
   - Count qualifying articles. Log progress every 5: logger.info("Summarised %d/%d articles", n_done, n_total)
   - For each qualifying article: call summarize_article with article["cleaned_text"] and config values.
   - Store result in article["summary"] (may be None).
   - Non-qualifying articles do NOT get a "summary" key.
   - Returns same list.

7. build_summary_quality_dataframe(articles: list[dict]) -> pd.DataFrame
   - Include only articles where article.get("summary") is not None.
   - For each summary compute:
     article_id, title (use id if title==""), country, topic,
     summary_char_count, summary_sentence_count (split on [.!?]),
     avg_sentence_chars, missing_terminal_punctuation (bool),
     repeated_whitespace (bool), very_long_sentence (bool — any sentence > 250 chars),
     issue_count (sum of boolean flags)
   - Return empty DataFrame with correct columns if no summaries.

Acceptance criterion: run tests from task 19 after implementation.
`uv run python -c "from src.summarizer import validate_summarization_config, summarize_article; print('ok')"` exits without error.
```

---

## Task 12 — src/similarity.py

```
Implement src/similarity.py exactly as specified.

Read docs/SPEC_v3.md section "src/similarity.py" for the full specification.

Implement:

1. Imports: logging, typing.Optional, pandas as pd,
   sentence_transformers.SentenceTransformer, sentence_transformers.util

2. Module-level logger: logger = logging.getLogger(__name__)

3. load_embedding_model(model_name: str) -> SentenceTransformer
   - SentenceTransformer(model_name)
   - Log model load.

4. calculate_similarity(original: str, summary: str, model: SentenceTransformer) -> float
   - Encode both strings with model.encode().
   - Compute cosine similarity with util.cos_sim().
   - cos_sim() returns shape (1,1) — extract with float(result[0][0]).
   - Return Python float. Do NOT assume a lower bound of 0.

5. score_all_articles(articles: list[dict], model: SentenceTransformer) -> list[dict]
   - Qualifying: article.get("cleaned_text") is not None and != "" AND article.get("summary") is not None.
   - Use .get() for both — non-English articles have no "summary" key.
   - Skip non-qualifying articles without logging.
   - Per-article errors: catch, log, leave "similarity_score" unset.
   - Returns same list with "similarity_score" added where applicable.

6. build_similarity_dataframe(articles: list[dict]) -> pd.DataFrame
   - Include articles that have "similarity_score" field.
   - Columns: article_id, title, country, topic, similarity_score.
   - Empty DataFrame with correct columns if no scored articles.

7. plot_similarity_distribution(df: pd.DataFrame, threshold: float) -> None
   - One subplot per (country, topic) pair.
   - x-axis: shared range [-1.0, 1.0] (sharex=True).
   - y-axis: independent per subplot (NOT sharey). Label: "Article count".
   - 20 bins across [-1.0, 1.0].
   - Vertical dashed line at threshold on every subplot.
   - Subplot title: f"{country} — {topic} (n={article_count})".
   - Overall title: "Similarity score distribution by country and topic".
   - Do NOT call plt.show().

8. explain_similarity_extremes(df: pd.DataFrame, n: int = 3) -> dict
   - Find top n (highest score) and bottom n (lowest score).
   - Tie-breaking: sort by str(article_id) lexicographically.
   - Return: {"highest": list of n dicts, "lowest": list of n dicts}
   - Each dict: article_id, title, country, topic, similarity_score.

Acceptance criterion: `uv run python -c "from src.similarity import calculate_similarity, score_all_articles; print('ok')"` exits without error.
```

---

## Task 13 — src/topic_predictor.py

```
Implement src/topic_predictor.py exactly as specified.

Read docs/SPEC_v3.md section "src/topic_predictor.py" for the full specification.

Implement:

1. Imports: logging, random, typing.Optional, transformers.pipeline as hf_pipeline, src.utils.get_device

2. Module-level logger: logger = logging.getLogger(__name__)

3. load_topic_pipeline(model_name: str) -> object
   - hf_pipeline("zero-shot-classification", model=model_name, device=get_device())

4. predict_topic(text, candidate_labels, topic_pipeline, hypothesis_template) -> Optional[str]
   - If text is empty: log warning, return None.
   - Call topic_pipeline(text, candidate_labels=candidate_labels, hypothesis_template=hypothesis_template)
   - Return the label with the highest score (result["labels"][0]).
   - If pipeline raises: log warning, return None.

5. predict_all_topics(articles, candidate_labels, topic_pipeline, hypothesis_template, sample_size, random_seed=42) -> list[dict]
   RNG: create ONE stateful object: rng = random.Random(random_seed). Never re-instantiate.

   Pre-filter: exclude articles where article.get("cleaned_text", "") == "". Log exclusion count.

   Log a warning for any article where article.get("language") is not "en".

   Balanced sampling:
   a. Group articles by (country, topic).
   b. quota = floor(sample_size / n_country_topic_groups)
   c. Sort articles within each group by str(article["id"]).
   d. Use rng.sample() to select min(quota, len(group)) from each group.
   e. Track selected article ids in a set.

   Redistribution:
   f. remaining_slots = sample_size - len(selected)
   g. If remaining_slots > 0: collect groups with unsampled articles.
   h. per_group_extra = ceil(remaining_slots / n_groups_with_remainder)
   i. For each such group: draw min(per_group_extra, available_unsampled) using same rng.
      Filter pool to exclude already-selected ids first.
      If pool smaller than requested: rng.sample(pool, len(pool)).
      Stop when remaining_slots reaches 0.

   For each sampled article:
   - Create shallow copy: {**original_article, "predicted_topic": None}
   - Call predict_topic(article.get("cleaned_text", ""), ...)
   - Store result in copy["predicted_topic"].
   - Per-article errors: catch, log, set predicted_topic=None.

   Return list of copied, sampled article dicts.

6. evaluate_topic_predictions(sampled_articles: list[dict]) -> dict
   CRITICAL: article["topic"] is always lowercase. predicted_topic preserves config casing.
   Comparison MUST use lower().strip() on BOTH sides.

   Skip articles where predicted_topic is None.

   Return:
   {
     "accuracy": float,      # correct / evaluated (0.0 if evaluated==0, never NaN)
     "correct": int,
     "evaluated": int,
     "total_sampled": int,
     "results": list[dict],  # title (substituted if ""), match (bool), country, topic, predicted_topic
   }

   Title substitution: if article["title"] == "", use f"[id: {article['id']}]".

Acceptance criterion: `uv run python -c "from src.topic_predictor import predict_topic, evaluate_topic_predictions; print('ok')"` exits without error.
```

---

## Task 14 — tests/conftest.py and test_data_loader.py

```
Create tests/conftest.py and tests/test_data_loader.py.

Read docs/SPEC_v3.md section "conftest.py — shared fixtures" and
"Expected test behaviours — test_data_loader.py" for the full specification.

--- tests/conftest.py ---
Implement exactly as specified in SPEC. The file must contain:

1. sample_en_article fixture — minimal valid English article with all post-normalisation fields.
2. sample_de_article fixture — minimal valid German article.
3. sample_raw_record fixture — raw record with non-standard field names (article_body, headline, etc.)
4. mock_ner_pipeline fixture — MagicMock callable returning raw HuggingFace NER format
   with "entity_group" (NOT "label") and "word" (NOT "text") keys.
   pipeline.return_value = [{"word": "Berlin", "entity_group": "LOC", "score": 0.98, "start": 5, "end": 11}]
5. mock_summ_pipeline fixture — MagicMock with .tokenizer.encode.return_value = list(range(100))
   and .return_value = [{"summary_text": "A short summary of the article."}]
6. mock_embedding_model fixture — MagicMock with .encode.return_value = torch.tensor([[0.5, 0.5, 0.5]])
   Shape MUST be (1, 3) not (3,) — cos_sim requires 2D input.
7. mock_device autouse fixture that monkeypatches:
   src.ner.get_device → lambda: -1
   src.summarizer.get_device → lambda: -1
   src.topic_predictor.get_device → lambda: -1

--- tests/test_data_loader.py ---
Implement these three tests:

1. test_skip_existing_data:
   Create a temp directory with one .jsonl file > 100KB.
   Call download_dataset(source_url="https://example.com", raw_path=str(tmpdir)).
   Assert: returns a Path, no network call was made (monkeypatch requests if needed).

2. test_does_not_skip_empty_dir:
   Create a temp directory with only a .gitkeep file.
   Mock requests.get to raise requests.ConnectionError.
   Call download_dataset and assert it raises RuntimeError (not a skip).

3. test_raises_on_network_failure:
   Mock requests.get to always raise requests.Timeout.
   Create empty temp dir, call download_dataset.
   Assert RuntimeError is raised after retries.

Run: uv run pytest tests/test_data_loader.py -v
All 3 tests must pass.
```

---

## Task 15 — tests/test_data_inspector.py

```
Create tests/test_data_inspector.py.

Read docs/SPEC_v3.md section "Expected test behaviours — test_data_inspector.py".

Implement these tests (use tmp_path fixture for temp directories):

1. test_detect_format_json_directory:
   Create tmp dir with 3 .json files. Assert detect_format returns "json".

2. test_detect_format_txt_directory:
   Create tmp dir with 3 .txt files. Assert detect_format returns "directory-of-txt".

3. test_detect_format_jsonl_with_readme:
   Create tmp dir with one .jsonl file and one README.md.
   Assert detect_format returns "jsonl" (README does not count toward dominance).

4. test_detect_format_empty_directory:
   Empty tmp dir. Assert detect_format returns "unknown".

5. test_category_profile_counts:
   Create a temp JSONL file with 3 records, each with categories=["Sports","Politics and conflicts"].
   Call category_profile. Assert Sports and Politics counts are correct, sorted by count descending.

6. test_category_profile_deduplicates_within_record:
   One record with categories=["Sports","Sports","Sports"].
   Assert Sports count is 1 (not 3).

7. test_category_profile_no_categories_field:
   Records with no "categories" key.
   Assert: returns empty DataFrame with columns ["category","count","percent"],
   df.attrs["missing_categories_count"] == number of records.

8. test_validate_normalised_zero_articles:
   Call validate_normalised([], config).
   Assert validation_passed is False.

9. test_validate_normalised_missing_topic:
   Articles that contain no "politics and conflicts" articles.
   Config topics includes "Politics and conflicts".
   Assert result.topics_missing_from_config is non-empty and validation_passed is True
   (missing topic is a warning, not an error, unless ALL topics are missing).

10. test_validate_normalised_missing_country:
    Similar to above but for countries.

Run: uv run pytest tests/test_data_inspector.py -v
All tests must pass.
```

---

## Task 16 — tests/test_data_normalizer.py

```
Create tests/test_data_normalizer.py.

Read docs/SPEC_v3.md section "Expected test behaviours — test_data_normalizer.py".

Use the sample_raw_record fixture from conftest.py where appropriate.

Mock load_raw_records to return controlled input without file I/O.
Patch target: `src.data_normalizer.load_raw_records`. This requires the import to be
module-level inside data_normalizer.py (`from src.data_inspector import load_raw_records`
at the top of the file), so that the local name `load_raw_records` exists on the
data_normalizer module and `monkeypatch.setattr` / `mocker.patch` can replace it.
Task 6's import list must include this line — adjust if it was omitted.
(If you instead import inside the function, the local name does not exist on the module
and patching at src.data_normalizer.load_raw_records will silently miss; you would
have to patch src.data_inspector.load_raw_records BEFORE the function runs.)

Implement these tests:

1. test_field_mapping_standard_fields:
   Pass a raw record with article_body, headline, publish_date, lang, category, country, article_id, pageid.
   Assert normalised article has: text (from article_body), title (from headline), date (from publish_date),
   language (from lang), id (from article_id).

2. test_pageid_stored_as_event_id:
   Record with pageid=12345. Assert article["event_id"] == "12345" and "id" != "12345".

3. test_wikinews_topic_from_categories:
   Record with categories=["Politics and conflicts", "Germany"].
   Config topics=["Politics and conflicts"]. Assert article["topic"] == "politics and conflicts".

4. test_wikinews_country_from_categories:
   Same record. Config countries=["Germany"]. Assert article["country"] == "germany".

5. test_drop_country_not_in_config:
   Record with categories=["United Kingdom"] when countries=["Germany","United States"].
   Assert DroppedRecord with reason "country_not_in_config".

6. test_drop_text_too_short:
   Record with article_body="short". Assert DroppedRecord with reason "text_too_short".

7. test_drop_language_not_in_config:
   Record with lang="fr" when languages=["en","de"]. Assert reason "language_not_in_config".

8. test_drop_duplicate:
   Two records with identical text. Assert second is dropped with reason "duplicate".

9. test_stable_id_reproducible:
   Run normalise_articles twice on the same record. Assert same id in both runs.

10. test_empty_languages_raises:
    normalise_articles(... languages=[]) raises ValueError.

Run: uv run pytest tests/test_data_normalizer.py -v
All tests must pass.
```

---

## Task 17 — tests/test_preprocessing.py

```
Create tests/test_preprocessing.py.

Read docs/SPEC_v3.md section "Expected test behaviours — test_preprocessing.py".

Implement these tests:

1. test_clean_text_wiki_link_with_display:
   Input: "Visit [[Berlin|Berlin, Germany]] today."
   Assert output contains "Berlin, Germany" and does not contain "[[" or "]]".

2. test_clean_text_wiki_link_no_display:
   Input: "Visit [[Berlin]] today."
   Assert output contains "Berlin" and not "[[Berlin]]".

3. test_clean_text_template_removed:
   Input: "{{cite web|url=http://example.com}} Some text."
   Assert template is removed from output.

4. test_clean_text_html_removed:
   Input: "Hello <b>world</b>."
   Assert output is "Hello world."

5. test_clean_text_url_removed:
   Input: "See https://example.com for details."
   Assert URL is not in output.

6. test_tokenize_and_tag_returns_correct_keys (mock spaCy or use a real model if installed):
   If spaCy model is installed: call tokenize_and_tag on sample_en_article["text"].
   Assert result contains keys "sentences", "tokens", "pos_tags".
   Assert all pos_tags are tuples of (str, str).
   Assert all pos_tag[1] values are valid Universal POS tags from the set:
   {"ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X","SPACE"}.

Note: if spaCy model is not installed in the test environment, mark this test with
@pytest.mark.skipif(not spacy_model_available, reason="spaCy model not installed").

Run: uv run pytest tests/test_preprocessing.py -v
All tests must pass.
```

---

## Task 18 — tests/test_ner.py

```
Create tests/test_ner.py.

Read docs/SPEC_v3.md section "Expected test behaviours — test_ner.py" and
the conftest.py mock_ner_pipeline fixture specification.

All tests must mock the HuggingFace pipeline — no model downloads.
Patch: src.ner.hf_pipeline

Implement these tests:

1. test_run_ner_skips_wrong_language:
   Articles include one English and one German article.
   Call run_ner(..., language="en").
   Assert English article has an "entities" field (list).
   Assert German article has entities=None.

2. test_run_ner_empty_entities:
   mock_ner_pipeline.return_value = [] (no entities found).
   Call run_ner on English article.
   Assert article["entities"] == [] (empty list, not None).

3. test_run_ner_key_rename:
   mock_ner_pipeline returns {"word": "Berlin", "entity_group": "LOC", "score": 0.98, "start": 5, "end": 11}
   Call run_ner. Assert entity dict uses "label" key (not "entity_group") and "text" key (not "word").

4. test_build_entity_dataframe_row_count:
   Article with 3 entities in "entities" field.
   Call build_entity_dataframe. Assert len(df) == 3.

5. test_build_entity_dataframe_skips_none_entities:
   One article with entities=None (NER not run). One with entities=[entity_dict].
   Assert DataFrame has only 1 row.

6. test_build_entity_dataframe_includes_empty_list:
   Article with entities=[] (NER ran, nothing found).
   Assert DataFrame has 0 rows (no error, just empty).

7. test_build_entity_dataframe_no_title_substitution:
   Article with title="" and entities. Assert "title" column contains f"[id: {article['id']}]".

Run: uv run pytest tests/test_ner.py -v
All tests must pass.
```

---

## Task 19 — tests/test_summarizer.py

```
Create tests/test_summarizer.py.

Read docs/SPEC_v3.md section "Expected test behaviours — test_summarizer.py".
Use mock_summ_pipeline fixture from conftest.py.
Patch: src.summarizer.hf_pipeline

Implement these tests:

1. test_validate_config_raises_when_min_gte_max:
   config with min_summary_length=100, max_summary_length=50.
   Assert validate_summarization_config raises ValueError.

2. test_validate_config_raises_when_min_equals_max:
   min_summary_length=100, max_summary_length=100.
   Assert ValueError.

3. test_validate_config_passes_valid:
   min=50, max=200. Assert no exception raised.

4. test_summarize_article_empty_string:
   Call summarize_article("", mock_summ_pipeline, min_length=50, max_length=200).
   Assert returns None.

5. test_summarize_article_short_article_guard:
   Set mock_summ_pipeline.tokenizer.encode.return_value = list(range(10))
   (10 tokens < min_length=50).
   Call summarize_article with non-empty text.
   Assert returns None (short-article guard triggered).

6. test_summarize_article_calls_pipeline_with_correct_kwargs:
   Normal article (tokenizer returns 100 tokens).
   Assert the pipeline was called with truncation=True, min_length=50, max_length=200
   as keyword arguments (not positional).
   Use mock_summ_pipeline.assert_called_once_with(text, truncation=True, min_length=50, max_length=200).

7. test_summarize_articles_english_only:
   articles list with one English and one German article (both have cleaned_text).
   Call summarize_articles.
   Assert English article has "summary" key.
   Assert German article does NOT have "summary" key.

8. test_build_summary_quality_missing_terminal_punctuation:
   Article with summary = "This is a sentence without punctuation".
   Assert missing_terminal_punctuation == True.

9. test_build_summary_quality_repeated_whitespace:
   Article with summary = "This  has   extra   spaces.".
   Assert repeated_whitespace == True.

Run: uv run pytest tests/test_summarizer.py -v
All tests must pass.
```

---

## Task 20 — tests/test_similarity.py and test_topic_predictor.py

```
Create tests/test_similarity.py and tests/test_topic_predictor.py.

Read docs/SPEC_v3.md sections "Expected test behaviours — test_similarity.py" and
"Expected test behaviours — test_topic_predictor.py".
Use mock_embedding_model fixture from conftest.py.
Patch: src.topic_predictor.hf_pipeline

--- tests/test_similarity.py ---

1. test_build_similarity_dataframe_no_summaries:
   Articles with no "similarity_score" field.
   Assert returns empty DataFrame with columns: article_id, title, country, topic, similarity_score.

2. test_explain_similarity_extremes_deterministic:
   Build df with 5 articles, two of which have identical similarity_score.
   Call explain_similarity_extremes twice.
   Assert identical output both times (tie-breaking by str(article_id) is deterministic).

3. test_score_all_articles_skips_no_summary:
   Articles without "summary" key (German articles).
   Call score_all_articles. Assert no "similarity_score" field added to those articles.

4. test_calculate_similarity_returns_float:
   Use mock_embedding_model (returns torch.tensor([[0.5, 0.5, 0.5]])).
   Call calculate_similarity("text a", "text b", mock_embedding_model).
   Assert result is a Python float.

--- tests/test_topic_predictor.py ---

1. test_predict_all_topics_reproducible:
   10 articles with cleaned_text. Pass a MagicMock as topic_pipeline (set
   `mock_pipeline.return_value = {"labels": ["Sports"], "scores": [0.9]}`) — do NOT
   call load_topic_pipeline, which would download the BART-MNLI model.
   Call predict_all_topics twice with random_seed=42, passing the same mock.
   Assert same article ids selected both times.
   Note: patching `src.topic_predictor.hf_pipeline` is not sufficient here because
   predict_all_topics receives the pipeline as a parameter — pass the mock directly.

2. test_evaluate_topic_predictions_all_correct:
   sampled_articles where article["topic"]=="sports" and predicted_topic=="Sports".
   Assert accuracy == 1.0.
   (Tests that case-insensitive comparison works correctly.)

3. test_evaluate_topic_predictions_excludes_none:
   2 articles: one with predicted_topic="Sports", one with predicted_topic=None.
   Assert evaluated==1, total_sampled==2, accuracy==1.0.

4. test_evaluate_topic_predictions_empty_returns_zero_accuracy:
   sampled_articles=[]. Assert accuracy==0.0 (not NaN, not None).

5. test_predict_all_topics_excludes_empty_cleaned_text:
   Mix of articles: some with cleaned_text, some with cleaned_text="".
   Assert articles with empty cleaned_text are not included in the sample.

Run: uv run pytest tests/test_similarity.py tests/test_topic_predictor.py -v
All tests must pass.
```

---

## Task 21 — scripts/review_spec.py

```
Implement scripts/review_spec.py using the Anthropic SDK with extended thinking.

Read docs/SPEC_v3.md section "Using LLMs to review your own work — Job 2" for context.
(docs/ai-engineering-field-guide.md does not exist in this repo — ignore the previous
reference to it; the spec text is self-contained.)

The script:

1. Load .env into the process environment at import/script start:
   `from dotenv import load_dotenv; load_dotenv()`.
   This is required because the error message instructs users to copy .env.example to .env,
   but `anthropic.Anthropic()` only reads from os.environ — without load_dotenv() a .env
   file is invisible to the SDK. python-dotenv is already in dependencies (Task 1).
   Then read ANTHROPIC_API_KEY from environment variable. If missing:
   raise EnvironmentError(
       "ANTHROPIC_API_KEY environment variable is not set. "
       "Copy .env.example to .env and add your key."
   )

2. REVIEWER_PROMPT (exact text):
   "You are a senior engineer doing a hostile review of a technical spec before any code
   is written. Find every ambiguity, every assumption stated as fact, every unjustified
   decision, every edge case not handled, and every place where a coding agent given only
   this spec would have to guess. Do not summarise what the spec does. Report problems
   only, numbered."

3. Function review_spec(spec_path: str) -> str
   - Reads the spec file with encoding="utf-8"
   - Creates anthropic.Anthropic() client (no key arg — reads from env automatically)
   - Calls client.messages.create with:
     model="claude-opus-4-7"
     max_tokens=16000
     thinking={"type": "enabled", "budget_tokens": 10000}
     system=REVIEWER_PROMPT
     messages=[{"role": "user", "content": spec_text}]
   - Returns only text blocks (not thinking blocks):
     "\n".join(block.text for block in response.content if block.type == "text")

4. if __name__ == "__main__" block:
   - Accepts optional command-line argument for spec path (default: "docs/SPEC_v3.md")
   - Calls review_spec() and prints the result

Acceptance criterion: `uv run python -c "import sys; sys.path.insert(0, 'scripts'); from review_spec import review_spec; print('ok')"` exits without error (does not need to actually call the API).

Note: the `anthropic` package is already in pyproject.toml dependencies — no separate install needed.
```

---

## Task 22 — notebooks/analysis.ipynb

```
Create notebooks/analysis.ipynb with all 17 cells exactly as specified in SPEC_v3.md
section "Notebook orchestration".

Read docs/SPEC_v3.md section "Notebook orchestration" — it contains the exact code
for every cell.

Rules:
- Cell 1: Setup — config loading, logging, imports. Include the FileNotFoundError guard.
- Cell 2: Validate config — validate_summarization_config and validate_ner_config.
- Cell 3: Download data.
- Cell 4: Raw profile (add a markdown comment HUMAN review gate).
- Cell 5: Category profile (add markdown comment about config editing).
- Cell 6: Normalise NER pass (en + de).
- Cell 7: Normalise summarisation pass (en only).
- Cell 8: Validate both article sets. Include the hard guard that raises RuntimeError if validation fails.
- Cell 9: Preprocessing both article sets.
- Cell 10: NER English — load, run, del, release_model().
- Cell 11: NER German — load, run, del, release_model().
- Cell 12: NER analysis — entity_df, plot_top_entities for each country and language,
  plot_entity_dynamics for top 5 English entities per country, investigate_ner_errors.
  Include the entity counting code exactly as in SPEC.
- Cell 13: Summarisation — load, summarize_articles, del, release_model(), quality df display.
- Cell 14: Similarity scoring — load, score_all_articles, del, release_model().
  Include the note about the 256-token limitation.
- Cell 15: Similarity analysis — build_similarity_dataframe, plot, explain_similarity_extremes.
- Cell 16: Topic prediction — load, predict_all_topics, del, release_model(), evaluate.
- Cell 17: Empty markdown cell — placeholder for human-written summary findings.

Format: create a valid .ipynb JSON file. Each code cell has cell_type="code".
Cell 17 is cell_type="markdown".

Acceptance criterion: `uv run jupyter nbconvert --to script notebooks/analysis.ipynb --stdout` exits without error (checks JSON validity of the notebook).
```

---

## Final verification

After completing all tasks, run the full test suite:

```
uv run pytest tests/ -v
```

All tests must pass. Then verify imports:

```
uv run python -c "
import src.utils
import src.data_loader
import src.data_inspector
import src.data_normalizer
import src.preprocessing
import src.ner
import src.summarizer
import src.similarity
import src.topic_predictor
print('All modules import cleanly.')
"
```

To launch the notebook:
```
uv run jupyter notebook notebooks/analysis.ipynb
```
