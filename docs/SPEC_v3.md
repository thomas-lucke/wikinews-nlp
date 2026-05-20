# SPEC - Wikinews NLP Analysis Pipeline

**Version:** 3.0  
**Status:** Draft - ready for implementation  
**Last updated:** 2026-05-11  
**Minimum Python version:** 3.10 (required for `dict` insertion-order guarantee; `match` syntax may be used in implementation but is not mandated by this spec)  
**Changes from v2:** Targeted patch addressing three hostile reviews. See CHANGELOG.md.

---

## Design principles

These rules apply to every module in the project. They are not suggestions.

**No print statements in src/.** All output from `src/` modules goes through the Python `logging` module. `print()` is permitted only in the notebook display layer.

**No hardcoded values in src/.** Every path, model name, threshold, language code, country label, and topic label comes from the config dict passed as an argument. Constants that apply project-wide live in `src/utils.py`.

**A single bad record never crashes the pipeline.** Every per-article operation is wrapped in a try/except. Errors are logged with the article ID and processing continues.

**One model in VRAM at a time.** Models are loaded, used, and explicitly deleted before the next model is loaded. GPU cache is cleared after every deletion.

**All paths use pathlib.Path.** No string concatenation for file paths. This ensures Windows compatibility.

**No secret values in config files.** API keys and credentials come from environment variables only.

**Random behaviour is seeded.** Any function that samples, shuffles, or has non-deterministic output takes a `random_seed` argument with a default of `42`.

---

## Directory structure

```
wikinews-nlp/
├── config/
│   └── config.yaml
├── src/
│   ├── utils.py              # Logging setup, device detection, GPU cleanup, constants
│   ├── data_loader.py        # Download raw data only
│   ├── data_inspector.py     # Format detection, raw/category profiling, post-normalisation validation
│   ├── data_normalizer.py    # Field mapping, deduplication, filtering
│   ├── preprocessing.py      # Text cleaning, tokenisation, POS tagging
│   ├── ner.py                # NER inference and analysis
│   ├── summarizer.py         # Abstractive summarisation
│   ├── similarity.py         # Similarity scoring and visualisation
│   └── topic_predictor.py    # Zero-shot topic classification
├── notebooks/
│   └── analysis.ipynb        # Entry point - orchestration and display only
├── docs/
│   ├── PRD.md
│   ├── SPEC.md               # This file
│   ├── CHANGELOG.md
│   └── decisions/
│       ├── 0001-huggingface-for-ner.md
│       ├── 0002-english-only-summarization.md
│       ├── 0003-dataset-agnostic-pipeline.md
│       └── 0004-sequential-gpu-loading.md
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_data_loader.py
│   ├── test_data_inspector.py
│   ├── test_data_normalizer.py
│   ├── test_preprocessing.py
│   ├── test_ner.py
│   ├── test_summarizer.py
│   ├── test_similarity.py
│   └── test_topic_predictor.py
├── scripts/
│   └── review_spec.py        # Automated spec review via Anthropic API
├── data/
│   ├── raw/                  # Git-ignored
│   └── processed/            # Git-ignored
├── logs/                     # Git-ignored
├── .env.example              # Template showing required environment variables
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
└── README.md
```

`.gitignore` must include: `data/`, `logs/`, `.env`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`

`.env.example` content:
```
ANTHROPIC_API_KEY=your_key_here
```

---

## config.yaml - full schema

```yaml
data:
  source_url: "https://github.com/PrimerAI/WikiNews-multilingual"
  raw_path: "data/raw"
  min_article_length: 100
  # Minimum character count for the text field.
  # Applied after stripping leading/trailing whitespace (str.strip()) on the raw text
  # value, before any markup cleaning or normalisation. This is the earliest possible
  # filter - it operates on whatever the source dataset provides as the text field.
  # Cleaning (remove markup, URLs) happens in preprocessing.py and may further reduce
  # article length; this threshold does not account for that reduction.

topics:
  selected:
    - "Politics and conflicts"
    - "Science and technology"
    - "Sports"
  articles_per_topic_min: 10    # Warn if a (language, topic) pair has fewer valid articles than this
  articles_per_topic_max: 20    # Hard cap - return at most this many per (language, topic) group
                                # (or per (country, language, topic) when countries filtering is active)
  articles_per_topic_ner: 100   # NER pass only - no sample size limit in Task 3

countries:
  selected:
    - "United States"
    - "Germany"
  # countries.selected is available for explicit country-filtered runs
  # (pass to normalise_articles as countries=config["countries"]["selected"]).
  # The default notebook pipeline passes countries=None (language-only grouping):
  # country is then extracted from the categories list as best-effort metadata
  # using a known-countries vocabulary, without dropping any articles.

languages:
  ner: ["en", "de"]
  summarization: ["en"]
  # normalise_articles is called twice:
  #   once with languages.ner to load articles for NER
  #   once with languages.summarization to load articles for summarisation
  # The notebook orchestrates these two calls separately.
  # With the same random_seed and source pool, the English articles selected in both
  # passes are identical. The passes are separate processing pools, not separate samples.
  # If different English samples are required, pass random_seed + 1 to the second call.

models:
  ner_english: "dslim/bert-base-NER"
  # dslim/bert-base-NER is fine-tuned on CoNLL-2003 English.
  # Labels: PER, ORG, LOC, MISC. aggregation_strategy="simple" resolves BIO tags.

  ner_german: "Davlan/bert-base-multilingual-cased-ner-hrl"
  # Davlan/bert-base-multilingual-cased-ner-hrl is fine-tuned for NER across 10
  # languages including German. Labels: PER, ORG, LOC (no MISC).
  # This is NOT dbmdz/bert-base-german-cased, which is a base language model only
  # and cannot produce NER labels.

  spacy_english: "en_core_web_sm"
  spacy_german: "de_core_news_sm"
  # spaCy models are CPU-based. Caching both simultaneously does not consume GPU VRAM
  # and is therefore an intentional exception to the "one model in VRAM at a time" rule.

  summarization: "facebook/bart-large-cnn"
  # Fine-tuned on CNN/DailyMail. Max input: 1024 tokens. Use truncation=True in pipeline.

  similarity: "sentence-transformers/all-MiniLM-L6-v2"
  # Max input: 256 tokens. Cosine similarity can theoretically return values in
  # [-1, 1], though related article/summary pairs should usually be positive.
  # Long articles are truncated at 256 tokens - similarity scores reflect only the
  # article's opening. This limitation is documented in the notebook.

  topic_prediction: "facebook/bart-large-mnli"
  # Zero-shot classification via MNLI.

summarization:
  min_summary_length: 50     # Minimum OUTPUT summary length in tokens (not characters).
                             # Passed as min_length to the HuggingFace pipeline.
                             # Also used as short-article guard: articles whose input
                             # token count (per the model tokeniser) is below this
                             # value are skipped - BART cannot produce a valid
                             # minimum-length summary from such short input.
  max_summary_length: 200    # Maximum OUTPUT summary length in tokens (not characters).
                             # Passed as max_length to the HuggingFace pipeline.
  # Constraint: min_summary_length < max_summary_length. Validated at startup.

ner:
  chunk_size: 400       # Max characters per NER chunk. Chunks split at whitespace.
  chunk_overlap: 50     # Character overlap between consecutive chunks.
  error_score_threshold: 0.6  # Entities below this confidence are flagged as candidates

topic_prediction:
  sample_size: 30
  hypothesis_template: "This news article is about {}."
  # Template is intentionally general. Model (bart-large-mnli) is English-only.

similarity:
  threshold: 0.8
  # Scores >= threshold are considered acceptable for business use.
  # Drawn as a vertical line on the distribution plot.

logging:
  log_file: "logs/pipeline.log"
  # Relative to project root. Directory is created if it does not exist.

random_seed: 42
# Applied to all sampling operations. Set once in the notebook at startup.
```

Target Wikinews dataset alignment:
- The repository contains one large `multilingual_wikinews.jsonl` data file plus
  non-data files such as `README.md`. Format detection must ignore non-data files.
- Raw fields include `title`, `pageid`, `categories`, `lang`, `url`, `text`, `date`,
  and `type`.
- `categories` is a list of Wikinews category labels. The selected topics in
  `config.yaml` must use dataset-native category names such as
  `"Politics and conflicts"`, `"Science and technology"`, and `"Sports"`.
- Country selection also uses `categories`; selected countries must match dataset
  category labels such as `"United States"` or `"Germany"`.
- `pageid` groups multilingual articles about the same event. Store it as
  `event_id`; do not use it as the unique article `id`.

---

## Data flow

Two separate loading passes are orchestrated by the notebook. This is intentional:
NER needs English and German articles. Summarisation needs English only. Loading
both language sets for summarisation and then filtering would require holding German
articles in memory for the entire summarisation stage - roughly doubling memory use
for no benefit. Two targeted passes load only what each stage needs.

**Date format:** `date` is stored as the raw string value from the source field,
without parsing or validation. The article schema reflects this: `date` is `Optional[str]`
with no assumed format. Downstream functions that use dates (e.g. `plot_entity_dynamics`)
parse with `pd.to_datetime(errors="coerce")`, which converts unparseable or non-ISO dates
to `NaT` and filters them out. The pipeline never assumes a specific date format - it
stores whatever arrives and handles parsing failures gracefully at the point of use.

```
source_url (config)
     |
data_loader.download_dataset()
     |
Raw files on disk
     |
data_inspector.detect_format()
data_inspector.raw_profile()      ← works on raw files only, no field semantics
data_inspector.category_profile() ← counts raw Wikinews category labels
     |
[Human reviews raw/category profile, edits config.yaml if needed, reloads config]
     |
data_normalizer.normalise_articles()   ← called twice by notebook:
     |   Pass 1: languages=["en","de"], for NER articles
     |   Pass 2: languages=["en"],      for summarisation articles
     |
data_inspector.validate_normalised()  ← runs on normalised articles, checks
     |                                   field completeness, country scope, and topic coverage
     |
[Human reviews validation report - decides whether to proceed]
     |
preprocessing.preprocess_articles()
     |
     ├── ner.run_ner()  [English model]
     |      then: del model, gc.collect(), empty_cache()
     ├── ner.run_ner()  [German model]
     |      then: del model, gc.collect(), empty_cache()
     ├── ner analysis and plots
     |
     ├── summarizer.summarize_articles()  [English articles only]
     |      then: del model, gc.collect(), empty_cache()
     ├── similarity.score_all_articles()
     |      then: del model, gc.collect(), empty_cache()
     ├── similarity plots and report
     |
     └── topic_predictor.predict_all_topics()
            then: del model, gc.collect(), empty_cache()
```

---

## Secrets and environment variables

The `review_spec.py` script requires an Anthropic API key. It must never be in `config.yaml` or committed to the repo.

```python
# In scripts/review_spec.py only:
import os
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY environment variable is not set. "
        "Copy .env.example to .env and add your key."
    )
```

All other modules in `src/` require no API keys and have no secret dependencies.

---

## Article schema

Fields are added at each stage. Optional fields are marked `Optional[type]`.
Fields present on all articles after normalisation are marked (guaranteed).

```python
{
    # Guaranteed after normalise_articles()
    "id":           str,            # Source id/article_id/uid if present, else sha256(text)[:16].
                                    # This is the only ID field in the article dict - there is no
                                    # separate source_id field. All references to "source id field"
                                    # elsewhere in this spec refer to this "id" key.
    "event_id":     Optional[str],  # Cross-language event/group id if present. For Wikinews this is
                                    # pageid: the same pageid links English and non-English articles
                                    # about the same event. Not required to be unique per article.
    "title":        str,            # Empty string "" if not in source - never None
    "text":         str,            # Raw article text as extracted from source
    "language":     str,            # ISO 639-1 lowercase: "en" or "de"
    "topic":        str,            # Category label, lowercased and stripped
    "country":      str,            # Country category, lowercased and stripped
    "date":         Optional[str],  # Raw source date string as-is, or None if missing.
                                    # Not parsed or validated to ISO format - may be any string
                                    # the source provides. Downstream functions parse with
                                    # pd.to_datetime(errors="coerce") and handle non-ISO values.

    # Added by preprocessing.preprocess_articles()
    "cleaned_text": str,            # Text after markup removal and normalisation
    "sentences":    list[str],      # Sentence strings, from spaCy sentence splitter
    "tokens":       list[str],      # Word tokens (excludes whitespace tokens)
    "pos_tags":     list[tuple[str, str]],  # [(token, universal_pos_tag), ...]

    # Added by ner.run_ner() - present only if NER was run for this article's language
    "entities":     Optional[list[dict]],   # See entity schema below. Empty list if NER
                                            # ran but found nothing. None if NER not run.

    # Added by summarizer.summarize_articles() - English articles only.
    # Non-qualifying articles (e.g. German) do NOT receive this key.
    # Use article.get("summary") rather than article["summary"] in all downstream code.
    "summary":      Optional[str],  # None if summarisation failed or short-article guard triggered

    # Added by similarity.score_all_articles() - only if summary is not None
    "similarity_score": Optional[float],  # Cosine similarity, theoretically in [-1, 1]

    # Added by topic_predictor.predict_all_topics() to copied sampled articles only (~30).
    # The original article list is not mutated and unsampled originals do not receive this key.
    "predicted_topic": Optional[str],
}
```

Entity schema:
```python
{
    "text":    str,    # Entity string as it appears in cleaned_text
    "label":   str,    # Normalised label - one of: "PER", "ORG", "LOC", "MISC"
                       # German model does not produce MISC; those entities are absent.
    "start":   int,    # Start char index in cleaned_text, inclusive
    "end":     int,    # End char index in cleaned_text, exclusive (Python slice convention)
    "score":   float,  # Model confidence, 0.0–1.0
}
```

---

## Logging specification

All `src/` modules use Python's built-in `logging` module. Setup is centralised in `utils.py`
and called once at the top of the notebook. Modules never call `logging.basicConfig()` directly.

```python
# In src/utils.py - see utils.py module specification for full implementation.
# Every src/ module declares its logger at module level:
import logging
logger = logging.getLogger(__name__)
```

Every `src/` module declares its logger at module level:
```python
import logging
logger = logging.getLogger(__name__)
```

Log level conventions:
- `logger.info()` - normal progress ("Loaded 342 articles", "NER complete: 18 entities found")
- `logger.warning()` - recoverable problems ("Article abc123 has no date field")
- `logger.error()` - article-level failures ("NER failed on article abc123: {error}")
- `logger.critical()` - pipeline-level failures that require stopping

---

## Error handling philosophy

**Per-article errors never crash the pipeline.** Each stage defines whether a failed
article is skipped, left unchanged, or marked with a `None`/empty result. The common
pattern is:

```python
results = []
for article in articles:
    try:
        result = process_one(article)
        results.append(result)
    except Exception as e:
        logger.error(
            "Processing failed for article %s (%s): %s",
            article.get("id", "unknown"),
            article.get("title", "no title")[:60],
            str(e)
        )
        # stage-specific fallback: skip this output, leave article unchanged,
        # or set the stage field to None/[] as documented by the module
```

**Pipeline-level errors stop execution.** If a model cannot be loaded, a required
file is missing, or config validation fails - raise immediately with a clear message.
Do not attempt to continue.

The distinction between pipeline-level and per-article errors:
- **Pipeline-level (raise and stop):** model load failure, missing config file,
  config validation failure (`validate_summarization_config`, `validate_ner_config`),
  `detect_format` returning `"unknown"`, `load_raw_records` raising `ValueError`,
  empty language/topic/country selection passed to `normalise_articles`.
  These affect all subsequent processing - there is no point continuing.
- **Per-article (log and continue):** NER inference failing on one article,
  preprocessing failing on one article, summarisation returning None, a single
  malformed entity span. These affect one record - the pipeline can continue with
  the rest. Module docstrings define the fallback value for each stage.

**No bare `except:` clauses.** Always catch `Exception` at minimum. Log the full exception string.

---

## GPU management

```python
# In src/utils.py

import gc
import logging
import torch

logger = logging.getLogger(__name__)

def get_device() -> int:
    """
    Return 0 (first GPU) if CUDA is available, -1 (CPU) otherwise.
    HuggingFace pipelines accept -1 for CPU.
    Single-GPU assumption: this project targets one GTX 1060.
    Multi-GPU and Apple MPS support are out of scope.
    """
    return 0 if torch.cuda.is_available() else -1

def release_model() -> None:
    """
    Clear GPU cache after a model has been deleted by the caller.

    IMPORTANT: the caller must delete their own variable reference BEFORE
    calling this function. del inside a function cannot free the caller's
    reference - only the local parameter binding is deleted.

    Correct pattern in notebook:
        del pipeline_variable       # removes caller's reference
        release_model()             # clears GPU cache

    Incorrect pattern (model stays in VRAM):
        release_model(pipeline_variable)   # only deletes local param copy
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU cache cleared.")
```

`torch.cuda.empty_cache()` alone does not free Python-held tensors. The sequence
`del` → `gc.collect()` → `empty_cache()` is required to reliably free VRAM.

---

## Windows compatibility

All path construction uses `pathlib.Path`. String paths from config are converted
on first use. Example pattern used throughout:

```python
from pathlib import Path

raw_path = Path(config["data"]["raw_path"])
raw_path.mkdir(parents=True, exist_ok=True)
```

Never use `os.path.join()` or string concatenation for paths. Never hardcode `/` or `\`.

**File encoding:** always specify `encoding="utf-8"` explicitly on all `open()` and
`Path.read_text()` calls. Windows systems default to the system code page (e.g.
cp1252) rather than UTF-8, which will corrupt or raise on multilingual text including
German umlauts (ä, ö, ü, ß). Never rely on the platform default encoding.

---

## Module specifications

---

### src/utils.py

Contains setup functions and constants shared across modules. Not a catch-all.

```python
import gc
import logging
import sys
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

RANDOM_SEED: int = 42  # Default seed. Notebook may override from config.

def setup_logging(log_file: str = "logs/pipeline.log", level: int = logging.INFO) -> None:
    """
    Configure logging for the entire pipeline. Call once at notebook startup.
    Creates the log directory if it does not exist.
    Writes to both stdout and the log file specified.
    Format: "YYYY-MM-DD HH:MM:SS | LEVEL     | module_name | message"

    Clears any existing handlers on the root logger before adding new ones.
    This prevents duplicate log output when the notebook cell is re-run in Jupyter,
    where logging.basicConfig() is a no-op if handlers already exist.

    Args:
        log_file: Path to log file. Directory is created if missing.
        level: Logging level (logging.INFO by default).
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    # Remove existing handlers to prevent duplicate output on notebook re-run
    root_logger.handlers.clear()

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger.setLevel(level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
```

---

### src/data_loader.py

Single responsibility: get raw files onto disk.

```python
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def download_dataset(source_url: str, raw_path: str) -> Path:
    """
    Download or clone the dataset from source_url into raw_path.

    Behaviour by URL type:
    - GitHub URL (contains "github.com"): use git clone via subprocess.
      Before cloning, wipe the direct children of raw_path - git clone refuses
      to write into a non-empty directory. This wipe only runs after the
      skip-condition check has already returned False, so any existing files
      were not usable data; pre-existing non-data files (user notes, README
      placeholders) will be removed alongside any partial download.
      Fall back to ZIP download in BOTH of these cases:
        a) git is not available on PATH, AND
        b) git is available on PATH but `git clone` returns a non-zero exit
           code (transient network failure, suppressed auth prompt, proxy).
      The ZIP URL is constructed by trying these branch names in order:
      main, master. If neither returns HTTP 200, raise RuntimeError with
      instructions to set source_url to a direct archive URL instead.
    - Direct file URL (ends in .zip, .tar.gz, .gz, .csv, .json, .jsonl):
      download with requests (timeout=60, stream=True) and extract if compressed.
    - Other URLs: download with requests (timeout=60) and save as-is.

    Skip condition: raw_path already exists AND contains at least 1 file with
    a recognised data extension (.json, .jsonl, .csv, .tsv, .txt) AND total
    size of recognised data files is > 100KB. Both conditions must be true.
    "Contains" is RECURSIVE - scan with Path.rglob("*"), not a flat
    Path.iterdir(). A freshly-cloned GitHub repo typically nests data under
    subdirectories (e.g. data/articles.jsonl); a non-recursive check would
    never trip the skip on a previously cloned dataset. "100KB" means 102400
    bytes (binary KB, 100 * 1024). The threshold is heuristic, not exact.
    This is a heuristic, not a guarantee. A partially downloaded dataset that
    meets these thresholds will be treated as complete and skip the download.
    This threshold fits the target Wikinews repo, whose actual data is one large
    JSONL file plus a README.
    If the pipeline behaves unexpectedly after a previously interrupted download,
    delete raw_path and rerun this function. See Known Limitations in SPEC.md.

    Archive extraction:
      .zip → zipfile.ZipFile.
      .tar.gz / .tgz → tarfile.
      .gz (bare, non-tar) → gzip stdlib. Decompress the single compressed
        stream to a file at raw_path / archive_path.stem. Bare .gz is NOT an
        archive - there is no directory structure - so the single-top-dir
        stripping logic below does not apply to this case.
    Many archives contain a single top-level folder (e.g. reponame-main/).
    If the archive contains exactly one top-level directory and all files are
    inside it, strip that directory and place files directly in raw_path.
    If the archive contains multiple top-level entries, extract as-is.

    Retry: one retry on network failure, with a 5-second delay between attempts.
    Failure cases that trigger retry: requests.Timeout, requests.ConnectionError.
    Failure cases that do NOT retry: HTTP 4xx (client error), extraction failure.
    HTTP response handling for non-200 statuses:
      - 3xx is followed transparently (requests follows redirects by default).
      - 4xx raises RuntimeError immediately (no retry).
      - 5xx raises RuntimeError immediately (no retry).
    Retry is reserved strictly for Timeout / ConnectionError above.

    Args:
        source_url: URL to download from.
        raw_path: Local directory to save into. Created if it does not exist.

    Returns:
        Absolute Path to raw_path.

    Raises:
        RuntimeError: If download fails after one retry, extraction fails,
                      GitHub ZIP fallback cannot find a valid branch, or git
                      is available but both `git clone` and the ZIP fallback
                      failed.
        EnvironmentError: If git is not available on PATH and the ZIP
                          fallback also fails.
    """
```

---

### src/data_inspector.py

Three inspection/validation functions are used at different notebook points:
raw_profile() and category_profile() run before normalisation; validate_normalised()
runs after normalisation.

```python
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RawProfile:
    """Results of pass 1 - structural analysis of raw files before any field mapping."""
    total_records: int
    detected_format: str          # "json" | "jsonl" | "csv" | "tsv" | "directory-of-txt" | "zip" | "unknown"
    detected_fields: list[str]    # Union of all field names seen across records (for JSON/CSV)
                                  # For directory-of-txt: field names inferred from filenames or empty list
    file_count: int               # Number of files found
    total_size_bytes: int         # Total raw data size
    sample_record: dict           # First valid record, for human inspection in notebook


@dataclass
class NormalisedValidation:
    """Results of pass 2 - quality checks on the normalised article list."""
    total_articles: int
    languages_found: dict[str, int]   # {"en": 280, "de": 62}
    countries_found: dict[str, int]   # {"united states": 120, "germany": 80}
    topics_found: dict[str, int]      # {"politics and conflicts": 40, "sports": 35}
    country_topic_counts: dict[tuple[str, str], int]  # {("germany", "sports"): 12}
    missing_date_count: int
    missing_title_count: int          # articles where title == ""
    very_short_article_count: int     # len(text) < min_article_length * 5
                                      # Note: normalise_articles already drops articles
                                      # below min_article_length. This field counts
                                      # articles that passed the minimum but are still
                                      # very short (under 5x the minimum), which may
                                      # produce poor summaries or sparse NER output.
                                      # The multiplier 5 is chosen to flag articles that
                                      # are technically above the drop threshold but
                                      # substantially shorter than a useful news article
                                      # (e.g. min_article_length=100 → flag < 500 chars,
                                      # roughly 80–100 words). Adjust the multiplier in
                                      # validate_normalised if a different project uses
                                      # shorter or longer average article lengths.
    country_topics_below_minimum: list[tuple[str, str]]  # (country, topic) pairs below articles_per_topic_min
    topics_missing_from_config: list[str]  # config topics not found in loaded articles
    countries_missing_from_config: list[str]  # config countries not found in loaded articles
    validation_passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def detect_format(raw_path: str) -> str:
    """
    Inspect raw_path and return a format string without loading the full dataset.

    If raw_path does not exist on disk, return "unknown". detect_format is
    intentionally non-failing so the calling notebook can log informatively;
    load_raw_records still raises FileNotFoundError for the same input per its
    own spec.

    Detection order:
    1. If raw_path is a single file: check extension.
       .json → "json", .jsonl → "jsonl", .csv → "csv", .tsv → "tsv".
       For archive detection, test name.lower().endswith(".tar.gz") FIRST,
       then fall back to suffix-based matching for .zip / .gz / .tgz. All four
       map to "zip", but the explicit .tar.gz check is required because
       Path("foo.tar.gz").suffix == ".gz" - a naive suffix-only check would
       conflate .tar.gz with bare .gz files. In normal pipeline use,
       download_dataset extracts archives before detect_format is called, so
       "zip" should not appear in standard runs. "zip" is returned only if the
       caller passes a path to an un-extracted archive.
    2. If raw_path is a directory:
       a. Collect extensions of direct-child files (non-recursive), but compute
          dominance using recognised data files only: .txt, .json, .jsonl, .csv, .tsv.
          Ignore README.md, license files, and other non-data files for dominance.
       b. Check if at least 80% of recognised data files share one extension
          (count / total >= 0.8 - exactly 80% counts as dominant). Apply checks in this order:
           - If dominant extension is .txt: return "directory-of-txt".
           - If dominant extension is .json: return "json".
           - If dominant extension is .jsonl: return "jsonl".
           - If dominant extension is .csv: return "csv".
           - If dominant extension is .tsv: return "tsv".
       c. If there are no recognised data files, no extension reaches 80% dominance, or the dominant extension is
           unrecognised: return "unknown".
        Minority data files and non-data files are ignored by detect_format.
        load_raw_records handles only the dominant format and skips files with other
        extensions - log a warning for each skipped recognised data file.
    3. Attempt to parse one record from the detected format to confirm.
       If parsing fails, return "unknown".

    Args:
        raw_path: Path to raw data file or directory.

    Returns:
        Format string.
    """


def raw_profile(raw_path: str, detected_format: str) -> RawProfile:
    """
    Count records, collect field names, and retrieve one sample record.
    Does not decode, clean, or interpret field values.

    For JSON/JSONL/CSV/TSV: detected_fields is the union of keys across all records.
    For directory-of-txt: detected_fields is empty list (no structured fields).
    total_records counts parseable records only; unparseable records are logged as warnings.

    Args:
        raw_path: Path to raw data.
        detected_format: String from detect_format().

    Returns:
        Populated RawProfile instance.
    """


def category_profile(raw_path: str, detected_format: str) -> pd.DataFrame:
    """
    Count raw category labels before normalisation so the user can choose
    countries and topics in config.yaml.

    This function is intentionally exploratory and non-interactive. It does not
    mutate config and does not prompt the user. The notebook displays its output;
    the user edits config.yaml manually and reloads the config before proceeding.

    Wikinews stores both topical categories (e.g. "Sports") and country categories
    (e.g. "Germany") in the same raw "categories" field. This function does not
    try to classify a category as a topic or country. It simply reports all category
    labels and counts so the user can pick viable values for both
    config["topics"]["selected"] and config["countries"]["selected"].

    For each parseable raw record:
    - If "categories" is a list, count each string item once for that record.
    - If "categories" is a string, count it as one category.
    - If "categories" is missing, null, or not a list/string, increment
      missing_categories_count internally and continue.
    - If "categories" is present but yields zero usable labels after whitespace-
      stripping and deduplication (e.g. categories="", "   ", [""], [], or
      list of whitespace-only strings), also increment missing_categories_count.
      Rationale: this matches the docstring's user-facing meaning ("records with
      no usable categories"). Spec earlier enumerated only the missing/null/
      wrong-type cases; this rule extends to records that survive the type check
      but contribute nothing once stripped.
    - Deduplicate repeated category strings within one record before counting so
      a malformed record cannot inflate one category.
    - "categories field exists anywhere" (the condition that decides whether
      category_profile returns the special empty-DataFrame branch) is judged by
      key presence, not value validity. A dataset where every record carries the
      key categories=None still returns a populated-shape DataFrame (zero rows,
      missing_categories_count == total_records), not the empty-DF branch.

    Return a DataFrame sorted by count descending, then category ascending, with
    columns:
        category (str)       - raw category label stripped of surrounding whitespace
        count (int)          - number of records containing that category
        percent (float)      - count / total parseable records * 100

    Attach missing category information in df.attrs:
        df.attrs["total_records"] = total parseable records
        df.attrs["missing_categories_count"] = records with no usable categories

    For non-Wikinews datasets with no "categories" field, the returned DataFrame
    is empty with the correct columns and attrs populated. This is not a pipeline
    failure; normalise_articles may still work if the dataset has dedicated topic
    and country fields.

    Args:
        raw_path: Path to raw data.
        detected_format: String from detect_format().

    Returns:
        pd.DataFrame with category counts.
    """


def validate_normalised(
    articles: list[dict],
    config: dict,
) -> NormalisedValidation:
    """
    Run quality checks on the normalised article list. Called after normalise_articles().

    Topic casing: when comparing topics from config["topics"]["selected"] to
    article["topic"] values or topics_found keys, normalise both sides with
    _normalise_topic_string (lowercase + strip) before comparing. Article topics
    are already lowercase after normalise_articles; config topics may be mixed-case.

    Country casing: apply the same lowercase + strip normalisation when comparing
    config["countries"]["selected"] to article["country"] values or countries_found
    keys. Article countries are already lowercase after normalise_articles.

    Validation rules:
        Errors (set validation_passed = False):
            - total_articles == 0
            - All selected topics are missing from topics_found
            - All selected countries are missing from countries_found
              Guard the "all missing" checks with `if selected_topics_raw and ...`
              (likewise for countries): empty config-selected lists do NOT trip
              this error, because `len([]) == len([])` would otherwise return
              True trivially. In normal pipeline flow `normalise_articles`
              raises ValueError earlier for empty config lists; this guard
              prevents misleading errors in ad-hoc use.

        Warnings (do not fail validation):
            - missing_date_count > 5% of total_articles
            - very_short_article_count > 10% of total_articles
            - Any (country, topic) in country_topics_below_minimum
            - Any topic in topics_missing_from_config
            - Any country in countries_missing_from_config
            - Any selected (country, topic) pair has fewer than articles_per_topic_min
              articles across all selected languages for that article set
              (record in country_topics_below_minimum)

    country_topics_below_minimum is built by iterating the FULL CROSS-PRODUCT of
    selected_countries × selected_topics (lowercased to match country_topic_counts
    keys), flagging every pair whose count is below articles_per_topic_min.
    Pairs that never appear in articles (count == 0) are included - they are
    below any reasonable minimum and would otherwise be silently missed by
    iterating only the observed pairs.

    topics_missing_from_config and countries_missing_from_config preserve the
    ORIGINAL config casing in the stored strings, even though the comparison
    that decides membership is lowercase+strip on both sides. Rationale: the
    user should see their own config values back (e.g. "Politics and conflicts"
    rather than "politics and conflicts").

    very_short_article_count: count of articles where len(text) < config["data"]["min_article_length"] * 5.
    This is distinct from the min_article_length filter already applied in normalise_articles.
    It flags articles that passed normalisation but may still be too short for quality summarisation
    or NER output.

    Args:
        articles: Normalised article list from normalise_articles().
        config: Full config dict.

    Returns:
        Populated NormalisedValidation instance.
    """


def print_raw_profile(profile: RawProfile) -> None:
    """
    Print a structured summary of the RawProfile using logger.info().
    Include: format, file count, total size, record count, field names, sample record keys.
    Never print the full sample record text - truncate text fields to 100 chars.
    """


def print_category_profile(df: pd.DataFrame, top_n: int = 50) -> None:
    """
    Print the category_profile table using logger.info().

    Show:
    - total parseable records from df.attrs["total_records"]
    - records with no usable categories from df.attrs["missing_categories_count"]
    - top_n rows with category, count, and percent

    End with a clear notebook-facing note:
        "Review these category labels, update config.yaml topics.selected and
        countries.selected if needed, then rerun Cell 1 or reload config before
        continuing."

    Do not prompt for input and do not modify config.yaml.
    """


def print_validation_report(report: NormalisedValidation) -> None:
    """
    Log a concise, human-scannable validation report.

    Headline stats via logger.info(): total articles, languages, topics, a
    one-line country summary (count of distinct + top 3 by frequency), and a
    one-line data-quality summary (missing dates / titles / very short).

    Actionable problems are surfaced only via the stored warnings
    (logger.warning) and errors (logger.error) - the report does NOT also log
    INFO lines for below-minimum pairs or missing-from-config items, since the
    warnings already enumerate them (avoid duplication).

    The full countries_found dict and country_topic_counts dict are NOT logged
    line by line - they remain on the NormalisedValidation object for
    programmatic access but are too verbose for a scannable report.

    End with "Validation passed." or "Validation FAILED - review errors above."
    No ANSI colour codes (unreliable in Jupyter environments).
    """
```

---

### src/data_normalizer.py

Maps raw field names to internal schema, deduplicates, filters, and logs dropped records.

```python
import hashlib
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Maps raw field names to internal field names.
# Precedence rule: iterate FIELD_MAPPINGS in insertion order; for each key,
# check if it exists in the raw record. The first FIELD_MAPPINGS key that is
# present in the raw record wins for that internal field.
# Example: a raw record with both "text" and "body" yields internal "text"
# from the "text" key, because "text" appears before "body" in this dict.
# Do NOT iterate raw record keys and look them up in FIELD_MAPPINGS - that would
# make the result depend on raw record key insertion order, which varies by source.
FIELD_MAPPINGS: dict[str, str] = {
    "text":         "text",
    "body":         "text",
    "article_body": "text",
    "content":      "text",
    "article_text": "text",
    "title":        "title",
    "headline":     "title",
    "article_title":"title",
    "date":         "date",
    "published":    "date",
    "publish_date": "date",
    "created_at":   "date",
    "timestamp":    "date",
    "language":     "language",
    "lang":         "language",
    "locale":       "language",
    "country":      "country",
    "country_name": "country",
    "location":     "country",
    "topic":        "topic",
    "category":     "topic",
    "section":      "topic",
    "label":        "topic",
    "categories":   "categories",
    # Wikinews stores both topical labels and country labels in "categories" as
    # list[str]. Keep the raw list in a temporary working field named "categories";
    # normalise_articles resolves article["topic"] from config["topics"]["selected"]
    # and article["country"] from config["countries"]["selected"]. Use config order,
    # not source list order, so selection is deterministic and controlled by scope.
    # Note: "tags" is intentionally excluded. Tags are often loose keywords rather
    # than the category taxonomy used for this assignment.
    "id":           "id",
    "article_id":   "id",
    "uid":          "id",
    "pageid":       "event_id",
    # Wikinews pageid groups multilingual articles about the same event. It is not
    # a unique article id and must not be used as article["id"].
    # "url" is NOT mapped to "id". URLs are long and structurally different from IDs.
    # A URL is preserved as-is or discarded; it does not replace the ID field.
}


@dataclass
class DroppedRecord:
    article_index: int   # Zero-based index of this record in the list returned by
                         # load_raw_records() - i.e., its position in the original raw
                         # input before any filtering or deduplication. This index is
                         # stable across the processing steps: it does not shift as
                         # other records are dropped.
    reason: str          # One of: "text_too_short" | "language_not_in_config" |
                         #         "topic_not_in_config" | "country_not_in_config" |
                         #         "duplicate" | "no_text_field"
    field_values: dict   # Subset of raw record fields for debugging (id, title, language, topic, country)
                         # Never includes the full text field to avoid memory bloat.


def _generate_stable_id(text: str) -> str:
    """Return first 16 hex chars of sha256 of text. Stable across reruns."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _normalise_topic_string(s: str) -> str:
    """Lowercase and strip whitespace. Used for case-insensitive topic matching."""
    return s.lower().strip()


def _select_from_categories(categories: list[str], selected_labels: list[str]) -> Optional[str]:
    """
    Select the first configured label that appears in a raw Wikinews categories list.

    Used for both topic selection and country selection. Comparison normalises both
    sides with _normalise_topic_string. Return the configured label lowercased and
    stripped, not the raw source spelling.

    Args:
        categories: Raw Wikinews category strings.
        selected_labels: Config labels in priority order.

    Returns:
        Normalised selected label, or None if no configured label appears.
    """


def _infer_text_field(raw_record: dict, min_article_length: int) -> Optional[str]:
    """
    Find the most likely text field in a raw record when no FIELD_MAPPINGS key
    matches the raw record for the "text" slot.

    Returns the FIELD NAME (not the value) of the best candidate, so the caller
    can extract the value with raw_record[returned_name].

    Candidate rule: a field is a candidate if ALL of the following are true:
        1. Its key is NOT already in FIELD_MAPPINGS (avoids re-selecting a field
           that was already considered and didn't match, e.g. "body" is in
           FIELD_MAPPINGS so it is always excluded even if it contains long text).
        2. Its value is a str (not a list, dict, int, etc.).
        3. len(value.strip()) >= min_article_length.

    Exclusion note: FIELD_MAPPINGS contains all standard field names. Any key in
    FIELD_MAPPINGS is excluded from inference regardless of its value, to prevent
    fields like "url" (which can be long strings) or "label" from being mistaken
    for body text if they happen to be long enough.

    If multiple candidates remain, return the name of the field with the longest
    stripped value. If none found, return None - do not guess.

    Args:
        raw_record: A single raw record dict.
        min_article_length: Minimum character length from config.

    Returns:
        Field name string (key in raw_record), or None if no candidate found.
    """


def load_raw_records(raw_path: str, detected_format: str) -> list[dict]:
    """
    Parse raw files into a list of dicts. Shared by both data_inspector and
    data_normalizer to avoid duplicate loading logic.

    If raw_path is a directory for JSON/JSONL/CSV/TSV formats, read only direct-child
    files with the detected extension, sorted by filename using Python's default
    Unicode code-point sort. Skip non-data files such as README.md silently and log
    a warning for recognised data files with non-dominant extensions.

    For JSON: load the top-level object.
              - If it is a list, return as-is.
              - If it is a dict with EXACTLY ONE key containing a list, return
                that list.
              - If it is a dict with MULTIPLE keys where two or more are list-
                valued: concatenate all list-valued keys' contents (filtered to
                dicts only) into a single output list. Log a warning per key
                reporting how many non-dict entries were skipped. Rationale:
                this preserves the most data and matches the spirit of the
                single-list-key branch above.
              - If it is a dict with multiple keys where none is a list: wrap
                in a list and return as a single-record dataset. Log a warning
                that the JSON structure was ambiguous and was treated as a
                single record - the human reviewer should confirm this is
                correct in the raw profile.
              - If a key contains a list of non-dict items (e.g. list of
                strings): skip that key and continue searching; log a warning.
              - If the top-level value is neither a list nor a dict (e.g. a
                string, number, bool, or null at top level): log a warning and
                return an empty list. The same observable shape as
                "unparseable"; the downstream pipeline fails at
                validate_normalised with total_articles == 0, surfacing the
                issue clearly.
              Open with encoding="utf-8". If the file is not valid UTF-8, let the
              UnicodeDecodeError propagate - do not silently replace characters in
              structured formats where replacement would corrupt field values.
    For JSONL: one dict per line, sorted by line number. Skip blank lines and
               unparseable lines (log warning with line number).
               Open with encoding="utf-8" (same rationale as JSON above).
    For CSV/TSV: use csv.DictReader. All values are strings.
                 Row order is preserved (matches file line order).
                 Open with encoding="utf-8". Same rationale: silent replacement
                 in structured data would corrupt field values.
    For directory-of-txt: each .txt file becomes one dict with key "text" = file contents.
                          "id" = filename stem. "language", "topic", "date" = None.
                          Files are read with encoding="utf-8", errors="replace".
                          Replacement is acceptable here because the format has no
                          structured fields - a replaced character degrades one article
                          rather than corrupting schema parsing.
                          Files are sorted alphabetically by filename stem using
                          Python's default Unicode code-point sort (str.sort / sorted()).
                          This is locale-independent and produces consistent results
                          across platforms. Do not use locale.strcoll or any
                          locale-aware sort - results would vary by system locale,
                          breaking cross-machine determinism.
                          This ensures deterministic ordering regardless of filesystem
                          traversal behaviour.
    For unknown: raise ValueError with a message explaining the problem.

    All formats: the returned list is always sorted by the record's stable key
    before returning:
        - JSON/JSONL/CSV/TSV: sorted by row/line index (already stable from read order)
        - directory-of-txt: sorted by filename stem alphabetically

    This sorting ensures that random.Random(seed).sample() produces identical
    results across machines and runs with the same seed.

    Args:
        raw_path: Path to raw data.
        detected_format: Format string from detect_format().

    Returns:
        List of raw record dicts in deterministic order.

    Raises:
        ValueError: If detected_format is "unknown".
        FileNotFoundError: If raw_path does not exist.
    """


def normalise_articles(
    raw_path: str,
    detected_format: str,
    languages: list[str],
    topics: list[str],
    countries: Optional[list[str]],
    max_per_topic: int,
    min_article_length: int,
    random_seed: int = 42,
) -> tuple[list[dict], list[DroppedRecord]]:
    """
    Load raw records, map fields to internal schema, filter, deduplicate, and sample.

    Pipeline-level validation: raise ValueError if languages or topics is empty.
    If countries is provided (not None), raise ValueError if it is empty — an empty
    list would silently drop every record. Passing None disables country filtering
    entirely; country is then extracted as best-effort metadata from categories.

    RNG: create ONE stateful object at the start of the function:
        rng = random.Random(random_seed)
    Use this single instance for all rng.sample() calls. Do NOT call
    random.Random(random_seed) again inside the function - re-instantiating
    resets state and can produce duplicate samples across passes.

    Processing order:
    1. Load via load_raw_records(). Records arrive in deterministic order (sorted
       by load_raw_records per format). No re-sorting needed here.
       Defensive guard at the start of the per-record loop: if a record is not
       a dict (isinstance(raw, dict) is False), skip it silently without
       logging a DroppedRecord. load_raw_records already filters non-dicts for
       JSON/JSONL, but the guard protects against unusual inputs. A non-dict
       record has no extractable fields, so a DroppedRecord with an empty
       field_values payload would be noise.
    2. For each record: apply FIELD_MAPPINGS. For unmapped fields, try _infer_text_field().
       Log which fields were inferred (logger.info).
    3. Drop records (and log to DroppedRecord) if:
       - No text field found or found text is shorter than min_article_length after strip()
       - Language field missing (no FIELD_MAPPINGS key present in the raw record for the
         "language" slot) → reason: "language_not_in_config". No language inference is
         attempted - missing language means drop immediately.
       - Language present but not in languages list → reason: "language_not_in_config"
         Language comparison is exact-match on lowercase ISO 639-1 codes (e.g. "en", "de").
         The raw language value is lowercased and stripped before comparison so a source
         that emits "EN" or " en " matches config ["en"]. Source datasets that use
         regional variants ("en-US", "de-DE") or full names ("English") will not match
         (lowercasing alone does not collapse "en-us" to "en") and all such records will
         be dropped. If the source uses a non-standard code, add a normalisation step
         before this filter or adjust the config languages list to match the actual
         codes in the data.
       - Topic field missing or no configured topic matches the source topic value
         → reason: "topic_not_in_config".
         Source-type precedence is STRICT, not fallback:
           * Wikinews path: if the raw record has a "categories" list, resolve topic
             from it via _select_from_categories(categories, topics). If this returns
             None, drop with "topic_not_in_config" - DO NOT fall back to a string
             topic/category field even if one is also present. Two sources of truth
             in one record would make precedence ambiguous.
           * Non-Wikinews path: only used when no "categories" list is present in
             the raw record. Compare the string topic/category field to config topics
             using _normalise_topic_string on both sides.
       - (Only when countries is not None) Country field missing or no configured
         country matches the source country signal → reason: "country_not_in_config".
         Same strict-precedence rule as topic: Wikinews categories list wins outright
         if present. For Wikinews, country is resolved from the raw categories list
         using _select_from_categories(categories, countries). For other sources with
         a dedicated country field, compare it to config countries using
         _normalise_topic_string on both sides.
         When countries is None: skip this filter entirely. Instead extract country
         as best-effort metadata from the categories list using the known-countries
         vocabulary (_KNOWN_COUNTRIES, generated from pycountry/ISO 3166 plus a
         small _COUNTRY_ALIASES set — see FIX-8). Matching is exact on the
         normalised string. If no recognised country is found, store "".
         This never causes a drop — it is metadata enrichment only.
    4. Deduplicate by text content only: hash the mapped article["text"] value (the
       internal field after FIELD_MAPPINGS has been applied) using _generate_stable_id.
       If the hash has been seen before in this run, drop with reason "duplicate".
       Rationale: text-content deduplication catches reprinted articles regardless of
       whether they have the same source ID. Source IDs may or may not be unique
       depending on the dataset - no assumption is made about source ID uniqueness.
       Two records with identical source IDs but different text are treated as distinct
       articles (they may be updates or corrections to the same story).
       Build hash set in O(n) - do not sort or reorder.
    5. Group deduped articles for sampling. When countries is None, group key is
       (language, topic) — country is metadata and must not fragment the pool. When
       countries is provided, group key is (country, language, topic).
       For each group, if count > max_per_topic: sort the group
       before sampling for determinism. The sort key is decided UNIFORMLY PER GROUP
       (not per article) to avoid mixing two key types within one sort:
         - If EVERY article in the group has a source id (i.e. a FIELD_MAPPINGS key
           that maps to "id" was present in the raw record): sort by
           str(article["id"]). Cast to str - source ids may be integers in some
           datasets and strings in others; str() ensures consistent lexicographic
           comparison regardless of type.
         - If ANY article in the group lacks a source id: sort the whole group
           by current list position (post-deduplication order). Mixed sort keys
           within one sort would produce arbitrary ordering, so the fallback
           applies uniformly to the whole group rather than per-article.
       Note: this fallback achieves determinism across repeated runs on identical
       input data, because the same records will be in the same order after
       deduplication. It does not guarantee cross-machine determinism for datasets
       with no source id field if the underlying file loading order differs - for
       those datasets, determinism relies on load_raw_records sorting by filename
       stem (directory-of-txt) or row order (JSON/CSV), which is already specified.
       After sorting, sample max_per_topic records using rng.sample() where rng is
       the stateful Random object created at the start of normalise_articles.
    6. Set missing optional fields: date=None, title="" if absent, event_id=None if absent.
        Title coercion: if the mapped "title" value exists but is not a str
        (e.g. None, integer, list), STORE "" instead. The schema requires
        title: str - never None and never a non-string type. The same rule
        covers both "missing" and "wrong-type" - both yield empty string.
        ID assignment: during FIELD_MAPPINGS application in step 2, track whether any
        raw key that maps to "id" (i.e. "id", "article_id", "uid") was present in the
        raw record. Store a boolean flag alongside the working dict (e.g. as a local
       variable `has_source_id`). At step 6:
         - If has_source_id is True: the "id" field in the working dict already holds
           the source value. If that value is not a str (some sources store ids as
           integers, e.g. article_id: 42), cast it with str() - this preserves the
           source value while satisfying the schema's str type.
         - If has_source_id is False: generate id via _generate_stable_id(article["text"]).
       The stable hash is also used for deduplication (step 4) independently of the
       final id assignment - these are two separate operations.
       Event ID assignment: if a raw key maps to "event_id" (Wikinews "pageid"), store
       it as str(raw_value). This field groups related multilingual articles but is
       not used for deduplication or article identity.
    7. Normalise topic and country: store lowercased stripped strings.
       Note: original capitalisation is permanently lost after this step. candidate_labels
       passed to topic_predictor must come from config.topics.selected (original casing),
       not from article["topic"] (lowercased).

    Pass min_article_length to _infer_text_field so inference adapts to the
    configured threshold rather than using a hardcoded assumption about text length.

    Args:
        raw_path: Path to raw data.
        detected_format: Format string.
        languages: Keep only articles with these language codes.
        topics: Keep only articles whose topic normalises to one of these (after lowercasing).
        countries: If a list, keep only articles whose country matches; if None,
            skip country filtering and extract country as metadata instead.
        max_per_topic: Maximum articles per (language, topic) group when countries
            is None; per (country, language, topic) group when countries is provided.
        min_article_length: Minimum character length of text field after stripping.
        random_seed: Seed for sampling when truncating to max_per_topic.

    Returns:
        Tuple of (valid_articles, dropped_records).
    """


def print_normalisation_summary(valid: list[dict], dropped: list[DroppedRecord]) -> None:
    """
    Log a summary via logger.info(). Show:
    - Total valid records
    - Breakdown by (country, language, topic)
    - Total dropped records with count per reason
    Example output (via logger.info):
        Normalisation complete: 342 valid, 58 dropped
          text_too_short:          31
          language_not_in_config:  14
          topic_not_in_config:     10
          country_not_in_config:    3
          duplicate:                3
    """
```

---

### src/preprocessing.py

Text cleaning and linguistic annotation. spaCy models are loaded once per session,
not per article. spaCy runs on CPU - caching both language models simultaneously
does not conflict with the "one model in VRAM at a time" GPU rule.

```python
import logging
import re
from typing import Optional

import spacy

logger = logging.getLogger(__name__)

# Module-level model cache. Populated by _get_spacy_model(). Never reload per article.
_SPACY_MODELS: dict[str, spacy.Language] = {}


def _get_spacy_model(language: str, model_name: str) -> spacy.Language:
    """
    Return cached spaCy model for language. Load on first call using model_name.
    Set nlp.max_length = 2_000_000 to handle long articles without raising.
    Raise RuntimeError with installation instructions if model not found.

    Args:
        language: "en" or "de" - used as cache key.
        model_name: spaCy model name from config (e.g. "en_core_web_sm").

    Returns:
        spaCy Language object.
    """
    if language not in _SPACY_MODELS:
        try:
            nlp = spacy.load(model_name)
            nlp.max_length = 2_000_000
            _SPACY_MODELS[language] = nlp
            logger.info("Loaded spaCy model: %s", model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            )
    return _SPACY_MODELS[language]


def clean_text(text: str) -> str:
    """
    Remove MediaWiki markup and normalise whitespace. Return cleaned string.

    Removal steps (applied in this order):
    1. MediaWiki templates: {{...}} including nested templates.
       Use mwparserfromhell.parse(text).strip_code() if available.
       Fall back to regex r'\{\{[^}]*\}\}' if mwparserfromhell is not installed,
       and log a warning that nested templates may not be fully removed.
    2. MediaWiki links: [[target|display]] → "display". [[target]] → "target".
       Regex: r'\[\[(?:[^\]|]*\|)?([^\]]*)\]\]' → r'\1'
    3. HTML tags: r'<[^>]+>' → "".
    4. URLs: r'https?://\S+' → "".
    5. Multiple whitespace/newlines → single space.
    6. Strip leading and trailing whitespace.

    Punctuation is NOT removed (required for sentence splitting).

    Args:
        text: Raw article text.

    Returns:
        Cleaned text string.
    """


def tokenize_and_tag(text: str, language: str, model_name: str) -> dict:
    """
    Split text into sentences, tokenise, and assign POS tags using spaCy.
    Calls _get_spacy_model(language, model_name) to get the cached model.
    POS tags use the Universal POS tagset (token.pos_) for cross-language consistency.
    Tokens list excludes whitespace tokens (token.is_space). Includes punctuation.

    Args:
        text: Cleaned article text.
        language: "en" or "de".
        model_name: spaCy model name from config.

    Returns:
        dict with:
            "sentences": list[str]             - sentence strings from spaCy's senter
            "tokens":    list[str]             - non-whitespace token strings
            "pos_tags":  list[tuple[str, str]] - [(token_str, universal_pos_tag), ...]
    """


def preprocess_articles(articles: list[dict], config: dict) -> list[dict]:
    """
    Apply clean_text and tokenize_and_tag to every article using spaCy's nlp.pipe()
    for batching. Processing order per language group:
      1. Filter articles to those matching the language.
      2. Run clean_text() on each article's "text" field - this is a separate pass
         before nlp.pipe(), because clean_text uses regex and cannot run inside a
         spaCy pipeline component.
      3. Collect the cleaned strings into a list.
      4. Feed that list to nlp.pipe() to get spaCy Doc objects in batch.
      5. For each (article, Doc) pair, extract sentences, tokens, and POS tags.
      6. Write "cleaned_text", "sentences", "tokens", "pos_tags" to the article dict.
    Articles with an unrecognised language field are skipped (logged as warning,
    fields not added, article remains in list).
    In the standard notebook pipeline, normalise_articles filters to configured
    languages before this function is called, so the unrecognised-language branch
    is a safety guard for ad-hoc use, not an expected normal-run path.

    spaCy model names are read from config["models"]["spacy_english"] and
    config["models"]["spacy_german"].

    Mutates each dict in-place (intentional - avoids list copy).

    Error handling has FIVE non-fatal layers; in each, the article remains in
    the list without preprocessing fields and the downstream pipeline must
    check `article.get("cleaned_text")` before using it:

      1. Config-level: if config["models"]["spacy_english"] (or _german) is
         missing or falsy for a language group, log a warning and skip that
         language group entirely. Avoids passing None to spacy.load.
      2. Model-load: if _get_spacy_model raises RuntimeError (e.g. model not
         installed), catch via `logger.exception` and skip that language
         group. _get_spacy_model still raises for direct callers (e.g.
         tokenize_and_tag); only preprocess_articles swallows the failure.
      3. Per-article clean_text: if clean_text raises for one article, catch
         per-article, remove that article from the nlp.pipe() input batch,
         and leave it in the list without preprocessing fields.
      4. Whole-batch nlp.pipe(): wrap `list(nlp.pipe(...))` in
         `try/except Exception`, log via `logger.exception`, and skip the
         rest of that language group. Same outcome as per-article failure,
         just at group granularity.
      5. Per-article field-extraction: errors after nlp.pipe() (e.g.
         accessing doc.sents on a malformed Doc) are caught per
         (article, Doc) pair, logged with article id, and leave the article
         without preprocessing fields.

    Args:
        articles: List of article dicts with "text" and "language" fields.
        config: Full config dict.

    Returns:
        Same list, with preprocessing fields added to successfully processed articles.
    """
```

---

### src/ner.py

NER inference using HuggingFace pipelines. One pipeline per language.

```python
import logging
from collections import Counter
from typing import Optional

import pandas as pd
from transformers import pipeline as hf_pipeline

from src.utils import get_device

logger = logging.getLogger(__name__)

# The two models produce different label sets after aggregation_strategy="simple":
# English (dslim/bert-base-NER):              PER, ORG, LOC, MISC
# German (Davlan/bert-base-multilingual-cased-ner-hrl): PER, ORG, LOC
# MISC is not produced by the German model. This is expected and documented.
# NER_CHUNK_SIZE and NER_CHUNK_OVERLAP are read from config, not defined here.


def validate_ner_config(config: dict) -> None:
    """
    Check that NER chunk config values are internally consistent.
    Called once at notebook startup (Cell 2) before any models load.

    Rules:
        Raise ValueError if config["ner"]["chunk_size"] <= 0.
        Raise ValueError if config["ner"]["chunk_overlap"] >= config["ner"]["chunk_size"].
        Raise ValueError if config["ner"]["chunk_overlap"] < 0.

    Rationale: _chunk_text raises ValueError for these conditions mid-run, which
    wastes time loading models. Early validation surfaces the error immediately.

    Args:
        config: Full config dict.

    Raises:
        ValueError: If NER chunk constraints are violated.
    """


def load_ner_pipeline(model_name: str, language: str) -> Pipeline:
    """
    Load a HuggingFace NER pipeline.
    device=get_device() for GPU/CPU selection.
    aggregation_strategy="average" - a word-level strategy that assigns one
    label per whole word. "simple" was used originally but splits a word into
    fragmentary "##"-prefixed entities when the model tags its BERT subword
    pieces inconsistently; "average" cannot produce subword fragments. See FIX-10.

    Args:
        model_name: HuggingFace model identifier from config.
        language: "en" or "de" - used for logging only.

    Returns:
        HuggingFace Pipeline object.

    Raises:
        RuntimeError: If model download or loading fails.
    """


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int]]:
    """
    Split text into overlapping chunks for NER inference.
    Chunks split at whitespace boundaries where possible to avoid cutting words.

    Algorithm:
    1. Start at position 0.
    2. If remaining text fits within chunk_size: add as final chunk and stop.
    3. Search backwards from position (start + chunk_size) for the nearest whitespace.
    4. If whitespace is found at position i: set end = i - 1 so text[end] is
       itself the whitespace character; the chunk text = text[start:end]
       therefore excludes the trailing whitespace (clean word boundary).
       If no whitespace is found within the window (e.g. a long URL or token):
       hard-break at position (start + chunk_size). This prevents infinite loops
       on text with no whitespace. Log a warning that a hard break was applied.
       Trade-off: breaking mid-word reduces NER accuracy for entities crossing the
       cut point. This is accepted - the alternative (skipping the article) is worse.
       Increase ner.chunk_size in config to reduce hard-break frequency.
    5. Compute next_start. There are two cases depending on step 4:
       a. NORMAL case (whitespace was found): next_start starts at
          (end_of_current_chunk - overlap), then searches backwards to the
          nearest whitespace within the overlap window only - i.e., search
          backwards from (end_of_current_chunk - overlap) but no further back
          than (end_of_current_chunk - chunk_size). This bounds the search to
          O(overlap) characters and prevents the start from jumping back further
          than one chunk's worth of content. If no whitespace found within that
          bounded window: next_start = (end_of_current_chunk - overlap) without
          adjustment.
       b. HARD-BREAK case (no whitespace was found in step 4): SKIP the backward
          whitespace search entirely. By definition there is no whitespace in
          the relevant character range, so the search would always fail. Use
          next_start = (end_of_current_chunk - overlap) directly.
       In both cases, apply the progress guard: if next_start <= start, force
       next_start = start + 1 to prevent infinite loops on pathological inputs.
    6. Repeat until all text is covered.

    Returns list of (chunk_text, start_offset) where start_offset is the
    character position of the chunk's first character in the original text.

    Edge cases:
    - chunk_size <= 0: raise ValueError.
    - overlap >= chunk_size: raise ValueError (would cause infinite loop).
    - Empty text: return empty list.

    Args:
        text: Full cleaned text.
        chunk_size: Target maximum characters per chunk.
        overlap: Target overlap in characters between consecutive chunks.

    Returns:
        List of (chunk_str, start_offset_int) tuples.

    Raises:
        ValueError: If chunk_size <= 0 or overlap >= chunk_size.
    """


def _resolve_overlapping_entities(
    entities: list[dict],
    cleaned_text: str,
) -> list[dict]:
    """
    Deduplicate entities produced by overlapping chunks and validate offsets.

    Execution order is mandated: exact-duplicate collapse FIRST, then partial-
    overlap resolution on the deduplicated list. This ensures the partial-
    overlap pass never has to compare an entity against multiple copies of
    itself.

    Exact duplicates: group by (start, end, label) - keep the one with the
    highest score.

    Partial overlaps: two entities whose spans overlap but are not identical
    (e.g. "New York" start=10 end=18 and "New York City" start=10 end=23).
    The overlap test uses STRICT inequalities:
        ent["end"] > prev["start"] AND ent["start"] < prev["end"]
    Entities that touch at a boundary (e.g. [0,5] and [5,10]) share no
    characters and are NOT treated as overlapping - both are kept. This
    matches the intuitive reading of "overlap" as "shares characters".
    Resolution when entities do overlap: keep the entity with the larger
    (end - start) value. If equal, keep the higher score. Use (end - start)
    rather than len(entity["text"]) as the authoritative measure - offsets
    are the ground truth after adjustment.
    Rationale: the longer span is more specific and generally more useful for
    downstream analysis. Confidence-weighted merging was considered but requires
    assumptions about score comparability across chunks that are not guaranteed.

    Non-overlapping entities: keep all.

    Offset handling (FIX-10): the fast tokenizer's character offsets are the
    ground truth; the pipeline's reconstructed `.word` is lossy (spurious spaces
    such as "U. S." for "U.S.", leftover "##" subword markers). For each entity:
    - If 0 <= start < end <= len(cleaned_text): the offsets are in bounds.
      Overwrite entity["text"] with cleaned_text[start:end] - the canonical
      source slice - and keep the entity.
    - Otherwise the offsets are out of bounds (a genuine bug, e.g. broken
      chunk-offset arithmetic): log a warning and discard the entity.
    The previous behaviour - discarding any entity whose `.word` did not equal
    the slice - silently dropped real entities (WikiLeaks, Thierry Henry, ...)
    whose only fault was a lossy `.word`; that was a recall bug.

    Sort final list by start offset ascending.

    Args:
        entities: Flat list of entity dicts from all chunks, with adjusted offsets.
        cleaned_text: The full article cleaned_text, used for offset validation.

    Returns:
        Deduplicated, validated, and sorted list of entity dicts.
    """


def run_ner(
    articles: list[dict],
    ner_pipeline: object,
    language: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    Run NER on all articles matching language. Add "entities" field to each.

    Key rename: HuggingFace NER pipelines return dicts with key "entity_group",
    not "label", and "word", not "text". run_ner MUST rename both keys when
    building each entity dict for the article schema:
        entity["label"] = raw_entity["entity_group"]
        entity["text"]  = raw_entity["word"]
    An implementation that stores "entity_group" or "word" directly will
    silently break every downstream function that reads entity["label"] /
    entity["text"]. This rename is not optional. Note that entity["text"] is a
    provisional value here - _resolve_overlapping_entities (called in both the
    chunked and single-chunk paths) overwrites it with the canonical source
    slice cleaned_text[start:end]. See FIX-10.

    The implementation isolates this rename in a `_convert_raw_entity(raw)`
    helper that defensively reads either key, preferring the HuggingFace
    native key when both are present:
        entity["text"]  = raw.get("word", raw.get("text", ""))
        entity["label"] = raw.get("entity_group", raw.get("label", ""))
    This protects against mock pipelines or future API versions that emit
    the post-rename keys directly. HuggingFace native keys still win when
    both are present, so the rename remains the canonical contract.

    For articles where "cleaned_text" is missing (preprocessing failed):
    set entities=[] and log a warning. Do not attempt NER on articles
    without cleaned_text.

    For articles with cleaned_text longer than chunk_size characters:
    split with _chunk_text, run NER on each chunk, adjust start/end offsets
    by the chunk's start_offset, then call _resolve_overlapping_entities(
    entities, cleaned_text) to deduplicate and validate offsets.

    For articles at or below chunk_size: run NER once on the whole text, then
    still call _resolve_overlapping_entities(entities, cleaned_text). For a
    single chunk there are no cross-chunk overlaps, but routing through the
    same function keeps entity-text canonicalisation (FIX-10) identical across
    both paths - short and long articles get the same treatment.
    Note: chunk_size is in characters, not tokens. A chunk of chunk_size=400
    characters will typically tokenise to well under BERT's 512-token limit for
    standard news text, but dense or non-Latin text may produce more tokens per
    character. This is a known limitation (character-based chunking is simpler
    and sufficient for the target dataset). The HuggingFace pipeline will
    truncate at 512 tokens if a chunk exceeds the limit.

    Set entities=[] (not None) for articles where NER ran but found nothing,
    or where cleaned_text is empty.
    Set entities=None for articles where NER was not run (wrong language).

    Per-article errors: catch, log with article ID, set entities=[], continue.

    Args:
        articles: List of article dicts with "language" fields.
        ner_pipeline: Loaded HuggingFace NER pipeline.
        language: Only process articles where article["language"] == language.
        chunk_size: Maximum characters per chunk (from config).
        chunk_overlap: Overlap characters between chunks (from config).

    Returns:
        Same list with "entities" field added to matching articles.
    """


def build_entity_dataframe(articles: list[dict]) -> pd.DataFrame:
    """
    Flatten all entities from all articles into one row per entity occurrence.

    Skip behaviour:
    - articles where "entities" key is absent: skip (NER was never run on this article)
    - articles where entities is None: skip (NER was not run for this article's language)
    - articles where entities is [] (empty list): include in iteration, produces zero rows
      (NER ran, found nothing - this is valid and expected for some articles)

    Do not add a guard that skips empty lists - iterating an empty list naturally
    produces zero rows, which is the correct output. Adding a skip guard would
    produce the same result but obscure the distinction between "NER not run"
    (None/absent) and "NER ran, nothing found" (empty list).

    This function is designed to be called with ner_articles (Cell 12). Calling it
    with summ_articles or a mixed list will silently skip most articles because
    they will not have an "entities" key. No error is raised in this case - the
    caller is responsible for passing the correct article list.

    Title handling: article["title"] is "" for articles with no source title.
    Empty title cells break the usability of investigate_ner_errors - a reviewer
    sees blank rows and cannot identify which articles to review.
    When title == "", substitute article["id"] as the display value:
        display_title = article["title"] if article["title"] else f"[id: {article['id']}]"
    Store display_title in the "title" column, not the raw title field.

    Columns:
        article_id (str), event_id (object - str or float NaN), title (str), date (object - str or float NaN),
        language (str), topic (str),
        entity_text (str), entity_label (str), score (float)
    Note: pandas stores None in an object-dtype column as float NaN, not Python
    None. Code that checks date absence must use pd.isna(date), not `date is None`.

    Args:
        articles: List of article dicts. Should be ner_articles from Cell 12.

    Returns:
        pd.DataFrame. Empty DataFrame (with correct columns) if no entities found.
    """


def plot_top_entities(df: pd.DataFrame, top_n: int, language: str) -> None:
    """
    Plot a horizontal bar chart of the top_n most frequent entity_text values
    for the given language. Count by number of distinct article_id values per
    entity_text (not total row occurrences), to avoid one verbose article
    dominating the chart. Group by entity_text alone - entities with the same
    surface string but different labels (e.g. "Washington" as both LOC and PER)
    are merged into one bar. This is intentional: the chart shows name frequency,
    not label distribution.
    Title: f"Top {top_n} entities - {language.upper()}".
    Use matplotlib. Do not call plt.show() - let the notebook handle display.
    Import `matplotlib.pyplot as plt` LAZILY inside the function body, not at
    module top. Keeps `from src.ner import run_ner` import-time fast and
    avoids backend-selection side effects during test collection.

    If the filtered DataFrame (after applying the language filter) is empty,
    log a warning and return without creating a figure. Without this guard,
    matplotlib silently produces an empty plot or raises on `groupby(...).head(0)`.

    Args:
        df: Entity DataFrame.
        top_n: Number of entities to display.
        language: Filter to this language code.
    """


def plot_entity_dynamics(
    df: pd.DataFrame,
    entity_names: list[str],
    language: str,
) -> None:
    """
    Plot monthly entity mention frequency over time for a list of entities.
    Filter df to the given language. Parse date column with pd.to_datetime(errors="coerce").
    Drop rows where date is NaT after parsing (missing dates).
    Derive a year-month period column: df["year_month"] = parsed_dates.dt.to_period("M").
    Group by (entity_text, year_month). Count unique article_id per group.
    Plot one line per entity. X-axis: year_month values as strings ("YYYY-MM") for
    readable tick labels - convert with period.strftime("%Y-%m") or str(period).
    Use matplotlib's default tick spacing; do not set custom locators.
    Y-axis: article count (integer). Title includes language.
    Import `matplotlib.pyplot as plt` LAZILY inside the function body (same
    rationale as plot_top_entities).
    If the filtered DataFrame (after applying the language filter) is empty,
    log a warning and return without creating a figure.
    If fewer than 3 data points exist for any entity, log a warning that the
    time series is too sparse for meaningful trend analysis - but STILL plot
    the line for that entity. A one- or two-point line is informative (shows
    when the entity appeared); silently hiding it would be worse than the
    warning.

    Args:
        df: Entity DataFrame with date column.
        entity_names: Entity text strings to include.
        language: Language filter.
    """


def investigate_ner_errors(
    articles: list[dict],
    language: str,
    error_score_threshold: float,
) -> pd.DataFrame:
    """
    Collect entity predictions that are likely to be errors, for the given language.
    An entity is flagged as a candidate error if score < error_score_threshold.
    MISC label alone is NOT treated as an error - it is a valid label class.

    Note: this function collects *candidates* for manual review, not confirmed errors.
    The notebook must state this clearly above the output.

    Return a DataFrame with columns: article_id, event_id, title,
    entity_text, entity_label, score.
    Sort by score ascending (lowest confidence first).

    Args:
        articles: Article dicts with entities populated.
        language: Language code to filter to.
        error_score_threshold: Confidence below which entities are flagged (from config).
        country: Optional lowercase country label to filter to.

    Returns:
        pd.DataFrame of error candidates.
    """
```

---

### src/summarizer.py

Abstractive summarisation using `facebook/bart-large-cnn`.

```python
import logging

import pandas as pd
from transformers import pipeline as hf_pipeline

from src.utils import get_device

logger = logging.getLogger(__name__)


def validate_summarization_config(config: dict) -> None:
    """
    Check that summarization config values are internally consistent.
    Raise ValueError if min_summary_length >= max_summary_length.
    Called once at the start of the notebook before the pipeline loads.

    Args:
        config: Full config dict.

    Raises:
        ValueError: If min >= max for summary length.
    """


def load_summarization_pipeline(model_name: str) -> object:
    """
    Load and return a HuggingFace summarisation pipeline.
    device=get_device().

    Args:
        model_name: HuggingFace model identifier from config.

    Returns:
        HuggingFace pipeline object.
    """


def summarize_article(
    text: str,
    summ_pipeline: object,
    min_length: int,
    max_length: int,
) -> Optional[str]:
    """
    Generate a summary for a single article.
    Pass truncation=True to the pipeline - do not manually pre-truncate.
    The pipeline's built-in tokeniser handles truncation at the correct token boundary.

    Pipeline call signature: pass the parameters to summ_pipeline as keyword
    arguments named exactly truncation=True, min_length=..., and max_length=...
    These are the documented parameter names of the HuggingFace summarisation
    pipeline. Do not rename them, alias them, or pass them positionally - tests
    and downstream tooling assert against these exact kwarg names.

    Short-article guard: before calling the pipeline, use the pipeline's own tokeniser
    to count tokens. Access it via summ_pipeline.tokenizer - this is a public attribute
    on all HuggingFace Pipeline subclasses (defined in transformers.Pipeline base class).
    If the pipeline is replaced with a custom class that does not inherit from
    transformers.Pipeline, this attribute may not exist and should be verified.

    Token count:
        token_count = len(summ_pipeline.tokenizer.encode(text, add_special_tokens=False))
    Use add_special_tokens=False to count only content tokens, not BOS/EOS tokens
    added by the tokeniser. This gives a comparable baseline to min_length, which
    is defined in terms of output content tokens.
    Use .encode() rather than calling summ_pipeline.tokenizer(text) directly -
    .encode() returns a list of token ids, which is all that is needed here.
    If token_count < min_length, log a warning and return None.

    Why compare input token count against the output min_length threshold?
    BART is a generative model - if the input has fewer tokens than the minimum
    summary length requested, the model will hallucinate content to reach that
    minimum. The check prevents this: it is not asking "is the input long enough to
    be interesting?" but "can BART satisfy the min_length constraint without
    fabricating content?" If token_count < min_length, it cannot.

    Note: punctuation is present in cleaned_text and consumes part of the token
    budget. Very punctuation-heavy text may hit this guard slightly earlier than a
    word count would suggest.

    If text is empty string: log warning and return None.
    If pipeline raises: log error and return None.

    Args:
        text: Article's cleaned_text field.
        summ_pipeline: Loaded HuggingFace summarisation pipeline.
        min_length: Minimum summary token length (from config).
        max_length: Maximum summary token length (from config).

    Returns:
        Summary string, or None on failure or short-article guard.
    """


def summarize_articles(
    articles: list[dict],
    summ_pipeline: object,
    config: dict,
) -> list[dict]:
    """
    Run summarize_article on articles whose language is in
    config["languages"]["summarization"]. Do not hardcode "en" - read from config.
    For each qualifying article, call:
        summarize_article(
            article["cleaned_text"],
            summ_pipeline,
            min_length=config["summarization"]["min_summary_length"],
            max_length=config["summarization"]["max_summary_length"],
        )
    and store the result in article["summary"] (may be None on failure or short-article guard).
    Skip non-qualifying articles without logging (expected behaviour).
    Non-qualifying articles (e.g. German) do not receive a "summary" key at all;
    downstream code must use article.get("summary") rather than article["summary"].

    Count qualifying articles before the loop. Log progress every 5 articles:
    logger.info("Summarised %d/%d articles", n_done, n_total)

    Args:
        articles: List of article dicts with "cleaned_text" and "language" fields.
        summ_pipeline: Loaded summarisation pipeline.
        config: Full config dict.

    Returns:
        Same list with "summary" field added to qualifying articles.
    """


def build_summary_quality_dataframe(articles: list[dict]) -> pd.DataFrame:
    """
    Build a lightweight grammar/style review table for generated summaries.

    This is a heuristic screen for the assignment requirement to comment on grammar
    and style quality. It is NOT a full grammar checker and does not require an
    external service or Java-based LanguageTool dependency.

    Include only articles where article.get("summary") is not None.

    For each summary, compute:
        - article_id, title, country, topic
        - summary_char_count
        - summary_sentence_count
        - avg_sentence_chars
        - missing_terminal_punctuation (bool)
        - repeated_whitespace (bool)
        - very_long_sentence (bool; any sentence > 250 chars)
        - issue_count (sum of boolean issue flags)

    Sentence splitting rules:
      - Split on sentence-ending punctuation: . ! ?
      - After splitting, strip() each fragment and drop empty fragments before
        counting. Naive `re.split(r"[.!?]", "Hello. World.")` returns
        `["Hello", " World", ""]`; the trailing empty string would inflate
        sentence_count and skew avg_sentence_chars.
      - summary_sentence_count is the number of non-empty stripped sentences.
      - avg_sentence_chars averages character lengths over the same non-empty
        sentence set. If sentence_count == 0 (a summary with no sentence-ending
        punctuation and no content after stripping), use avg_sentence_chars = 0.0
        - DO NOT raise ZeroDivisionError. A zero-sentence summary is already
        flagged via missing_terminal_punctuation, so 0.0 is a non-misleading
        placeholder.

    missing_terminal_punctuation is tested against the WHOLE summary, not the
    last sentence after splitting: `summary.endswith((".", "!", "?"))`. This
    correctly flags summaries ending in whitespace, quotes, or truncated by
    max_length. Testing the last split sentence would always pass for any
    summary containing at least one `. ! ?` anywhere.

    Title handling: if article["title"] == "", use f"[id: {article['id']}]" as
    the display value.

    The notebook should display the highest issue_count rows and provide a short
    human-written finding such as: "Most summaries are grammatical on surface
    checks; the most common style issue is long sentences."

    Args:
        articles: Article dicts after summarization.

    Returns:
        pd.DataFrame. Empty with correct columns if no summaries exist.
    """
```

---

### src/similarity.py

Similarity scoring between originals and summaries.

```python
import logging
from typing import Optional

import pandas as pd
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# IMPORTANT: all-MiniLM-L6-v2 has a 256-token input limit.
# Long articles are truncated at 256 tokens before encoding.
# Tokenization is handled internally by SentenceTransformer.encode() using the
# model's own WordPiece tokenizer - the same tokenizer used during training.
# This is not word tokenization: subword tokens are produced, so 256 tokens
# corresponds roughly to 180–220 words depending on vocabulary coverage.
# Non-English text (German) is not passed to this model in this pipeline,
# but if it were, German compound words would tokenize into more subword tokens
# than English equivalents, reaching the limit sooner.
# This limitation must be stated in the notebook above the similarity results.
# See also: punctuation note in summarizer.py - punctuation consumes token budget
# here too, making the "~200 words" description approximate.


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load and return a SentenceTransformer model.

    Args:
        model_name: Model identifier from config.

    Returns:
        SentenceTransformer model object.
    """


def calculate_similarity(
    original: str,
    summary: str,
    model: SentenceTransformer,
) -> float:
    """
    Encode original and summary as sentence embeddings using model.encode().

    Call model.encode() TWICE - once for `original` and once for `summary` -
    each call returning a (1, dim) tensor. Do NOT pass [original, summary] as
    a single batch and slice the result. Rationale: the mock_embedding_model
    fixture (and the spec's mock contract at conftest.py) returns shape (1, dim)
    per call, not (n, dim) per batch. A batched call would return (1, dim) on
    the mock (one row) but (2, dim) on the real model - slicing [1:2] on the
    mock yields an empty tensor and float(result[0][0]) raises IndexError.
    Two separate calls yield two (1, dim) tensors that pass directly into
    util.cos_sim() regardless of mock or real model.

    Compute cosine similarity using sentence_transformers.util.cos_sim().

    cos_sim() returns a 2D PyTorch Tensor of shape (1, 1).
    Extract the scalar value with float(cos_sim_result[0][0]) before returning.

    Cosine similarity is theoretically in [-1, 1]. Related article/summary pairs
    should usually be positive, but implementation must not assume a lower bound of 0.

    Args:
        original: Article's cleaned_text.
        summary: Article's summary field.
        model: Loaded SentenceTransformer.

    Returns:
        Cosine similarity as Python float.
    """


def score_all_articles(
    articles: list[dict],
    model: SentenceTransformer,
) -> list[dict]:
    """
    Run calculate_similarity on every article where:
        article.get("cleaned_text") is not None and article.get("cleaned_text") != ""
        and article.get("summary") is not None
    Use .get() for both keys - non-English articles do not have a "summary" key at all,
    so article["summary"] would raise KeyError. article.get("summary") returns None safely.
    Add "similarity_score" field to qualifying articles.
    Skip non-qualifying articles without logging (expected behaviour).

    Per-article errors: catch, log, leave "similarity_score" unset.

    Args:
        articles: List of article dicts.
        model: Loaded SentenceTransformer.

    Returns:
        Same list with "similarity_score" added where applicable.
    """


def build_similarity_dataframe(articles: list[dict]) -> pd.DataFrame:
    """
    Build a DataFrame from articles that have a "similarity_score" field.
    Columns: article_id (str), title (str), topic (str), similarity_score (float).

    Title handling: use the raw `article.get("title", "")` value WITHOUT the
    `[id: {article['id']}]` substitution that ner.build_entity_dataframe and
    summarizer.build_summary_quality_dataframe apply. This is an intentional
    cross-module inconsistency: the similarity DataFrame is consumed by the
    histogram and extremes table (which always carry article_id alongside
    title), so the substitution is unnecessary. Downstream readers who need
    it can derive it from article_id.

    Args:
        articles: List of article dicts.

    Returns:
        pd.DataFrame. Empty with correct columns if no scored articles.
    """


def plot_similarity_distribution(
    df: pd.DataFrame,
    threshold: float,
) -> None:
    """
    Plot one subplot per topic showing the distribution of similarity scores
    as a histogram. Use matplotlib.pyplot.subplots() with one column per topic.

    Axis behaviour:
    - x-axis: shared range [-1.0, 1.0] across all subplots (use sharex=True).
    - y-axis: independent per subplot (do NOT use sharey). Sharing the y-axis
      when topics have different article counts makes smaller topics nearly
      invisible. Each subplot's y-axis label shows "Article count".
    - Bin count: 20 bins across the [-1.0, 1.0] range.

    Subplot order: sort topics lexicographically before plotting. Deterministic
    layout across runs is required for reproducible notebook output.

    Draw a vertical dashed line at threshold on every subplot.
    Title each subplot: f"{topic} (n={article_count})".
    Overall figure title: "Similarity score distribution by topic".
    Use matplotlib. Do not call plt.show().

    Args:
        df: Similarity DataFrame.
        threshold: Score value for the vertical dashed line.
    """


def explain_similarity_extremes(
    df: pd.DataFrame,
    n: int = 3,
) -> dict:
    """
    Find the n articles with the highest similarity_score and the n with the lowest.
    Tie-breaking: sort by str(article_id) lexicographically for deterministic output.
    str() cast is required because IDs may be source integers or hash strings depending
    on the dataset; lexicographic sort on str() is consistent regardless of original type.

    Empty-input guard: if the input DataFrame is empty, short-circuit and return
    {"highest": [], "lowest": []} without creating the `_id_str` helper column
    or calling `.head(n)`. Calling `.head(n)` on an empty DataFrame technically
    yields the same result, but the explicit guard is clearer and avoids
    unnecessary column derivation.

    Return:
        {
            "highest": list of n dicts, each with keys: article_id, title, topic, similarity_score
            "lowest":  list of n dicts, same keys
        }

    Args:
        df: Similarity DataFrame.
        n: Number of articles to return in each group.

    Returns:
        dict with "highest" and "lowest" keys.
    """
```

---

### src/topic_predictor.py

Zero-shot topic classification.

```python
import logging
import random
from typing import Optional

from transformers import pipeline as hf_pipeline

from src.utils import get_device

logger = logging.getLogger(__name__)

# HYPOTHESIS_TEMPLATE is read from config["topic_prediction"]["hypothesis_template"].
# It is not defined as a module-level constant.


def load_topic_pipeline(model_name: str) -> object:
    """
    Load and return a HuggingFace zero-shot-classification pipeline.
    device=get_device().

    Args:
        model_name: HuggingFace model identifier from config.

    Returns:
        HuggingFace pipeline object.
    """


def predict_topic(
    text: str,
    candidate_labels: list[str],
    topic_pipeline: object,
    hypothesis_template: str,
) -> Optional[str]:
    """
    Run zero-shot classification on text.
    Pass hypothesis_template to the pipeline.
    Return the label with the highest score.

    Failure handling has TWO layers, both returning None and logging a warning:
      1. Outer guard: if the pipeline call itself raises, return None.
      2. Inner guard: wrap the `result["labels"][0]` extraction in
         try/except (KeyError, IndexError, TypeError) and return None on
         failure. The pipeline contract usually guarantees a well-formed
         result, but mock pipelines or future API versions may return objects
         without a "labels" key or with an empty list. The inner guard
         preserves the spec's "return None on failure" contract for those
         shapes too.
    If text is empty: log warning and return None without calling the pipeline.

    Args:
        text: Article's cleaned_text.
        candidate_labels: Possible topic labels from config.topics.selected.
        topic_pipeline: Loaded zero-shot pipeline.
        hypothesis_template: From config["topic_prediction"]["hypothesis_template"].

    Returns:
        Predicted topic label string, or None.
    """


def predict_all_topics(
    articles: list[dict],
    candidate_labels: list[str],
    topic_pipeline: object,
    hypothesis_template: str,
    sample_size: int,
    random_seed: int = 42,
) -> list[dict]:
    """
    Sample up to sample_size articles, balanced across country-topic groups, and run predict_topic.

    Language filtering is the caller's responsibility. This function does not
    have access to a language list and cannot enforce it. The notebook passes
    summ_articles which is already English-only. If a caller passes a mixed-language
    list, non-English articles will be processed - this model (bart-large-mnli) is
    English-only and will produce unreliable results on other languages. Log a warning
    for any article where article.get("language") is not "en".

    Pre-filter before sampling: exclude articles where "cleaned_text" is absent or
    empty. Use article.get("cleaned_text", "") - do not raise KeyError. Log how many
    articles were excluded from sampling due to missing cleaned_text.

    Short-circuit guard: AFTER the pre-filter, if the eligible pool is empty
    OR if sample_size <= 0, return [] immediately. Do not build groups, do not
    call predict_topic. The quota math would otherwise divide by zero or run
    a useless pass.

    Sampling: create ONE stateful RNG object at the start:
        rng = random.Random(random_seed)
    Use this single rng instance for ALL sampling calls throughout the function -
    both the initial per-group pass and the redistribution pass.
    Do NOT call random.Random(random_seed) again inside the function. Re-instantiating
    with the same seed resets the state and will re-draw the same sequence, potentially
    re-selecting already-chosen articles (random.sample() prevents this within one call
    but not across re-instantiated RNG objects).

    For each (country, topic) group, compute quota =
    floor(sample_size / n_country_topic_groups).
    Sort articles within each group by str(article["id"]) before sampling.
    article["id"] is guaranteed to be present after normalisation (it is either
    the source id or a generated hash). Do not fall back to list index.
    Use rng.sample() to select quota articles per country-topic group.
    Track all selected article ids in a set.
    If a group has fewer articles than its quota, include all of them.
    A quota of 0 (when sample_size < n_country_topic_groups) is allowed and
    needs no special branch: the initial per-group pass selects nothing, and
    the redistribution pass below fills the entire sample_size from groups
    with remaining capacity.

    Redistribution pass: after the initial per-group pass, compute:
        remaining_slots = sample_size - len(selected_articles)
    If remaining_slots <= 0, skip redistribution. Otherwise:
        collect groups that still have unsampled articles after the initial quota
        pass. These are groups with available_unsampled > 0, usually because they
        had more articles than their quota while other groups were undersized.
        Compute per_group_extra = ceil(remaining_slots / n_groups_with_remainder).
        For each such group, draw min(per_group_extra, available_unsampled) articles
        using the same rng object. Stop when remaining_slots reaches 0.
        Filter the pool to exclude already-selected article ids before sampling -
        if the filtered pool has fewer than the requested count, use rng.sample(pool,
        len(pool)) rather than passing an undersized k to random.sample (which raises
        ValueError).

    Important: when the same random_seed is used for both normalisation passes
    (NER pass and summarisation pass), the English article samples in both passes
    will be identical if both passes drew from the same input pool. This is expected
    and acceptable - the two passes serve different analysis purposes and need not
    differ. If different samples are required, use random_seed + 1 for one pass.
    This is a configuration decision, not a code change.

    Create a shallow copy of each sampled article dict before adding "predicted_topic":
        sampled_article = {**original_article, "predicted_topic": None}
    This prevents mutation of the original articles list.
    Return only the copied, sampled articles.

    predict_topic receives article.get("cleaned_text", "") - never raises KeyError.
    An empty string causes predict_topic to return None per its own spec.

    Per-article errors: catch, log, set "predicted_topic" = None on that article's copy.

    Args:
        articles: Article list - caller should pass English-only articles.
        candidate_labels: Topic label strings from config.topics.selected (original casing).
        topic_pipeline: Loaded zero-shot pipeline.
        hypothesis_template: Template string from config.
        sample_size: Target number of articles to sample (from config).
        random_seed: Seed for balanced sampling.

    Returns:
        List of copied, sampled article dicts with "predicted_topic" added.
    """


def evaluate_topic_predictions(sampled_articles: list[dict]) -> dict:
    """
    Compare predicted_topic to topic for each sampled article.

    IMPORTANT - systematic casing difference: article["topic"] is always lowercase
    (normalise_articles step 7 lowercases all topic strings). predicted_topic
    preserves the original casing from config["topics"]["selected"] (e.g. "Sports").
    A direct equality check (topic == predicted_topic) will ALWAYS return False for
    correct predictions, producing accuracy=0.0 silently.

    Comparison MUST normalise both sides: lower().strip() on both values before
    comparing. The comparison is case-insensitive and strips whitespace.
    The normalisation is applied ONLY to the values used in the equality check.
    The topic value written into the results entries below is passed through
    from the article dict WITHOUT coercion (it is already lowercase after
    normalise_articles; coercing again would diverge from the article dict
    shape and confuse callers who construct test fixtures by hand).
    Skip articles where predicted_topic is None.

    Title handling: use the same substitution as build_entity_dataframe -
    if article["title"] == "", use f"[id: {article['id']}]" as the display value.

    Return:
        {
            "accuracy": float,          # correct / evaluated (excludes None predictions)
            "correct":  int,
            "evaluated": int,           # articles where predicted_topic is not None
            "total_sampled": int,       # all sampled articles including None predictions
            "results": list[dict],      # one per article:
                                        #   title (str) - substituted id if title is "",
                                        #   match (bool),
                                        #   topic (str) - copied directly from article, lowercase,
                                        #   predicted_topic (str|None) - original casing from config
        }

    Note: accuracy is computed on the summarisation pool (up to 60 articles) and is
    indicative only. The notebook must state the sample size and warn against
    over-interpreting the number.

    Empty-evaluation rule: when evaluated == 0 (every prediction was None, or
    sampled_articles is empty), accuracy is 0.0 - not NaN. The notebook prints
    f"{accuracy:.1%}" without NaN-handling, and downstream comparisons against
    accuracy assume a finite float. Do not return float('nan') or None.

    Args:
        sampled_articles: Articles returned by predict_all_topics().

    Returns:
        Evaluation results dict.
    """


def plot_topic_confusion_matrix(
    eval_results: dict,
    candidate_labels: list[str],
) -> None:
    """
    Plot a confusion-matrix heatmap of true vs predicted topic counts.

    Row index: true topic (article["topic"]). Column index: predicted topic.
    The label order on both axes follows `candidate_labels` (typically
    config["topics"]["selected"]) for deterministic layout. Cells on the diagonal
    are correct predictions; off-diagonal cells are errors.

    Case handling: article topics are stored lowercase after normalise_articles;
    predicted_topic preserves the source casing from candidate_labels. Both sides
    are lowered+stripped before indexing into the label map, mirroring
    evaluate_topic_predictions's comparison logic.

    If `results` is empty, log a warning and return without creating a figure.

    Cell annotations: each cell is labelled with its integer count. Use white
    text on cells above half-max and black text below for legibility.

    Import `matplotlib.pyplot as plt` LAZILY inside the function body, same
    rationale as the other plot functions (see plot_top_entities).

    Args:
        eval_results: Dict returned by evaluate_topic_predictions.
        candidate_labels: Topic labels in display order — typically
            config["topics"]["selected"].
    """


def plot_topic_error_breakdown(eval_results: dict) -> None:
    """
    Three-panel error breakdown for topic prediction.

    Panel 1 (left, "Errors by TRUE topic"): bar chart of error count grouped
        by the article's true topic. Answers "which topics get misinterpreted
        most often?".
    Panel 2 (middle, "Errors by PREDICTED topic"): bar chart of error count
        grouped by the model's predicted topic. Answers "which predictions are
        unreliable?".
    Panel 3 (right, "Overall"): three bars — Correct, Wrong, None. None counts
        articles where predicted_topic is None (excluded from accuracy
        denominator). Title shows the accuracy percentage.

    Sort the bars in panels 1 and 2 in descending order by count so the worst
    offender is left-most.

    If `results` is empty, log a warning and return without creating a figure.
    If a panel has no errors (empty Counter), show a centered "No errors" label
    rather than an empty axis.

    Args:
        eval_results: Dict returned by evaluate_topic_predictions. Must contain
            "results", "correct", "evaluated", "total_sampled".
    """
```

---

## Test specifications

### conftest.py - shared fixtures

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def sample_en_article() -> dict:
    """Minimal valid English article after normalisation."""
    return {
        "id": "abc123",
        "title": "Scientists discover new element",
        "text": "Researchers at the University of Berlin announced today that they "
                "have discovered a new chemical element. The element, temporarily "
                "named Berlinium, was found during experiments at the particle "
                "accelerator. Professor Hans Schmidt led the research team.",
        "language": "en",
        "country": "germany",
        "topic": "science and technology",
        "date": "2021-06-15",
        "event_id": None,
    }

@pytest.fixture
def sample_de_article() -> dict:
    """Minimal valid German article after normalisation."""
    return {
        "id": "def456",
        "title": "Bundestagswahl 2021",
        "text": "Die Bundestagswahl findet am 26. September 2021 statt. "
                "Kanzlerkandidat Olaf Scholz der SPD liegt in Umfragen vorne. "
                "Angela Merkel tritt nach 16 Jahren als Bundeskanzlerin nicht "
                "mehr an. Die CDU/CSU kämpft um den Erhalt der Regierungsmacht.",
        "language": "de",
        "country": "germany",
        "topic": "politics and conflicts",
        "date": "2021-09-01",
        "event_id": None,
    }

@pytest.fixture
def sample_raw_record() -> dict:
    """A raw record with non-standard field names, for normaliser tests."""
    return {
        "article_body": "This is the article text. It is longer than one hundred "
                        "characters to pass the minimum length check in the normaliser.",
        "headline": "Test Article",
        "publish_date": "2021-03-10",
        "lang": "en",
        "category": "Sports",
        "country": "Germany",
        "article_id": "raw001",
        "pageid": 12345,
    }

@pytest.fixture
def mock_ner_pipeline():
    """
    Mock HuggingFace NER pipeline for use in test_ner.py.
    Returns a callable that produces raw HuggingFace NER output - i.e., the
    pre-rename format with "entity_group" and "word" keys, exactly as the real
    pipeline returns before run_ner applies its key rename.
    Tests of run_ner should assert that the output articles use "label" (renamed),
    not "entity_group" (raw). This fixture intentionally uses the raw format to
    exercise the rename logic in run_ner.
    """
    pipeline = MagicMock()
    pipeline.return_value = [
        {"word": "Berlin", "entity_group": "LOC", "score": 0.98, "start": 5, "end": 11}
    ]
    return pipeline

@pytest.fixture
def mock_summ_pipeline():
    """
    Mock HuggingFace summarisation pipeline.
    Must include a .tokenizer attribute with a working .encode() method,
    because summarize_article calls summ_pipeline.tokenizer.encode(text)
    to count input tokens before summarising.
    len(MagicMock()) raises TypeError - the return value must be a real list.
    """
    pipeline = MagicMock()
    pipeline.return_value = [{"summary_text": "A short summary of the article."}]
    pipeline.tokenizer = MagicMock()
    pipeline.tokenizer.encode.return_value = list(range(100))
    # 100 fake token ids - enough to pass the min_summary_length=50 guard.
    # Adjust if a test needs to exercise the short-article guard specifically.
    return pipeline

@pytest.fixture
def mock_embedding_model():
    """
    Mock SentenceTransformer that returns fixed embeddings.
    Shape must be (1, dim) - cos_sim() requires 2D input.
    torch.tensor([0.5, 0.5, 0.5]) is shape (3,) and will cause cos_sim to fail.
    torch.tensor([[0.5, 0.5, 0.5]]) is shape (1, 3) and is correct.
    """
    import torch
    model = MagicMock()
    model.encode.return_value = torch.tensor([[0.5, 0.5, 0.5]])
    return model
```

### Functions intentionally not covered by unit tests

These functions produce visual or environment-dependent output. They are exercised
manually in the notebook, not in CI. Their behaviour is verified by reviewing the
notebook output during development. These omissions are explicit scope decisions, not
unresolved spec gaps - a test file MUST NOT include placeholder assertions for them.

- **`ner.plot_top_entities`, `ner.plot_entity_dynamics`, `similarity.plot_similarity_distribution`** -
  matplotlib output. The spec is explicit that these never call `plt.show()`; visual
  correctness is reviewer-judged in the notebook. Asserting axis labels or pixel layout
  in unit tests is brittle and does not exercise the actual visual claim.
- **`preprocessing.clean_text` mwparserfromhell branch** - the spec defines a regex
  fallback when `mwparserfromhell` is not installed. Which branch executes depends on the
  environment, not on the input. Tests assert behaviour on flat markup; nested-template
  handling is a known degradation in the fallback path and is logged as a warning per
  the spec.
- **`data_inspector.print_raw_profile`, `print_category_profile`, `print_validation_report`,
  `data_normalizer.print_normalisation_summary`** - log-output helpers. Tests assert only
  that they do not raise on populated inputs; the formatted text is reviewer-judged.
- **`ner.load_ner_pipeline`, `summarizer.load_summarization_pipeline`,
  `similarity.load_embedding_model`, `topic_predictor.load_topic_pipeline`** - thin
  wrappers around `transformers.pipeline()` / `SentenceTransformer()`. The RuntimeError
  contract on model-load failure is a documented behaviour of the underlying library;
  asserting it in CI without downloading a real model produces tautological tests.

### Expected test behaviours per module

**test_data_loader.py**
- `download_dataset` with an existing directory containing one recognised data file >100KB: skips download, returns Path
- `download_dataset` with an empty directory (only `.gitkeep`): does not skip, attempts download
- `download_dataset` raises `RuntimeError` on network failure (mock `requests.get` to raise)

**test_data_inspector.py**
- `detect_format` on a directory of `.json` files: returns `"json"`
- `detect_format` on a directory of `.txt` files: returns `"directory-of-txt"`
- `detect_format` on a directory with one `.jsonl` data file plus `README.md`: returns `"jsonl"`
- `detect_format` on an empty directory: returns `"unknown"`
- `category_profile` on Wikinews-style records with `categories` list: returns counts sorted by count descending
- `category_profile` deduplicates repeated category labels within one record
- `category_profile` on records without `categories`: returns empty DataFrame with correct columns and attrs
- `validate_normalised` with zero articles: sets `validation_passed = False`
- `validate_normalised` with a config topic not present in articles: adds a warning
- `validate_normalised` with a config country not present in articles: adds a warning

**test_data_normalizer.py**
- `normalise_articles` with `sample_raw_record`: fields correctly mapped (article_body→text, headline→title, etc.)
- `normalise_articles` with Wikinews-style `categories=["Politics and conflicts", ...]` maps to configured topic `"politics and conflicts"`
- `normalise_articles` with Wikinews-style `categories=["Germany", ...]` maps to configured country `"germany"`
- `normalise_articles` with Wikinews-style `pageid` stores it as `event_id`, not `id`
- Article with no configured country match appears in DroppedRecord with reason `"country_not_in_config"`
- Article with text shorter than `min_article_length`: appears in DroppedRecord with reason `"text_too_short"`
- Article with language `"fr"` when languages=`["en","de"]`: dropped with reason `"language_not_in_config"`
- Two identical articles: second is dropped with reason `"duplicate"`
- Stable ID: running normalisation twice on the same record produces the same `id`

**test_preprocessing.py**
- `clean_text` on text with `[[Berlin|Berlin, Germany]]`: returns "Berlin, Germany"
- `clean_text` on text with `{{cite web|url=...}}`: template removed
- `tokenize_and_tag` on `sample_en_article["text"]`: returns non-empty sentences, tokens, and pos_tags
- `pos_tags` contains only Universal POS tags (NOUN, VERB, PROPN, etc.)

**test_ner.py** (mock the HuggingFace pipeline to avoid model download in CI)
- `run_ner` skips articles where language != target language (entities remains None)
- `run_ner` sets entities=[] when pipeline returns no spans
- `build_entity_dataframe` on article with 3 entities: DataFrame has 3 rows

**test_summarizer.py** (mock pipeline)
- `validate_summarization_config` raises ValueError when min >= max
- `summarize_article` with empty string: returns None
- `summarize_articles` only calls pipeline on English articles
- `build_summary_quality_dataframe` flags missing terminal punctuation and repeated whitespace

**test_similarity.py** (mock SentenceTransformer)
- `explain_similarity_extremes` with tied scores: output is deterministic (same order on repeated calls)
- `build_similarity_dataframe` on articles with no summaries: returns empty DataFrame with correct columns

**test_topic_predictor.py** (mock pipeline)
- `predict_all_topics` with random_seed=42: same sample on every call
- `evaluate_topic_predictions` with all correct predictions: accuracy=1.0
- `evaluate_topic_predictions` with one None prediction: excluded from accuracy denominator

### CI note
HuggingFace pipelines must be mocked in all tests. Because the modules import
`pipeline` as the module-local alias `hf_pipeline`, patch the alias where it is used:
`src.ner.hf_pipeline`, `src.summarizer.hf_pipeline`, and
`src.topic_predictor.hf_pipeline`. Tests must not download any model. The
`conftest.py` should provide a `mock_ner_pipeline` fixture that returns a list of
dicts with the raw HuggingFace entity structure (`entity_group`, `word`, `score`,
`start`, `end`).

GPU and device detection must also be mocked in CI environments without CUDA.
Because `get_device` is imported into each module, patch the module-local names
used by model loaders: `src.ner.get_device`, `src.summarizer.get_device`, and
`src.topic_predictor.get_device`. Without this, `get_device()` returns `0` on any
machine where `torch.cuda.is_available()` is True, which causes pipeline loading to
attempt GPU allocation - failing in CI. Add to conftest.py:

```python
@pytest.fixture(autouse=True)
def mock_device(monkeypatch):
    """Force CPU device in all tests - prevents GPU allocation in CI."""
    monkeypatch.setattr("src.ner.get_device", lambda: -1)
    monkeypatch.setattr("src.summarizer.get_device", lambda: -1)
    monkeypatch.setattr("src.topic_predictor.get_device", lambda: -1)
```

---

## Notebook orchestration

`analysis.ipynb` contains cells in this order. Each cell has one responsibility.

```
Cell 1  - Setup
    from pathlib import Path
    import yaml
    import pandas as pd
    from src.utils import setup_logging, RANDOM_SEED

    # Config is loaded from Path.cwd() / "config" / "config.yaml".
    # The notebook MUST be run with the project root as the working directory.
    # In Jupyter: File > Change Kernel Working Directory, or start jupyter from
    # the project root. __file__ is not defined in Jupyter notebooks and cannot
    # be used for path resolution.
    config_path = Path.cwd() / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Run Jupyter from the project root directory: "
            "cd /path/to/wikinews-nlp && jupyter notebook"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    setup_logging(log_file=config["logging"]["log_file"])
    random_seed = config.get("random_seed", RANDOM_SEED)

    import src.data_loader as data_loader
    import src.data_inspector as data_inspector
    import src.data_normalizer as data_normalizer
    import src.preprocessing as preprocessing
    import src.ner as ner
    import src.summarizer as summarizer
    import src.similarity as similarity
    import src.topic_predictor as topic_predictor
    from src.utils import release_model

Cell 2  - Validate config
    summarizer.validate_summarization_config(config)
    ner.validate_ner_config(config)
    # Both fail fast if config values are internally inconsistent.
    # Catches errors before any model is loaded.

Cell 3  - Download data
    raw_path = data_loader.download_dataset(
        config["data"]["source_url"],
        config["data"]["raw_path"]
    )

Cell 4  - Raw profile (human review gate)
    fmt = data_inspector.detect_format(str(raw_path))
    profile = data_inspector.raw_profile(str(raw_path), fmt)
    data_inspector.print_raw_profile(profile)
    # HUMAN: review output. Confirm format and record count look correct before continuing.

Cell 5  - Category profile (config selection gate)
    category_df = data_inspector.category_profile(str(raw_path), fmt)
    data_inspector.print_category_profile(category_df, top_n=50)
    display(category_df.head(50))
    # HUMAN: review available category labels and counts.
    # If topics.selected or countries.selected in config.yaml need changes:
    #   1. Edit config/config.yaml.
    #   2. Rerun Cell 1 to reload config, or run:
    #        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    #        random_seed = config.get("random_seed", RANDOM_SEED)
    #   3. Rerun Cell 2 validation, then continue from Cell 6.
    # No interactive input is requested in the notebook; config remains the source of truth.

Cell 6  - Normalise (NER pass: en + de)
    ner_articles, dropped_ner = data_normalizer.normalise_articles(
        str(raw_path), fmt,
        languages=config["languages"]["ner"],
        topics=config["topics"]["selected"],
        countries=config["countries"]["selected"],
        max_per_topic=config["topics"]["articles_per_topic_ner"],
        min_article_length=config["data"]["min_article_length"],
        random_seed=random_seed,
    )
    data_normalizer.print_normalisation_summary(ner_articles, dropped_ner)

Cell 7  - Normalise (summarisation pass: en only)
    summ_articles, dropped_summ = data_normalizer.normalise_articles(
        str(raw_path), fmt,
        languages=config["languages"]["summarization"],
        topics=config["topics"]["selected"],
        countries=config["countries"]["selected"],
        max_per_topic=config["topics"]["articles_per_topic_max"],
        min_article_length=config["data"]["min_article_length"],
        random_seed=random_seed,
    )
    data_normalizer.print_normalisation_summary(summ_articles, dropped_summ)
    # Note: ner_articles and summ_articles are separate processing pools.
    # With the same random_seed, English selections are identical when both passes
    # draw from the same source pool. Use random_seed + 1 for the second pass only
    # if the analysis deliberately needs different English samples.

Cell 8  - Validate both article sets (human review gate)
    ner_report = data_inspector.validate_normalised(ner_articles, config)
    data_inspector.print_validation_report(ner_report)
    summ_report = data_inspector.validate_normalised(summ_articles, config)
    data_inspector.print_validation_report(summ_report)
    # Hard guard - do not rely on the human to notice a failed validation.
    # Raise immediately so the notebook cannot silently continue on bad data.
    if not ner_report.validation_passed or not summ_report.validation_passed:
        raise RuntimeError(
            "Validation failed - review errors above before continuing. "
            "Fix the config or dataset, then re-run from Cell 6."
        )
    # HUMAN: also review warnings above. Warnings do not stop execution but
    # may indicate countries/topics with too few articles or high rates of missing dates.

Cell 9  - Preprocessing
    ner_articles = preprocessing.preprocess_articles(ner_articles, config)
    summ_articles = preprocessing.preprocess_articles(summ_articles, config)

Cell 10 - NER: English
    en_ner_pipeline = ner.load_ner_pipeline(config["models"]["ner_english"], "en")
    ner_articles = ner.run_ner(
        ner_articles, en_ner_pipeline, "en",
        chunk_size=config["ner"]["chunk_size"],
        chunk_overlap=config["ner"]["chunk_overlap"],
    )
    del en_ner_pipeline   # caller must del their reference before release_model()
    release_model()

Cell 11 - NER: German
    de_ner_pipeline = ner.load_ner_pipeline(config["models"]["ner_german"], "de")
    ner_articles = ner.run_ner(
        ner_articles, de_ner_pipeline, "de",
        chunk_size=config["ner"]["chunk_size"],
        chunk_overlap=config["ner"]["chunk_overlap"],
    )
    del de_ner_pipeline
    release_model()

Cell 12 - NER analysis
    entity_df = ner.build_entity_dataframe(ner_articles)

    # Top entities: one plot per language across all articles of that language
    ner.plot_top_entities(entity_df, top_n=20, language="en")
    ner.plot_top_entities(entity_df, top_n=20, language="de")

    # Entity dynamics: top 5 entities per language across all articles
    for lang in ["en", "de"]:
        top_entities = (
            entity_df[entity_df["language"] == lang]
            .groupby("entity_text")["article_id"]
            .nunique()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        ner.plot_entity_dynamics(entity_df, entity_names=top_entities, language=lang)

    # Task 3: Investigate low-confidence NER predictions for both languages.
    # NER models for well-resourced languages (e.g. English) typically have fewer
    # low-confidence predictions than models for lower-resource languages (e.g. German).
    # Comparing both makes the accuracy gap concrete.
    threshold = config["ner"]["error_score_threshold"]
    for lang in ["en", "de"]:
        print(f"\n=== Low-confidence NER candidates — {lang.upper()} (score < {threshold}) ===")
        error_df = ner.investigate_ner_errors(
            ner_articles,
            language=lang,
            error_score_threshold=threshold,
        )
        if error_df.empty:
            print(f"No low-confidence entities found for {lang.upper()}.")
        else:
            display(error_df)
    print("Note: these are confidence-based candidates for manual review, not confirmed errors.")

Cell 13 - Summarisation
    summ_pipeline = summarizer.load_summarization_pipeline(config["models"]["summarization"])
    summ_articles = summarizer.summarize_articles(summ_articles, summ_pipeline, config)
    del summ_pipeline
    release_model()
    summary_quality_df = summarizer.build_summary_quality_dataframe(summ_articles)
    display(summary_quality_df.sort_values("issue_count", ascending=False).head(10))
    print("Note: grammar/style findings are based on lightweight surface heuristics, "
          "not a full grammar checker.")

Cell 14 - Similarity scoring
    embedding_model = similarity.load_embedding_model(config["models"]["similarity"])
    summ_articles = similarity.score_all_articles(summ_articles, embedding_model)
    del embedding_model
    release_model()
    print("Note: similarity scores reflect only the first ~256 tokens of each article "
          "due to the embedding model's input limit. See known limitations in SPEC.md.")

Cell 15 - Similarity analysis
    sim_df = similarity.build_similarity_dataframe(summ_articles)
    similarity.plot_similarity_distribution(sim_df, threshold=config["similarity"]["threshold"])
    extremes = similarity.explain_similarity_extremes(sim_df)
    print("Highest scoring articles (summaries most faithful to original):")
    display(pd.DataFrame(extremes["highest"]))
    print("Lowest scoring articles (summaries lost most information):")
    display(pd.DataFrame(extremes["lowest"]))

Cell 16 - Topic prediction
    topic_pipeline = topic_predictor.load_topic_pipeline(config["models"]["topic_prediction"])
    sampled = topic_predictor.predict_all_topics(
        summ_articles,
        candidate_labels=config["topics"]["selected"],
        topic_pipeline=topic_pipeline,
        hypothesis_template=config["topic_prediction"]["hypothesis_template"],
        sample_size=config["topic_prediction"]["sample_size"],
        random_seed=random_seed,
    )
    del topic_pipeline
    release_model()
    eval_results = topic_predictor.evaluate_topic_predictions(sampled)
    print(f"Topic prediction accuracy: {eval_results['accuracy']:.1%} "
          f"({eval_results['correct']}/{eval_results['evaluated']} articles evaluated, "
          f"sample size={eval_results['total_sampled']})")
    print("Note: accuracy is based on a small sample and is indicative only.")
    display(pd.DataFrame(eval_results["results"]))

Cell 17 - Summary report (markdown cell, not code)
    # The cell is a markdown placeholder, not strictly empty. It contains the
    # `## Cell 17 - Summary report` heading plus a one-line italic prompt
    # listing what to summarise: country scope, entity counts, most frequent
    # entities, similarity score distributions, topic prediction accuracy,
    # any notable NER errors. The reviewer overwrites the italic line with
    # their own findings.
    # The notebook has exactly 17 cells total: 16 code cells (Cells 1–16) +
    # the Cell 17 markdown summary. `jupyter nbconvert --to script
    # notebooks/analysis.ipynb --stdout` exits 0 on the resulting file.
```

---

## Dependencies (requirements.txt)

```
torch>=2.1.0
# PyTorch CUDA compatibility: torch 2.1.x supports CUDA 11.8 and 12.1.
# Install the CUDA-enabled build explicitly - pip install torch alone may
# install the CPU-only build. Check https://pytorch.org/get-started/locally/
# for the correct install command for your CUDA version.
# For GTX 1060 (CUDA 11.x): pip install torch --index-url https://download.pytorch.org/whl/cu118
# For CPU-only systems: pip install torch (default PyPI build)
transformers>=4.40.0,<5.0.0
sentence-transformers>=2.7.0,<3.0.0
spacy>=3.7.0,<4.0.0
pandas>=2.0.0
matplotlib>=3.8.0
pyyaml>=6.0
requests>=2.31.0
mwparserfromhell>=0.6.4
jupyter>=1.0.0
pytest>=8.0.0
pytest-mock>=3.12.0
python-dotenv>=1.0.0
black>=24.0.0
ruff>=0.4.0
```

spaCy language models (run after pip install):
```
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

PEP8 enforcement: `ruff` checks line length (88 chars), import order, and common style issues.
`black` formats code. Both run as pre-commit hooks and in CI.

`.pre-commit-config.yaml` content:
```yaml
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
```

`pyproject.toml` excerpt:
```toml
[tool.ruff]
line-length = 88
select = ["E", "F", "I"]

[tool.black]
line-length = 88
```

---

## Known limitations

These are deliberate scope decisions, not bugs. Each is documented here so the notebook
can reference them clearly and reviewers do not flag them as missing features.

**Similarity scores reflect only article openings.** `all-MiniLM-L6-v2` truncates input
at 256 tokens. For long articles, the similarity score compares the summary against
the first ~200 words of the article, not the full text. This is a known constraint of
the embedding model chosen for its hardware footprint.

**Single GPU, single device.** `get_device()` returns device 0 only. Multi-GPU
systems and Apple Silicon MPS are not supported.

**No entity canonicalisation.** "USA", "U.S.", and "United States" are counted as
separate entities. No normalisation or alias resolution is implemented.

**Country filtering uses Wikinews category tags, not geocoding.** A record is included
when one configured country label appears in its raw `categories` list. The pipeline
does not infer country from article text, URLs, named entities, or event locations.
This keeps the country scope transparent and reproducible, but it depends on source
category quality.

**No download resume or checksum validation.** If a download is interrupted, the
partial file may pass the skip condition on the next run. In that case, delete the
`data/raw/` directory and rerun Cell 3.

**Two-pass normalisation produces identical English samples with the same seed.** When `normalise_articles()` is called twice with the same `random_seed` and the same source data, the English articles selected in both passes will be identical. This is expected - same seed, same sorted input, same `random.sample()` result. The two passes are independent analyses, not independent samples. If different English article sets are required for NER and summarisation, set `random_seed + 1` for the second call in the notebook. This is documented as a configuration choice.

**Download skip condition assumes one recognised data file over 100KB means a valid dataset.** This is a heuristic, not a guarantee. A partially downloaded dataset that happens to contain one recognised data file over 100KB will pass the skip check and be treated as complete. If the pipeline behaves unexpectedly after a previously interrupted download, delete `data/raw/` and rerun Cell 3.

**GitHub ZIP fallback tries `main` then `master` branch names.** If the source repository uses a different default branch, the fallback will fail. In that case, set `source_url` in config to a direct archive URL rather than the repository root URL.

**Full raw dataset loaded into memory.** No streaming or chunking at the dataset level.
This is acceptable for the target Wikinews JSONL file (~43MB, ~15,200 records).
After normalisation, downstream model stages process only the selected capped subset
(countries x languages x topics x articles_per_topic_max).

**No persistent cache.** Every notebook run reprocesses from scratch. There is no
persistence layer. If intermediate results need to be inspected between runs, save
them manually to `data/processed/` using standard Python (e.g. `json.dump` or
`pd.DataFrame.to_csv`) in an ad-hoc notebook cell.

**Monthly time-series resolution is sparse.** With 20 articles per country/language/topic
maximum, monthly entity dynamics plots may have only 1–2 data points per month.
The visualisation is illustrative rather than statistically meaningful.

**Text-content deduplication does not distinguish reprints from duplicates.** Two records with identical text but different source IDs - such as syndicated articles published across multiple outlets - are treated as duplicates and the second is dropped. Distinguishing a duplicate from a legitimate separate publication requires metadata beyond text content and is out of scope.

**Entities spanning beyond chunk overlap may be missed.** NER chunking uses a fixed overlap of `chunk_overlap` characters. If an entity spans the boundary between two non-overlapping regions (e.g. a long organisation name that starts before the overlap begins), it will not appear in either chunk's output. This is a known limitation of character-based chunking for NER. Increasing `chunk_overlap` in config reduces but does not eliminate this risk.

**Topic prediction evaluated on ~30 articles.** The accuracy figure is indicative only
and should not be compared against benchmark results.

---

## Implementation history

Decisions made during code implementation against this spec - including
ambiguities that were resolved, bugs that were found, and minor design
choices - are recorded in `docs/code_implementation_issues.md`. Every
behaviour from that log has been folded back into the module specifications
above, so a fresh implementation pass against this document should not
re-encounter the same ambiguities. The issues log is preserved for audit
trail only; the spec is the source of truth.

---

## Resolved open questions

All previously open questions are now resolved:

1. **Dataset format** - detected programmatically by `detect_format()`. No assumption made.
2. **Topic field consistency** - comparison uses `_normalise_topic_string()` (lowercase + strip) on both sides.
3. **BART truncation** - use `truncation=True` in the pipeline call. Do not pre-truncate manually.
4. **Entity dynamics date range** - show full available date range. Filter out None dates silently.
5. **Caching / resume** - no caching. Every notebook run reprocesses from scratch. This is a deliberate scope decision for this project.
6. **Language / topic mismatch across languages** - if a topic exists in English articles but not German, the German pass returns fewer articles and logs a warning in `validate_normalised()`. The pipeline continues.
7. **Country filtering** - required by the assignment and implemented via Wikinews
   `categories` labels. This is separate from language filtering; language controls
   model choice, while country controls analysis scope.
