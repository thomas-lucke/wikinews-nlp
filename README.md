# Wikinews Multilingual NLP Pipeline

A reproducible pipeline for named entity recognition, abstractive summarization, semantic similarity scoring, and zero-shot topic prediction on the multilingual Wikinews corpus. English + German for NER and analysis; English-only for summarization.

---

## About this project

Implementation was developed with AI-assisted coding (Claude Code) within the [Turing College AI-use guidelines](https://turingcollege.com/). Project design, specification ([`docs/SPEC_v3.md`](docs/SPEC_v3.md)), architectural decision records ([`docs/decisions/`](docs/decisions/)), model selection, threshold tuning, error analysis, and result interpretation are the author's work. The documentation tree exists so that any reviewer can trace any line of code back to its motivation and verify comprehension.

Author: Thomas Lucke · Module: M4 S2 · Branch: `fix/code_test`

---

## What it does

A 17-cell notebook orchestrates eight pipeline stages. Each stage lives in a dedicated `src/` module with full unit-test coverage.

| Stage | Module | What it does |
|---|---|---|
| Ingestion | [`src/data_loader.py`](src/data_loader.py) | Downloads the Wikinews multilingual dataset (zip / git clone fallback). |
| Profiling | [`src/data_inspector.py`](src/data_inspector.py) | Format-detects the raw data and reports language / category / length distributions. |
| Normalisation | [`src/data_normalizer.py`](src/data_normalizer.py) | Field-mapping, deduplication, balanced sampling by `(language, topic)`. |
| Preprocessing | [`src/preprocessing.py`](src/preprocessing.py) | MediaWiki / HTML cleanup, spaCy tokenisation, POS tagging, sentence segmentation. |
| NER | [`src/ner.py`](src/ner.py) | BERT-based NER for English (`dslim/bert-base-NER`) and German (`Davlan/bert-base-multilingual-cased-ner-hrl`), chunked at 400 chars. |
| Summarization | [`src/summarizer.py`](src/summarizer.py) | Abstractive summarization via `facebook/bart-large-cnn`, with grammar/style quality metrics. |
| Similarity | [`src/similarity.py`](src/similarity.py) | Cosine similarity (`all-MiniLM-L6-v2`) between each original article and its summary. |
| Topic prediction | [`src/topic_predictor.py`](src/topic_predictor.py) | Zero-shot classification with `facebook/bart-large-mnli` against the configured topic set. |

---

## Quick start

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management.

### 1. Install PyTorch

PyTorch CUDA builds are not on PyPI, so you pick your build at install time. Check your hardware first:

```powershell
nvidia-smi   # look for "CUDA Version: XX.X" — skip if no NVIDIA GPU
```

Then install the matching extra:

```powershell
uv sync --extra cpu        # no GPU, or unsure
uv sync --extra cu121      # CUDA 12.1
uv sync --extra cu124      # CUDA 12.4
```

`cpu` is the safe default — all pipeline stages work on CPU, just slower (~25 min end-to-end vs ~5 min on GPU).

### 2. Set device in config

Open [`config/config.yaml`](config/config.yaml) and set the `device` key to match your install:

```yaml
device: "auto"    # GPU if available, CPU otherwise (recommended)
device: "cpu"     # force CPU
device: "cuda"    # force GPU — errors if no CUDA GPU found
```

### 3. Launch the notebook

```powershell
uv run jupyter notebook notebooks/analysis.ipynb
```

The notebook **must** be launched from the project root — paths to `config/`, `data/`, and `src/` are resolved relative to `Path.cwd()`.

First-time run will download:
- The Wikinews multilingual dataset (~43 MB → `data/raw/`)
- Five HuggingFace models to your local cache (~3 GB cumulatively):
  - `dslim/bert-base-NER` (~430 MB)
  - `Davlan/bert-base-multilingual-cased-ner-hrl` (~700 MB)
  - `facebook/bart-large-cnn` (~1.6 GB)
  - `sentence-transformers/all-MiniLM-L6-v2` (~90 MB)
  - `facebook/bart-large-mnli` (~1.6 GB)

Subsequent runs use the local cache (no re-download). On CPU, end-to-end runtime is ~25 minutes; on a CUDA GPU, ~5 minutes.

---

## Configuration

All hyperparameters, model paths, and thresholds live in [`config/config.yaml`](config/config.yaml) — this is the single source of truth. The notebook reads it in Cell 2 and never hardcodes parameters. Change config → restart kernel → re-run from Cell 2.

Key knobs:

| Key | Default | Purpose |
|---|---|---|
| `topics.selected` | Politics, Science, Sports | Three Wikinews categories analysed end-to-end. |
| `topics.articles_per_topic_max` | 20 | Cap for the summarization pass (60 summaries total). |
| `topics.articles_per_topic_ner` | 100 | Cap for the NER pass (600 articles total — Task 3 has no sample-size restriction). |
| `languages.ner` | `["en", "de"]` | Languages run through NER analysis. |
| `languages.summarization` | `["en"]` | English-only for summarization ([ADR 0002](docs/decisions/0002-english-only-summarization.md)). |
| `ner.error_score_threshold` | 0.6 | Confidence below this is surfaced for manual review. |
| `similarity.threshold` | 0.8 | Cosine score above this is considered "faithful". |
| `random_seed` | 42 | Determinism for sampling and zero-shot topic prediction. |
| `device` | `"auto"` | Inference device: `"auto"`, `"cpu"`, or `"cuda"`. |

---

## Project structure

```
.
├── config/
│   └── config.yaml             # Single source of truth for all parameters
├── data/
│   └── raw/                    # Downloaded dataset (gitignored)
├── docs/
│   ├── SPEC_v3.md              # Function-level specification — every public API
│   ├── PRD.md                  # Product requirements
│   ├── CHANGELOG.md            # Versioned changes
│   ├── code_implementation_issues.md   # Deviation log (FIX-1..FIX-6)
│   └── decisions/              # Architectural decision records
│       ├── 0001-huggingface-for-ner.md
│       ├── 0002-english-only-summarization.md
│       ├── 0003-dataset-agnostic-pipeline.md
│       ├── 0004-sequential-gpu-loading.md
│       └── 0005-language-only-grouping.md
├── notebooks/
│   └── analysis.ipynb          # 17-cell orchestration notebook (entrypoint)
├── src/                        # All pipeline modules — one per stage
├── tests/                      # 152 unit tests, one suite per module
├── pyproject.toml              # Dependencies + ruff/black/pytest config
└── uv.lock                     # Reproducible dependency lockfile
```

---

## Tests

```powershell
uv run pytest tests/ -v
```

152 tests, ~25 seconds. Every module in `src/` has a matching `tests/test_<module>.py`. Tests use `MagicMock` for HuggingFace pipelines (no model downloads in CI), and `pytest --autouse` fixtures pin the device to CPU to keep CI off the GPU.

Linting + formatting:

```powershell
uv run ruff check . --fix       # auto-fixes import order, unused imports, etc.
uv run ruff format .            # black-style line-length=100 formatting
```

---

## Architectural decisions

Significant choices are recorded in [`docs/decisions/`](docs/decisions/). Each ADR follows a Context / Decision / Consequences / Alternatives format. The five current records:

1. [ADR 0001 — HuggingFace BERT for NER](docs/decisions/0001-huggingface-for-ner.md)
2. [ADR 0002 — English-only summarization](docs/decisions/0002-english-only-summarization.md)
3. [ADR 0003 — Dataset-agnostic ingestion](docs/decisions/0003-dataset-agnostic-pipeline.md)
4. [ADR 0004 — Sequential GPU loading](docs/decisions/0004-sequential-gpu-loading.md)
5. [ADR 0005 — Language-only sampling](docs/decisions/0005-language-only-grouping.md)

Deviations and bugs encountered during implementation are logged in [`docs/code_implementation_issues.md`](docs/code_implementation_issues.md) as `FIX-N` entries.

---

## Known limitations

These are intentional or unavoidable; each is acknowledged in the notebook output or relevant ADR.

- **Embedding context window (256 tokens).** `all-MiniLM-L6-v2` truncates inputs at 256 tokens, so similarity scores reflect only the article's opening. For news content this is partly mitigated by the inverted-pyramid convention. See [Cell 15 / 16](notebooks/analysis.ipynb) and [`docs/SPEC_v3.md`](docs/SPEC_v3.md).
- **Lowercasing / stopword removal / lemmatization skipped.** BERT and BART work on case-sensitive subword tokens — applying these would degrade NER and summarisation quality.
- **Entity labels limited to `PER / ORG / LOC / MISC`.** The chosen NER models don't emit "political entity" or "event" labels — those fall under `ORG` or `MISC`.
- **German summarization not attempted.** See [ADR 0002](docs/decisions/0002-english-only-summarization.md).
- **Country metadata removed from outputs.** Extracted unreliably from categories; not a sampling axis. See [ADR 0005](docs/decisions/0005-language-only-grouping.md) and FIX-6.
- **Manual labelling of NER errors.** `investigate_ner_errors` produces *candidates* for review (`score < 0.6`); confirming which are true errors vs. uncertain-but-correct is human work.

---

## Documentation map for reviewers

If you're reviewing this project, the documents below trace the work top-down:

- **Why each design choice exists** → [`docs/decisions/`](docs/decisions/) (five ADRs)
- **What each function is supposed to do** → [`docs/SPEC_v3.md`](docs/SPEC_v3.md)
- **What went wrong and how it was fixed** → [`docs/code_implementation_issues.md`](docs/code_implementation_issues.md) (FIX-1..FIX-6)
- **Product-level intent** → [`docs/PRD.md`](docs/PRD.md)
- **Version history** → [`docs/CHANGELOG.md`](docs/CHANGELOG.md)
- **Coding instructions and conventions** → [`CLAUDE.md`](CLAUDE.md)

Any line of code in `src/` can be traced back to either a SPEC function contract or a fix entry. Any unusual architectural choice has an ADR.

---

## License

Coursework project — Turing College, Module M4 Sprint 2. Not licensed for production use.
