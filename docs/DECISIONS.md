# Technology and Architecture Decisions

This document explains the choices made when building this pipeline — which tools were selected, why, and what the alternatives were. It is written for anyone reading this repository who wants to understand the reasoning behind the code, not just the code itself.

---

## How to read this document

Each section covers one category of decisions. For every choice, three questions are answered: what the tool is, why it was picked over the alternatives, and what you would use instead in a different situation. Where a choice involved a real trade-off, that trade-off is stated honestly.

---

## Project structure

### Why `src/` modules instead of a single notebook

Most data science tutorials and course projects put all logic inside a Jupyter notebook. That works for exploration, but it creates two problems: the code cannot be tested (you cannot import a notebook cell into a test file), and it cannot be reused (another script cannot call a function that only exists inside a notebook).

This project separates responsibilities. All logic lives in `src/` as proper Python modules with importable functions. The notebook imports those functions and uses them for display and orchestration. This means every function can be tested independently, and any module can be used outside the notebook context.

The notebook is a presentation layer, not an engine.

### Why YAML configuration

Every setting that a user might want to change — topic selections, model names, thresholds, file paths — lives in `config/config.yaml`. None of those values are hardcoded inside `src/` modules.

The reason is reusability. If you want to run the same pipeline on different topics, a different language, or a faster model, you change one file. No source code is touched. This also makes it safe to share the codebase: the config is the only file that describes your specific run, and it is readable without any Python knowledge.

### Why a three-module data pipeline instead of one loader

Early versions of this project had a single `data_loader.py` that downloaded, parsed, and loaded data in one pass. That approach assumes you know what the dataset looks like before you load it.

The three-module design (loader → inspector → normalizer) treats every dataset as unknown until proven otherwise. The loader only downloads. The inspector profiles and validates the raw files without touching or changing them. The normalizer maps whatever field names the source uses to the internal schema. The result is a pipeline that works on any dataset, not just Wikinews — you only need to update the field mappings if the source uses unusual column names.

### Why two normalisation passes

The NER pipeline needs English and German articles. The summarisation pipeline needs English only. Rather than loading all data once and filtering later — which would require holding German articles in memory during summarisation — the notebook calls `normalise_articles()` twice with different language lists.

The trade-off is double disk I/O. The benefit is that each pipeline stage receives exactly the data it needs, and nothing else. On the dataset sizes involved in this project, the I/O cost is negligible.

---

## NLP framework choices

### Why HuggingFace Transformers for NER and summarisation

HuggingFace is the standard interface for loading and running pre-trained transformer models. The key advantage is that swapping models requires one line of config change, not a code change. The pipeline API is identical regardless of which model is loaded.

The main alternative is calling a cloud NLP API — AWS Comprehend, Google Cloud Natural Language, or Azure Text Analytics. These handle infrastructure and require no GPU. The trade-offs are cost at scale, no control over the underlying model, and data leaving your machine. For a pipeline running on public news data where the model choice matters to the analysis, self-hosted HuggingFace is more appropriate.

Using a general-purpose LLM (GPT-4 or similar) for NER and summarisation via API is also technically possible and often produces better results on unusual text. The practical blocker is cost: API-based LLMs run 17–27x more expensive than self-hosted models for batch processing, which makes them unsuitable for processing thousands of articles.

### Why spaCy for preprocessing instead of HuggingFace or NLTK

Preprocessing — cleaning text, splitting sentences, tokenising, tagging parts of speech — does not require a transformer model. spaCy's small CPU models do this job accurately and very quickly, processing text in batches via `nlp.pipe()`.

HuggingFace can do preprocessing but it is designed for model inference, not text pipeline utilities. Using it for tokenisation alone is like using a drill to hammer a nail — it works, but it brings unnecessary overhead.

NLTK is the other common choice. It is older, slower, and primarily designed for education and research rather than production pipelines. spaCy processes text up to ten times faster and has a cleaner API for the tasks this project needs.

An important distinction: spaCy can also run NER using transformer models as a backbone. We chose not to use this because it uses more memory than calling HuggingFace directly for the same model, which matters on a 6GB GPU. spaCy handles preprocessing; HuggingFace handles inference. Each tool does what it is best at.

---

## Model choices

### English NER — `dslim/bert-base-NER`

This is a BERT model fine-tuned specifically on the CoNLL-2003 English NER benchmark, which was built from Reuters news articles. Because Wikinews is news text, the training domain matches the inference domain closely. This matters for NER accuracy — a model trained on medical records would perform poorly on news.

The stronger alternative is `dbmdz/bert-large-cased-finetuned-conll03-english`, the large variant. It benchmarks higher but requires roughly twice the VRAM. The base model fits within the 6GB budget and performs well enough for news text.

**What not to use:** `dbmdz/bert-base-german-cased`. Despite appearing in several tutorials alongside NER examples, this is a base language model — it was pre-trained for general language understanding, not fine-tuned for NER. It will not produce entity labels. This was a critical error in the first version of the spec and is worth documenting explicitly.

### German NER — `Davlan/bert-base-multilingual-cased-ner-hrl`

This model is fine-tuned for NER across 10 high-resource languages including German, using the multilingual BERT base. It produces PER, ORG, and LOC labels. It does not produce MISC, unlike the English model. This difference is expected and documented in the entity schema.

The alternative for German specifically is a monolingual German NER model. These can benchmark slightly higher on German text, but they introduce a third dependency and a different label schema. The Davlan multilingual model handles German accurately enough for news NER and keeps the codebase consistent — both languages are loaded through the same pipeline call, only the model name differs.

### Summarisation — `facebook/bart-large-cnn`

BART uses an encoder-decoder architecture and was fine-tuned on CNN/DailyMail news articles. The training domain is news, and the task is news summarisation — this is the closest pre-trained match available without fine-tuning your own model.

The direct competitors are T5 and PEGASUS. T5 is more general-purpose and not specifically trained for summarisation. PEGASUS was designed specifically for summarisation and benchmarks slightly higher on some news datasets, but the HuggingFace implementation, documentation, and community support for BART-CNN is considerably better, and it is the more common choice in production pipelines.

One constraint to be aware of: BART's maximum input is 1024 tokens. Articles longer than this are truncated by the pipeline internally when `truncation=True` is set. Information beyond the first 1024 tokens is lost. For most news articles this is not a problem, but long investigative pieces will have their second half cut.

### Similarity scoring — `sentence-transformers/all-MiniLM-L6-v2`

Similarity scoring works by encoding both texts (the original article and its summary) as dense vectors, then measuring the angle between them — known as cosine similarity. The `all-MiniLM-L6-v2` model is a distilled model trained specifically to produce high-quality sentence embeddings in a small footprint.

The main trade-off is the 256-token input limit. For long articles, only the first 256 tokens are encoded, meaning the similarity score compares the summary against the article's opening rather than the full text. This is a documented limitation and should be stated in any presentation of the results.

The alternative that removes this constraint is `sentence-transformers/all-mpnet-base-v2`, which supports 512 tokens and produces higher quality embeddings. It was not chosen here due to higher VRAM usage. If hardware is not a constraint, mpnet is the better model for this task.

A note on the scores: cosine similarity mathematically ranges from -1 to 1. For this model specifically, embeddings are L2-normalised during training, which means in practice scores land in the range [0, 1]. This is a property of this model, not cosine similarity in general.

### Topic prediction — `facebook/bart-large-mnli`

Zero-shot classification lets you assign a text to a category from a predefined list without training a classifier on labelled examples. BART-MNLI achieves this by rephrasing classification as a natural language inference task: given "This article is about sports", does the article text support or contradict that statement?

The alternative with higher benchmark accuracy is `cross-encoder/nli-deberta-v3-large`. It was not selected because it is larger, slower, and the accuracy gain does not justify the additional VRAM on this hardware. BART-MNLI is the standard starting point for zero-shot classification in 2024–25.

One important limitation to document: this model is English-only. Running it on German text will produce unreliable results. The pipeline enforces English-only input for topic prediction.

---

## Code quality tooling

### Why pytest

pytest is the standard Python testing framework. The alternative (unittest) is built into Python's standard library but has a more verbose API and is less flexible for parameterised tests and fixtures. All NLP projects of any size use pytest.

### Why ruff and black together

ruff is a linter — it checks for code style problems, import ordering, and common errors. black is a formatter — it rewrites your code to a consistent style. They do different jobs and work together.

ruff replaced flake8 and isort because it runs roughly 10–100x faster and handles both tasks in one tool. black is the standard Python formatter and removes all arguments about indentation, line length, and spacing. Neither requires configuration beyond setting the 88-character line limit to match black's default.

Both are run as pre-commit hooks, meaning they check your code automatically before each commit. This prevents style inconsistencies from entering the codebase.

### Why python-dotenv

API keys, passwords, and tokens should never appear in source code or config files. `python-dotenv` loads values from a `.env` file into environment variables at runtime. The `.env` file is listed in `.gitignore` so it is never committed. A `.env.example` file is committed in its place, showing which variables are needed without revealing their values.

---

## Hardware considerations

This project was designed for a single NVIDIA GTX 1060 GPU with 6GB of VRAM. Every model size and memory management decision reflects this constraint.

The most significant consequence is that models must run sequentially. BERT NER models use approximately 1–2GB of VRAM. BART-large-CNN uses approximately 3–4GB. Running two at the same time would exceed the 6GB budget and crash. The pipeline always loads one model, uses it, explicitly deletes it, clears the GPU cache, and then loads the next.

`torch.cuda.empty_cache()` alone is not sufficient to free memory. Python's reference counter must also be satisfied. The correct sequence is `del model` followed by `gc.collect()` followed by `torch.cuda.empty_cache()`. Skipping any step may leave tensors resident in VRAM and cause an out-of-memory error on the next model load.

spaCy models are an intentional exception to the one-model rule. They run on CPU and consume no GPU VRAM, so both the English and German spaCy models can be cached simultaneously without conflict.

---

## Decisions that were explicitly not made

**No model fine-tuning.** All models are used off-the-shelf. Fine-tuning on Wikinews data would likely improve NER and summarisation accuracy, but it requires labelled training data and significantly more compute time. It is the obvious next step if this pipeline were taken into production.

**No entity canonicalisation.** "USA", "U.S.", and "United States" are counted as three separate entities. Resolving aliases requires either a knowledge base (Wikidata, Freebase) or a dedicated entity linking model. It is out of scope here but would be essential in a real business intelligence context.

**No streaming or incremental processing.** The full filtered dataset is loaded into memory. For the article counts in this project this is fine, but a production pipeline processing millions of articles would need chunked reading, generator-based processing, and a persistent storage layer between steps.

**German summarisation.** Summarisation runs on English articles only. Suitable multilingual summarisation models (mBART, mT5) exist but are significantly larger and would not fit in 6GB of VRAM alongside the other pipeline components. This is a hardware constraint, not a technical impossibility.
