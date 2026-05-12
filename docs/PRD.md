# PRD — Wikinews NLP Analysis Pipeline

**Project:** Wikinews NLP Analysis  
**Role:** Data Scientist  
**Status:** Planning  
**Last updated:** 2026-05-08

---

## What this project is

This project builds a reusable Python pipeline that reads news articles from the Wikinews dataset, extracts named entities from them, summarises selected articles, and measures how much information the summaries retain. The analysis is scoped by selected countries and topics. The pipeline works on English and German articles for NER, while summarisation and similarity scoring run on English articles only. All analysis is driven by a config file, so countries, topics, languages, models, and thresholds can be changed without touching any code.

The final deliverable is a Jupyter notebook that presents the analysis results, backed by a set of Python modules that do the actual work.

---

## Why it exists

News articles are long. A company monitoring competitor activity, political developments, or market trends cannot read every article manually. This pipeline automates three things that reduce that effort:

1. It finds and counts which entities (people, organisations, places) appear most often across a set of articles, and shows how that changes over time.
2. It shortens long articles into summaries that preserve the key facts.
3. It checks whether those summaries are good enough to trust, using a similarity score.

---

## What the pipeline does — in plain English

**Step 1 — Data loading and exploration**  
Download the Wikinews dataset from GitHub if it is not already present. Inspect the raw data to identify available category labels and article counts. Because Wikinews stores both topical labels and country labels in the `categories` field, the notebook should show counts that help the user choose viable topics and countries. The workflow is non-interactive: review the inspection table, update `config.yaml` with the chosen countries and topics, then rerun/proceed through the notebook.

**Step 2 — Preprocessing**  
Clean the raw article text (remove markup, extra whitespace, URLs). Tokenise each article into sentences and words. Tag each word with its part of speech. Store the processed result alongside the original article and its metadata (title, date, language, country, topic, event ID).

**Step 3 — Named Entity Recognition (NER)**  
Run NER on the preprocessed articles. Use `dslim/bert-base-NER` via HuggingFace for English articles and `Davlan/bert-base-multilingual-cased-ner-hrl` for German articles. Record each entity found: its text, its type (person, organisation, location, etc.), the article it came from, and the article's country, topic, language, and date. Then analyse the results in two ways: an aggregated count of which entities appear most often, and a view of how entity frequency changes over time. Separately, investigate likely German NER errors using low-confidence entity predictions; these cases should be collected and shown as candidates for manual review.

**Step 4 — Summarisation**  
For each selected country-topic group, take 10 to 20 English articles and summarise them using `facebook/bart-large-cnn` via HuggingFace. Each summary goes into the article result data alongside the original text and metadata. The notebook should display summary results and a lightweight grammar/style quality table.

**Step 5 — Similarity scoring**  
For each original–summary pair, calculate a similarity score using sentence embeddings. Scores above 0.8 are considered acceptable for business use. Visualise the score distribution per country-topic group. Explain in the notebook which articles got the lowest scores and why.

**Step 6 — Topic prediction**  
Take a sample of articles whose topic label has been hidden. Use a zero-shot classifier to predict the most likely topic. Compare predictions against the real labels and report accuracy.

---

## What is out of scope

- No model fine-tuning. All models are used as pre-trained, off-the-shelf.
- No web scraping. The dataset is downloaded from the existing GitHub repository.
- Summarisation runs on English articles only. German summarisation is not included because suitable multilingual summarisation models are too large for the available hardware.
- No real-time or streaming processing. This is a batch pipeline.
- No deployment, API, or web interface.
- No interactive topic/country picker in the application. Selection is made by editing `config.yaml` after reviewing inspection output.

---

## Constraints

- Hardware: NVIDIA GTX 1060 GPU (6GB VRAM). Models run sequentially, not simultaneously.
- The raw dataset is modest enough to load for inspection (~15,000 articles). Model-heavy downstream analysis is limited to the selected country-topic-language subset.
- The pipeline must be reproducible. Running it from scratch on a new machine should work by installing requirements and running the notebook.
- Code must follow PEP8 standards and be organised into functions or classes.

---

## How success is measured (evaluation criteria from brief)

| Criterion | What it means in this project |
|---|---|
| NER framework applied on pre-processed data | HuggingFace NER pipeline runs on cleaned text and returns entities with type and article metadata |
| NER analysis includes at least two criteria | Aggregated entity frequency + dynamic over time (both required) |
| Similarity scores visualised and explained | Distribution plot per country-topic group, written explanation of high/low-scoring articles |
| Topic predictions are logical | Zero-shot classifier predictions match or are close to real topic labels |
| PEP8, functions/OOP | All `src/` modules follow PEP8; logic is in functions, not inline in the notebook |

---

## Assumptions

- The dataset's format and field names are detected programmatically. For the target Wikinews dataset, the expected raw fields include `title`, `pageid`, `categories`, `lang`, `url`, `text`, `date`, and `type`.
- The user selects countries and topics after running the data inspection step and reviewing category-count output. The chosen values are then written to `config.yaml`; the notebook itself remains non-interactive.
- Country filtering uses Wikinews `categories` tags, not geocoding or country inference from article text.
- English articles are used for summarisation and similarity scoring. German articles are used for NER only. This is a hardware constraint and is documented in the notebook.
- GPU memory is managed by running models one at a time and clearing cache between steps.
