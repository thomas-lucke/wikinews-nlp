# AI Engineering Field Guide
### How to build software with AI agents — practical knowledge for people starting out

*Built from real project work, May 2026*

---

## The core mindset shift

The most important thing to understand before anything else: when you code with AI agents, **the specification is the product, not the code.**

When a human writes code, ambiguity gets resolved in their head. They know what you meant even when you were vague. When an AI agent writes code, ambiguity becomes a bug you find in production — or worse, code that looks correct but does the wrong thing quietly.

This changes everything about how you work. Most of your time and skill goes into the work that happens *before* the agent writes a single line. The agent's output quality is a direct reflection of your input quality. Vague spec in, broken code out. Precise spec in, working code out.

A secondary shift: experienced developers who use agents well do not "vibe code" — they do not just describe a goal and accept whatever comes back. They retain control of design decisions, review every step, and treat the agent as a very fast, very literal implementation partner that needs exact instructions.

---

## The 6-phase workflow

This is the order of operations for any project that will be built by an AI agent. Do not skip phases or reorder them.

### Phase 1 — Understand the project

Before writing anything down, be able to explain in plain language:
- What the software does
- What goes in, what comes out
- Where the hard parts are
- What you do not know yet

This sounds obvious but most people skip it. If you cannot explain it simply, you are not ready to write a spec. The spec will inherit your confusion.

### Phase 2 — Write the PRD and SPEC

Two separate documents, two separate purposes.

**The PRD** (Product Requirements Document) answers: *what are we building and why?* Written for humans. A non-technical manager could read it and understand the project. It covers goals, scope, what is out of scope, constraints, and how success is measured. No code, no function signatures, no technical implementation detail.

**The SPEC** answers: *exactly how does it work?* Written precisely enough that an AI agent can generate code from it without guessing. Every module, every function signature, every config field, every input and output is defined. If you find yourself writing "something like" or "approximately" — stop and make it exact.

A spec is not just a detailed PRD. Even a more structured prompt with explicit technical constraints produces better code than a plain PRD. The spec is closer to a programming interface than a planning document.

Save both as `.md` files in your `docs/` folder. These are your source of truth. Everything else — the code, the tests, the notebook — should match what is in these files.

**Write the schema first.** Before specifying any module, define the data structures that flow between them. What fields does an article have after each processing stage? What are their types? Which are optional? Lock this down before writing a single function signature. Many first-pass spec errors come from writing the data schema and the functions simultaneously — the schema says `label` but the function produces `entity_group`, and neither section catches it because they were written without checking each other.

**Simulate execution while writing each module.** After writing a function's spec, trace one real item through it mentally. What does the caller pass in? What does this function return? Does that match what the next function expects? The gap between "what a function outputs" and "what the next function reads" is where most silent bugs live, and they are invisible unless you explicitly trace the data flow.

**Do a cross-cutting concerns pass before calling the spec done.** There are five things that touch every module and are therefore the most likely to be inconsistently specified. Before finishing, go through each one as a dedicated read:

- **Logging** — does every module have a `logger = logging.getLogger(__name__)`? Is `print()` used anywhere in `src/`?
- **Error handling** — for every loop over records, is there a try/except? Are there any bare raises that should be caught per-record instead?
- **Paths** — is every path constructed with `pathlib.Path`? Is `encoding="utf-8"` specified on every file open?
- **Config** — is every hardcoded value actually in `config.yaml`? Search for number literals and string constants in the module specs.
- **Resource cleanup** — is cleanup called after every model or connection? Is the cleanup pattern correct (not just `empty_cache()` but `del` first)?

Each of these takes ten minutes and catches entire categories of errors before any review happens.

**Write test fixtures before module specs.** Before writing the spec for `run_ner`, write out what `mock_ner_pipeline.return_value` looks like. That forces you to look up what HuggingFace actually returns, which is exactly where key-rename bugs live. Before writing `calculate_similarity`, write the mock embedding return value — which is how you catch tensor shape issues. Writing mocks first defines the exact data shape of every external dependency before specifying how your code uses it.

### Phase 3 — The anti-ambiguity pass

Feed the spec to an AI with a hostile prompt before any code is written. The goal is to find every hole in the spec while fixing it is cheap, not after 300 lines of code have been built around a wrong assumption.

**The hostile reviewer prompt** (this exact framing works):

> *"You are a senior engineer doing a hostile review of this spec before any code is written. Your job is to find every ambiguity, every assumption stated as fact, every unjustified decision, every edge case not handled, and every place where a coding agent given only this spec would have to guess. Do not summarise what the spec does. Only report problems, numbered."*

Also ask separately: *"What architectural decisions will you make while implementing this, and why?"* Force the model to declare its assumptions before it acts on them. This is how you catch bad choices before they are baked into 500 lines of generated code.

Revise the spec based on the findings. Repeat until the reviewer finds nothing significant. Only then move forward.

**Run three focused passes instead of one general one.** A single "find all problems" prompt spreads attention across everything and often produces long lists where critical bugs sit next to style preferences. Three shorter passes with specific mandates produce more targeted findings:

*Pass 1 — Data boundary check:*
> *"For every place in this spec where data crosses a module boundary — where one function's output becomes another function's input — verify that the shape, type, and field names match exactly. List every mismatch or unresolved ambiguity."*

*Pass 2 — Assumption hunt:*
> *"Find every place where this spec states something as fact that is actually an assumption about the environment, the data source, the library, or the user's behaviour. For each one, describe what would break if the assumption were wrong."*

*Pass 3 — Execution trace:*
> *"Trace one record through the entire pipeline from raw input to final output. List every field that is read, every field that is written, and every place where a field might not exist when it is expected to."*

Each pass takes about ten minutes. Between them they catch the majority of what a full hostile review finds, and the findings are easier to triage because they are already categorised.

**What Phase 3 cannot catch.** Some errors require runtime knowledge — knowing that `random.Random(seed).sample()` on a re-instantiated object replays the same sequence, or that `len(MagicMock())` raises `TypeError`. These are caught by running the code, or by a reviewer who has personally hit those bugs. No spec review, however thorough, eliminates them entirely. The goal of Phase 3 is to reduce the number of review rounds from five to two, and to reserve those rounds for edge cases that only emerge from adversarial scrutiny — not for basic consistency errors that the writing habits above would have caught.

**Scope re-runs of Phase 3 by the type of change.** Phase 4 will surface gaps and you will loop back to fix the spec (see the Bucket A/B/C triage in Phase 4). When you do, do not default to running the full three-pass review on the whole spec — that costs enough that you will stop reading the output and start skipping the review. Scope the re-review by what changed:

- **Architectural change** — anything that touches a function signature, a config key, a data shape, or an error contract. Run the full three-pass review on the whole spec. The change can ripple across modules in ways you cannot predict.
- **Narrowing edit** — pinning an ambiguous value, defining an edge case, clarifying a comparison rule. Run a scoped review on the changed sections plus their direct callers, not the whole spec. The risk you are checking for is "does this narrowing contradict an adjacent rule I did not change?" — not "did I redesign the system?"

When in doubt, scope broader. But the goal is to keep Phase 3 a tool you actually use, not a ritual you start skipping because it is too expensive.

### Phase 4 — Write tests first

Define what "correct" looks like before the agent writes anything. This is called test-driven development (TDD) and it works especially well with AI agents because it gives the agent an objective, checkable finish line.

Instead of saying "write me the code for X", you say "make these tests pass." The difference:
- "Write me the code" → the agent decides what correct looks like
- "Make these tests pass" → you decide what correct looks like

Each test should cover one small, isolated requirement from your spec. If a test is hard to write, that usually means the spec is still ambiguous — go back and fix it.

This is also how you catch hallucinations before they reach production. An agent can produce code that looks plausible but does the wrong thing. Tests catch this before you find out the hard way.

*Prompt for tests from spec alone, in a separate context.* Give the agent only the spec section for the module — not the code, not surrounding context. Frame the task adversarially: the agent's job is not to confirm the implementation but to challenge it. Require it to cite the exact spec line for each test, and to flag spec gaps rather than invent assumptions to fill them. A test that cannot point to a specific spec requirement is not a test — it is a guess.

*The loop between Phase 3 and Phase 4 is normal — plan for it.* Phase 3 reduces review rounds from five to two; it does not eliminate them. The tests you write here will surface more gaps. When that happens, pause test writing, loop back to Phase 2 to fix the spec, and only then resume. Tests that paper over ambiguity by guessing will be wrong in the same direction as the eventual implementation, which defeats the purpose of TDD.

*Triage every flagged gap into one of three buckets.* A test file that ships with a non-empty "SPEC GAPS" list is a signal that the spec is not done yet. Before moving to Phase 5, resolve or document every entry:

- **Bucket A — fix the spec, then update the test.** A real ambiguity where the agent would have to guess at code time (e.g., "what value does accuracy take when the denominator is zero?"). Resolve it in the spec, update the test to assert the resolved value, commit both.
- **Bucket B — write a better test, no spec change.** The spec is fine; the test was weak. Tighten the assertion (e.g., write a test that exercises a deterministic tie-break the spec already defines).
- **Bucket C — document as intentionally not unit-tested.** Some functions are not worth automated coverage — plotting, environment-dependent fallbacks, output that requires visual review. Note this next to the function in the spec, or as an ADR. The gap then becomes an explicit scope decision rather than an unresolved question.

This triage is also a cross-check on your hostile review. Bucket A items are the gaps Phase 3 *should* have caught. If the same category keeps reappearing in Phase 4, your Phase 3 prompts need sharpening.

### Phase 5 — Generate code one task at a time

Break the spec into a numbered task list. Small, isolated steps, each with a clear acceptance criterion. Then give the agent one task at a time — not the whole list.

For each task:
1. Give it the task description
2. Give it the relevant section of the spec
3. Give it the tests it needs to pass
4. Review the output before moving to the next task

Never say "do everything in the task list." Agents working on unbounded tasks make architectural decisions you did not ask for, drift from the spec, and produce changes that are hard to review because they are too large.

If a task turns out to be more complex than expected — stop, update the task list with sub-tasks, then continue. You are the driver. The agent is the fast typist.

### Phase 6 — Review, update spec, ship

Run all tests. Check that the code matches the spec, not just that it passes tests. Tests verify behaviour; the spec also captures intent and structure.

When anything changes during implementation — a different data structure, a library swap, a feature cut — go back and update the spec. The spec should always reflect what was actually built. It is a living document, not a snapshot from day one.

---

## What makes a spec good enough to generate code from

A useful checklist before handing a spec to an agent:

- Every function has a name, argument types, return type, and a description of what it does
- Every data structure has all its fields defined with types
- Every config option is listed with its type, default value, and what it controls
- Edge cases are handled explicitly (what happens if a field is missing? what if a list is empty?)
- Nothing says "something like" or "for example" where a precise value is needed
- The order of operations is unambiguous
- Error handling is specified (does it raise? log? return None? return a default?)
- If two modules interact, the exact shape of what passes between them is defined

A good test: paste only the spec into a fresh conversation with no context and ask "what would you build from this?" If the answer surprises you, the spec is not specific enough.

---

## Documentation in a professional environment

### Git is already your version history — use it

Never save `SPEC_v1.md`, `SPEC_v2.md`, `SPEC_final.md`. That is the amateur approach. Save one `SPEC.md`, update it in place, and commit every meaningful change with a clear message. Anyone on the team can run `git log docs/SPEC.md` to see what changed and when. Git handles version history for documents the same way it handles version history for code.

### Three layers of documentation

Every professional project has three layers. They answer different questions.

**Layer 1 — CHANGELOG.md** (root of the repo)

Answers: *what changed in the project, and when?*

Written for humans, updated manually when you finish a significant piece of work. Follow the "Keep a Changelog" format. Sections: `Added`, `Changed`, `Fixed`, `Removed`. Use semantic versioning for the headers.

```markdown
## [0.2.0] - 2026-05-08
### Added
- data_inspector.py: format detection, profiling, validation
- data_normalizer.py: dataset-agnostic field mapping

### Changed
- data_loader.py now has a single responsibility (download only)
- SPEC.md updated to reflect new three-module data pipeline
```

**Layer 2 — PRD.md and SPEC.md** (docs/ folder)

Answers: *what are we building and how does it work?*

Updated in place. Git history shows old versions. One file per document — never duplicate them.

**Layer 3 — Architecture Decision Records** (docs/decisions/ folder)

Answers: *why did we make this choice, and what did we consider instead?*

This is the most important layer and the most overlooked. An ADR (Architecture Decision Record) is a short document — one page maximum — that captures a single significant decision. Every time you make a meaningful technical choice, write an ADR.

What counts as a meaningful choice:
- Which framework or library to use (and why, and what you rejected)
- A design pattern you chose over an alternative
- A constraint you accepted (e.g. English-only summarisation due to hardware limits)
- A scope decision (what you decided not to build, and why)

The standard template:

```markdown
# ADR-0001: Use HuggingFace directly for NER

## Status
Accepted

## Date
2026-05-08

## Context
The project needs NER on English and German articles. Three options were
considered: spaCy, Flair, and HuggingFace transformers directly.

## Decision
Use HuggingFace pipeline() for both languages with different models per
language. Use spaCy only for preprocessing.

## Alternatives considered
- spaCy with transformer backbone: adds memory overhead on a 6GB GPU
- Flair: research-oriented, being replaced by transformer-native approaches

## Consequences
- Consistent API regardless of language — only the model name changes
- Easier to swap models later without changing calling code
- No Flair dependency needed
```

**Why ADRs matter:** six months from now, a teammate (or you) will look at the code and wonder why it is structured the way it is. Without ADRs, the answer is "I don't remember." With ADRs, the answer is a one-minute read. They are also invaluable during onboarding — a new person can read the decisions folder and understand the project's thinking history in an hour.

### Folder structure for documentation

```
project/
├── docs/
│   ├── PRD.md
│   ├── SPEC.md
│   ├── CHANGELOG.md
│   └── decisions/
│       ├── 0001-huggingface-for-ner.md
│       ├── 0002-english-only-summarization.md
│       ├── 0003-dataset-agnostic-pipeline.md
│       └── 0004-sequential-gpu-loading.md
├── src/
├── tests/
├── config/
└── notebooks/
```

---

## Using LLMs to review your own work — two different jobs

This is worth understanding clearly because the right tool depends entirely on what you are trying to do.

### Job 1 — Technology validation

*"Are these the right tools? Is anything outdated? Is there something better?"*

**Use the web interface** (claude.ai, ChatGPT, Perplexity).

Why: the model needs to search the web to answer this well. Library recommendations, benchmark comparisons, deprecation notices — this information changes and the model's training data has a cutoff. Web search gives it current information. Without it, you might get a confident recommendation for a library that was superseded eight months ago.

Good questions for this job:
- "Is `dslim/bert-base-NER` still the recommended model for English NER, or has something better come out?"
- "Compare the current state of BART vs PEGASUS vs T5 for news summarisation"
- "What are the current best practices for GPU memory management in HuggingFace inference?"

Do this when you are making technology choices, and revisit it if the project runs for more than a few months.

### Job 2 — Spec integrity review

*"Is this document complete, unambiguous, and consistent? Can an agent follow it without guessing?"*

**Use a script with the API** — not the web interface.

Why: this task requires no external information. It is pure reasoning over a document you already have. What you need is:
- A fixed, hostile system prompt that does not change between runs
- Extended thinking enabled so the model works through the spec thoroughly
- Reproducibility — the same spec should produce the same quality of review every time
- Automation — the review should run automatically when the spec changes, not only when you remember to do it manually

A basic implementation:

```python
import anthropic

REVIEWER_PROMPT = """You are a senior engineer doing a hostile review of a 
technical spec before any code is written. Find every ambiguity, every 
assumption stated as fact, every unjustified decision, every edge case not 
handled, and every place where a coding agent given only this spec would have 
to guess. Do not summarise what the spec does. Report problems only, numbered."""

def review_spec(spec_path: str) -> str:
    client = anthropic.Anthropic()
    
    with open(spec_path, "r") as f:
        spec_text = f.read()
    
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000   # how much reasoning to allow before answering
        },
        system=REVIEWER_PROMPT,
        messages=[{"role": "user", "content": spec_text}]
    )
    
    # Return only the text blocks, not the thinking blocks
    return "\n".join(
        block.text for block in response.content 
        if block.type == "text"
    )

if __name__ == "__main__":
    findings = review_spec("docs/SPEC.md")
    print(findings)
```

Save this as `scripts/review_spec.py`. Run it whenever you update the spec. The `budget_tokens` parameter controls how much reasoning the model does before answering — higher means more thorough, higher cost.

The professional version of this runs automatically in CI/CD: whenever you push a change to `docs/SPEC.md`, the review script fires and posts findings as a comment on the pull request. You have a quality gate on your documentation, not just your code.

**Why the model and the interface give different results here:** extended thinking gives the model a scratchpad to reason through the spec step by step before producing output. The web interface toggles this on or off. The API lets you set exactly how much thinking budget to allocate. For a complex spec, that control produces noticeably more thorough findings.

### Using multiple models

Running the same spec through two different models (Claude and GPT-4o, for example) is not overkill for important specs. Different models have different training data and different tendencies. One might catch an architectural problem the other missed. For a spec that will drive a significant amount of generated code, the cost of two API calls is nothing compared to the cost of discovering a fundamental flaw after the code is written.

---

## Framework vs model — the most misunderstood thing in NLP

Most people new to NLP confuse the framework with the model. Understanding this distinction will save you a lot of confusion.

**A model** is the actual AI brain — the thing that was trained on millions of texts and learned to recognise entities, summarise text, or classify topics. Examples: `bert-base-NER`, `bart-large-cnn`, `all-MiniLM-L6-v2`.

**A framework** is the toolbox that loads the model, feeds text into it, and gives you back results. Examples: spaCy, Flair, HuggingFace Transformers.

The critical point: **spaCy, Flair, and HuggingFace can all load the same underlying transformer model.** If they are running the same model, they produce the same quality output. The quality difference between frameworks comes from what *else* they give you around the model, not from the model itself.

| Framework | What it adds around the model | Best used for |
|---|---|---|
| spaCy | Fast pipeline, tokenisation, POS tagging, dependency parsing, production-grade serving | Preprocessing and anything where speed matters |
| Flair | Sequence labelling architecture, stacked embeddings, confidence scores | Research, was strong before transformers took over |
| HuggingFace | Direct model access, largest model hub, most flexible API, GPU-optimised | Production NLP in 2024/25 — this is where the industry has landed |

In practice for most NLP projects today: use **spaCy for preprocessing** (cleaning text, splitting sentences, tokenising) because it is fast and built for exactly that, and use **HuggingFace directly for the actual model tasks** (NER, summarisation, classification). This combination is what professional teams use.

When a colleague says "we use BERT for NER" — they mean they use a BERT-based model loaded through HuggingFace. The framework question is separate.

---

## The professional data pipeline — never assume the data

The single biggest gap between a student project and a production one is how the data layer is handled. A student assumes the dataset looks a certain way and writes code around that assumption. A professional treats every dataset as unknown until proven otherwise.

### The professional order of operations

**1. Acquire** — download or read the raw files. Nothing else. No parsing, no assumptions.

**2. Detect** — figure out the structure programmatically. What is the file format? What fields exist? What are their types? Write code that *discovers* this rather than *assumes* it.

**3. Profile** — measure the data before touching it. How many records? How many missing a date? How many with text shorter than 100 characters? What languages are actually present? What is the distribution of article lengths? You cannot clean what you have not measured. This step produces a data quality report that a human reviews before anything proceeds.

The quality dimensions to check:
- **Completeness** — are required fields present?
- **Validity** — do fields contain what they claim to? (a "date" field full of strings is a validity problem)
- **Consistency** — is the same concept represented the same way throughout?
- **Uniqueness** — are there duplicates?
- **Encoding** — for multilingual data especially, are there UTF-8 decode errors?

**4. Validate** — check the data against rules you define. Each rule either passes, produces a warning, or produces an error. Log every finding. Do not silently drop records.

**5. Clean** — only after profiling and validation do you fix things. Every record you drop gets logged with a reason. This matters because if your analysis later shows something unexpected, you can check whether the cleaning step accidentally removed records it should have kept.

**6. Normalise** — map whatever field names the raw dataset uses to your own internal schema. This is what makes code reusable across datasets. The raw data might call a field `"body"`, `"content"`, `"article_text"`, or `"text"` — your normaliser maps all of them to the internal field `"text"`. Everything downstream only sees your internal schema, never the raw one.

**7. Report** — before any ML runs, the notebook shows the data quality report: how many records loaded, how many dropped and why, what topics and languages are available. A senior data scientist shows this to stakeholders before claiming any results.

**8. Then → ML pipeline**

### Why normalisation is the key step

The normalisation layer — a mapping from raw field names to your internal schema — is what makes a pipeline truly reusable. Without it, every new dataset requires changes to your processing code. With it, you update one mapping table and everything works.

A `FIELD_MAPPINGS` dict that covers common variations is all you need:

```python
FIELD_MAPPINGS = {
    "text": "text", "body": "text", "content": "text", "article_body": "text",
    "title": "title", "headline": "title",
    "date": "date", "published": "date", "publish_date": "date",
    "language": "language", "lang": "language",
    "topic": "topic", "category": "topic", "section": "topic",
    "id": "id", "article_id": "id", "url": "id",
}
```

For fields not in the mapping, attempt to infer based on content (a long string field with no numbers is probably text, a two-letter lowercase field is probably a language code). Log what was inferred and what could not be mapped.

---

## Project structure for a data science / NLP project

This structure works for most NLP projects and is what reviewers expect to see:

```
project-name/
├── config/
│   └── config.yaml          # All user settings — no hardcoded values in src/
├── src/
│   ├── data_loader.py        # Download only
│   ├── data_inspector.py     # Profile and validate
│   ├── data_normalizer.py    # Map to internal schema
│   ├── preprocessing.py      # Text cleaning, tokenisation
│   └── (task-specific modules)
├── notebooks/
│   └── analysis.ipynb        # Entry point — calls src/ functions, shows results
├── docs/
│   ├── PRD.md
│   ├── SPEC.md
│   ├── CHANGELOG.md
│   └── decisions/
├── tests/
│   └── test_*.py             # One test file per src/ module
├── scripts/
│   └── review_spec.py        # Automated spec review
├── data/
│   ├── raw/                  # Git-ignored
│   └── processed/            # Git-ignored
└── requirements.txt
```

**The key rule about notebooks:** the notebook is a presentation layer, not an engine. No logic lives in notebook cells. All functions live in `src/`. The notebook imports from `src/` and calls functions. This makes the code testable, reusable, and reviewable — none of which is true for logic buried in notebook cells.

**Config over hardcoding:** nothing in `src/` modules should contain hardcoded paths, model names, or settings. All of those live in `config.yaml`. A module receives config values as arguments. This means you can change the model you use, the topics you analyse, or the similarity threshold without touching source code — just edit the YAML file.

---

## GPU memory management in HuggingFace

Practical rules for working with limited VRAM (tested on 6GB):

- Load one model at a time. Never keep two loaded simultaneously if you can avoid it.
- After a pipeline is done, delete the caller's variable first, then clear the cache.
- Run steps sequentially in the notebook, not in parallel.

```python
import gc
import torch

# Correct pattern — must do all three steps in this order:
del pipeline           # removes the caller's reference
gc.collect()           # Python's garbage collector frees the memory
torch.cuda.empty_cache()  # returns freed blocks to CUDA's allocator
```

`torch.cuda.empty_cache()` alone is not enough. Python's reference counter must be satisfied first. Calling `del` inside a helper function only deletes the local parameter binding — the caller's variable still holds a reference and the model stays in VRAM. The `del` must happen in the same scope as the variable, before any cache-clearing call.

Approximate VRAM requirements for common models:
- BERT-base NER models: ~1–2GB
- BART-large-CNN (summarisation): ~3–4GB
- Sentence-transformers (similarity): ~0.5GB

On a 6GB card, NER and summarisation fit individually but not together. Plan your notebook's order of operations around this.

---

## Tools worth knowing about for agent-driven development

The field is moving fast. These are the tools that had emerged as of early 2026 and are worth being aware of:

**For spec-driven development workflows:**
- **GitHub Spec-Kit** — open-source toolkit from GitHub that provides a structured four-phase workflow: spec → plan → tasks → implement. Works with Claude Code, Copilot, and Gemini CLI.
- **BMAD** — a fuller lifecycle framework with separate elicitation and course-correction workflows. Good for complex projects.
- **Amazon Kiro** — enterprise platform with a three-phase workflow: Specify → Plan → Execute. Deep AWS integration.
- **Claude Code** — Anthropic's agentic CLI tool. Works with `CLAUDE.md` files that provide persistent project context.

**For the actual coding:**
- **Cursor** — AI-first code editor. Strong community, fast iteration.
- **Windsurf** — similar to Cursor, with a Memories feature for long-term project context.

**For running the review script:**
- Any CI system (GitHub Actions, GitLab CI) can trigger `scripts/review_spec.py` on pushes to the docs folder.

---

## Things that will trip you up — lessons learned

**"The spec is done" is almost never true on the first pass.** The anti-ambiguity review almost always finds problems. Budget time for two or three revision cycles before you start generating code.

**Write the schema before the modules.** Defining every data structure first, then writing function specs that reference it, catches a whole category of cross-section inconsistencies — the function produces `entity_group`, the schema expects `label`, and neither section catches the mismatch unless you wrote one first and checked the other against it.

**Invisible assumptions are the hardest bugs to catch.** The things you do not know you assumed never appear in the spec as risks — they appear as confident statements. "The GitHub repo uses `main` as the default branch" reads like a fact. A hostile reviewer with the specific mandate to find assumptions will catch it. Self-review almost never does, because you have the same assumptions the spec has.

**Agents make architectural decisions when you leave gaps.** An agent given an incomplete spec will fill the gaps with something — usually something reasonable, often something you would not have chosen. The only way to control this is to leave no gaps.

**Tests before code is not optional.** It feels slower at the start. It is faster overall. The first time an agent generates plausible-looking code that fails your tests and you catch it before it becomes a dependency for five other functions, you will understand why.

**Write test fixtures before module specs, not after.** Defining `mock_ner_pipeline.return_value` forces you to look up what HuggingFace actually returns. That five-minute step catches key-rename bugs, wrong tensor shapes, and missing attributes before the spec is even finished.

**Do not generate too much code in one step.** The bigger the task you hand the agent, the harder the output is to review and the more likely it drifts from the spec. Small tasks, frequent reviews, incremental progress.

**The spec is not done when the code is done.** If the implementation differs from the spec — even in ways that turned out to be better — update the spec. A spec that does not match the code is worse than no spec, because it misleads everyone who reads it afterward.

**Framework choice matters less than model choice.** When evaluating NLP tools, ask "which model?" before "which framework?". The same BERT model through spaCy and through HuggingFace produces the same quality output. The framework is just the wrapper.

**Normalise before you clean.** You cannot write good cleaning code until you know what fields actually exist in the data. Profile first, clean second.

---

## A one-page summary of the whole approach

The specification is the product. Write it before the code, make it precise enough to be unambiguous, review it adversarially, write tests that define correctness, then generate code one small task at a time.

Write the data schema first. Simulate execution through each module as you write it. Check cross-cutting concerns — logging, error handling, paths, config, cleanup — as a dedicated pass before calling the spec done. Write test fixtures before module specs, not after.

Run three focused anti-ambiguity reviews rather than one general one: a data boundary check, an assumption hunt, and an execution trace. Accept that some bugs only appear at runtime, and that the goal is to reduce review rounds from five to two — not to eliminate them entirely.

Document architectural decisions as you make them — not the what, but the why and the alternatives you rejected. Keep docs in the repo alongside the code. Update them when things change.

Use the LLM web interface for tasks that need current information (technology choices, benchmark comparisons). Use the API with a fixed system prompt and extended thinking for tasks that need deep reasoning over documents you already have (spec review, integrity checks).

Treat every dataset as unknown. Profile before cleaning, normalise before processing, log every dropped record with a reason.

Never hardcode settings. Config files make things reusable.

The notebook is a presentation layer. Logic lives in `src/`.

---

*Last updated: 2026-05-08*
