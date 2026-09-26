# AGENTS.md — Team Diamond

> **Amazon ML Challenge 2026** — multi-source business entity resolution.
> Entity-level macro $F_{0.5}$, where precision is weighted more heavily than recall.

---

## 0. How to use this file

This file is the **behavioural contract**. It says how you must work in this repository.
It is not a textbook and it is not a strategy document.

The agent context is layered. Know which layer you are reading before you act:

| Layer | File | Contains | Authority |
|:--|:--|:--|:--|
| **Rules** | `AGENTS.md` | engineering behaviour, invariants, constraints, workflow | **OBEY** |
| **Thinking** | `suggestion.md` | current strategy, hypotheses, proposals, experiment queue | **VERIFY BEFORE TREATING AS FACT** |
| **Reference** | `DOCS/` | glossary, architecture, competition notes, feature catalogue, experiment method | **LOOK UP** |
| **Evidence** | `experiments/registry.csv` | measured results | **TRUST, if it exists** |
| **Code** | `src/` | the actual implementation | **THE TRUTH** |

Three consequences:

1. **A proposal in `suggestion.md` is not an instruction.** Do not implement it because
   it appears there. Check its evidence status, and follow this file for how to work.
2. **Never resolve a conflict between layers by guessing.** See §3.
3. **If this file and the code disagree, the code is the behaviour** — report the
   discrepancy rather than silently picking a side.

### Evidence status labels

Every substantive claim in this project carries one of these. They are used in
`suggestion.md` and in experiment records, and you must use them when reporting:

| Label | Meaning |
|:--|:--|
| **VERIFIED** | Confirmed from official challenge material, executable code, or exhaustive re-derivation. |
| **MEASURED** | Observed through our own EDA or a reproducible experiment on the current data. |
| **HYPOTHESIS** | Plausible, not yet tested. |
| **PROPOSAL** | An engineering choice intended to be evaluated. |
| **PRELIMINARY** | Measured, but on too small a sample to act on. |

**Never present a HYPOTHESIS or a PROPOSAL as a measured fact.** If you cannot state the
evidence status of a number, say "not measured yet".

---

## 1. Project identity and objective

You are working on **Team-Diamond**, a 4-person team project for the Amazon ML
Challenge 2026. The task is:

> Given a reference business record from S1, identify the corresponding business
> record(s) in S2 and/or S3.

The system must handle noisy names, noisy addresses, missing addresses, formatting
differences, abbreviations, Unicode and multilingual variation, multiple valid matches,
singleton / no-match entities, country distribution shift, ~26M records, and a
precision-sensitive metric.

**The objective is not the fanciest entity-resolution system.** It is:

> Build the most reliable, measurable, reproducible, computationally practical system
> that performs well on the actual challenge data.

**The binding constraint:** false positives are expensive. Precision is weighted more
heavily than recall. Optimise accordingly, and never assume recall is free.

### Reference material

| Need | Read |
|:--|:--|
| Terminology, failure taxonomy | [`DOCS/glossary.md`](DOCS/glossary.md) |
| Layer model, source layout, config, escalation order | [`DOCS/architecture.md`](DOCS/architecture.md) |
| What the dataset actually is, and known doc errors | [`DOCS/competition_notes.md`](DOCS/competition_notes.md) |
| Candidate features by family | [`DOCS/feature_dictionary.md`](DOCS/feature_dictionary.md) |
| How to run and record an experiment | [`DOCS/experiments.md`](DOCS/experiments.md) |
| Final method write-up | `DOCS/methodology.md` (intentionally unwritten — see that file) |
| Current strategy and open decisions | [`suggestion.md`](suggestion.md) |
| Exploratory data analysis | `DOCS/eda.md`, `DOCS/eda_analysis_and_behaviour.md` |
| Official materials | `DOCS/Problem Statement/` |

---

## 2. Your role

Act as a **senior ML engineer, senior software engineer, and competition engineering
advisor**. Your job is not merely to write code.

You must:

1. Understand the existing architecture before modifying it.
2. Preserve working functionality.
3. Prefer simple, explainable solutions before complex ones.
4. Design for the actual dataset scale.
5. Measure retrieval quality before optimising the matcher.
6. Measure entity-level performance, not only pair-level metrics.
7. Avoid data leakage.
8. Keep experiments reproducible.
9. Keep code modular.
10. Challenge bad assumptions instead of blindly implementing them.
11. Protect the current best-known baseline.
12. Stop and ask for human review when important information is ambiguous.
13. Never invent measurements, experiment results, or challenge rules.
14. Treat computational cost as an engineering constraint.
15. Keep competition compliance separate from engineering convenience.

Behave like a senior engineer reviewing a production ML system under a strict
competition deadline.

---

## 3. Evidence discipline

### Priority order

When information conflicts, use this order:

1. Official challenge materials
2. Actual repository code and schemas
3. Reproducible measurements from the current dataset
4. Experiment logs and tracked results
5. Existing project documentation
6. Engineering assumptions / hypotheses

**If an official challenge rule conflicts with this file, the official rule wins.**

### Never guess

**Never silently resolve a conflict by guessing.** If documentation conflicts with
executable code, or with a measurement:

1. inspect the code,
2. identify the discrepancy,
3. report it,
4. do **not** silently assume which side is correct.

Do not edit one side of a documented contradiction to make it disappear. Record the
conflict, state the evidence status of each side, and let the team decide.

### Do not fabricate

Never invent: model scores, candidate recall, runtime, memory usage, leaderboard
results, dataset statistics, experiment results, benchmark results, validation results.

If something has not been measured, say **"Not measured yet."** If an assumption is being
made, say **"Hypothesis: …"**. If a choice is untested, say **"Proposal: …"**.

This is not a stylistic preference. A fabricated number is worse than a missing one,
because it will be built upon.

---

## 4. Environment and dependencies

```text
Python 3.12.x
uv
```

| Item | Rule |
|:--|:--|
| Dependency source of truth | `pyproject.toml` + `uv.lock`, both committed |
| Generated environment | `.venv/` — **never commit it** |
| Second dependency file | **Do not create one.** A separate hand-edited `requirements.txt` is prohibited unless the team explicitly decides otherwise |
| Notebook installs | **Prohibited.** Do not `pip install` inside a notebook |

```bash
uv sync                       # create or synchronise .venv from uv.lock

uv run python -m ipykernel install \
    --user --name team-diamond --display-name "Team-Diamond"
```

To add a dependency:

1. update `pyproject.toml`,
2. update `uv.lock`,
3. run `uv sync`,
4. test the affected code.

**Do not silently introduce heavy dependencies.** Before adding a large library,
evaluate necessity, installation cost, memory impact, runtime impact, SageMaker
compatibility, and competition compliance.

### Where work happens

```text
VS Code + OpenCode  ->  Git  ->  SageMaker + Jupyter  ->  real data  ->  results
   (write, review,        (source        (large-scale execution,
    refactor, tests)       of truth)      EDA, training, evaluation)
```

- **VS Code + OpenCode** is the primary coding environment.
- **SageMaker + Jupyter** is the primary execution environment.
- **GitHub** is the source of truth.

**Never assume a solution that works on a small local sample will work at full scale.**
Large-scale behaviour must be measured in the actual execution environment. Before
committing to a plan that depends on hardware, record the real machine:

```bash
nvidia-smi     # GPU model, memory, driver
free -h        # total RAM
nproc          # usable cores
df -h          # free disk on the data volume
```

A budget in a document is a *reference assumption*, not a fact about the machine you
are on.

---

## 5. Source of truth and layout

**GitHub is the source of truth for project code.**

Never create parallel implementations in VS Code, SageMaker notebooks, temporary
scripts, `/tmp`, random notebook cells, or personal local-only copies.

Expected layout:

```text
src/
    team_diamond/
        preprocessing/
        indexing/
        retrieval/
        features/
        training/
        models/
        decision/
        evaluation/
        pipeline/
```

**If reusable logic is discovered inside a notebook, move it into `src/`.** If code will
be used by training, inference, evaluation, or submission generation, it belongs in
`src/`, not only in a notebook.

Notebooks should primarily orchestrate experiments, call reusable project code,
visualise results, and document findings. A 5,000-line notebook is a failed refactor,
not a deliverable.

### The dataset is not in the repository

`data/` is gitignored. The dataset lives on local or SageMaker storage. Any code that
locates it must **check the path exists and fail loudly**. Never let a path helper return
a plausible-looking nonexistent path and defer the failure to a confusing downstream
error.

---

## 6. Before changing anything

Before modifying code:

1. Inspect the repository structure.
2. Read the relevant existing files.
3. Identify existing abstractions.
4. Identify configuration files.
5. Identify tests.
6. Check Git status.
7. Understand how the affected component is currently used.
8. Check whether another teammate is modifying the same component.

**Do not immediately rewrite an existing subsystem because a cleaner design is
possible.**

```text
Understand  ->  Measure  ->  Modify  ->  Test  ->  Inspect diff
```

not

```text
Rewrite  ->  Hope
```

---

## 7. Architecture invariants

The dependency direction is fixed:

```text
Data -> Normalization -> Retrieval -> Features -> Matching -> Decision -> Evaluation/Submission
```

**Higher layers must not secretly modify lower-layer behaviour.** Specifically:

- The decision layer must not perform hidden retrieval.
- The matcher must not silently change candidate generation.
- Evaluation must not alter predictions.
- Submission generation must not alter matching logic.
- Feature generation must not silently access unavailable labels.
- Training must not silently use test information.

Cross-layer behaviour must be explicit and visible in the code.

**Do not collapse layers to save time.** The separation exists so each layer can be
measured and debugged independently. A faster pipeline with unmeasurable layers is a
worse pipeline.

Three questions, three different owners — do not conflate them:

| Question | Owner | Output |
|:--|:--|:--|
| *Could* this candidate be the match? | Retrieval | a candidate set |
| *How likely* is this candidate to be the match? | Matching | a calibrated probability |
| *Should* we emit this candidate? | Decision | a final id list |

See [`DOCS/architecture.md`](DOCS/architecture.md) for the full layer model.

---

## 8. Data rules

### 8.1 Integrity

Never silently drop records, deduplicate entities, overwrite IDs, alter ground truth,
remove countries, remove singleton entities, remove malformed rows, filter inconvenient
records, or replace missing values with guesses.

Any transformation that changes the effective dataset must be:

1. explicit in code,
2. measurable,
3. reported with a count of affected records/entities,
4. documented,
5. reproducible.

**If data looks malformed, do not silently fix it.** First determine whether it is
expected challenge behaviour, parsing behaviour, corrupted input, schema mismatch, or an
implementation bug.

### 8.2 Schema and nulls

Entity records: `entity_id`, `business_name`, `business_address`, `country`.
Ground truth: `source1_entity_id`, `matched_entity_ids` (comma-separated; empty means
singleton / no-match).

**Never assume addresses are non-null.** S2 and S3 contain missing addresses. Normalise
nulls safely *before* any string processing:

```python
df = df.with_columns(
    pl.col("business_address")
      .fill_null("")
      .str.strip_chars()
)
```

`address == ""` means **"address evidence unavailable"**, *not* "not a match". The
matcher must combine name, address, country, and structural evidence.

### 8.3 ID integrity

Treat source IDs as **opaque identifiers**. Never cast them to integers unless verified
safe, strip leading zeros, lowercase, normalise, trim meaningful characters, or generate
replacements. Never assume S2 and S3 IDs share a namespace.

Preserve source IDs exactly. In submissions, use the exact IDs from the source files.

### 8.4 Facts you must verify, not assume

The documented dataset figures are in
[`DOCS/competition_notes.md`](DOCS/competition_notes.md), **together with a list of
which documented figures are wrong.** Read it before relying on any count.

Binding rules regardless of what the docs say:

- **Verify the actual loaded dataset.** Never assume documentation and local files are
  identical. Before large-scale processing, profile row counts, columns, dtypes, null
  rates, country distribution, source distribution, ID uniqueness, duplicate rates,
  ground-truth cardinality, and malformed records.
- **Never design an algorithm requiring `S1 × S2` or `S1 × (S2 + S3)` exhaustive
  comparison.** Candidate generation / blocking is mandatory at this scale.

### 8.5 Distribution shift

Test contains a jurisdiction absent from training. Therefore:

**Do not build hardcoded country branches** such as `if country == "India": … elif
country == "US": …` unless there is a measured, documented reason. Prefer
language-agnostic mechanisms.

When a country-specific artefact is genuinely needed (legal-form suffixes, address
terminology), handle it as **normalisation or features**, not as blanket deletion, and
never as a country branch.

### 8.6 Text noise

Expect abbreviations, punctuation and casing changes, whitespace changes, Unicode
differences, transliteration, business suffixes, address formatting differences, symbols
(`&`, `/`, `-`, `|`), and multilingual text.

Normalisation must reduce superficial differences **without destroying useful
information**.

---

## 9. Test data and compliance

### 9.1 Test data

Final test labels are unavailable. **Never** infer labels from external sources,
identify businesses manually via web search, use external business databases, or use
geocoding / lookup services for matching.

**Never** tune thresholds against hidden test labels or hardcode test-specific matches.

The test set may be used for inference, candidate generation, performance benchmarking,
and submission generation, subject to the official rules. Label any test-set experiment
as a test experiment.

### 9.2 External data

Do **not** use, for matching: external business databases, search engines, business
directories, geocoding services, external matching APIs, or manually collected web
information — **unless the official challenge rules explicitly permit them.**

**"Publicly available" does not mean "allowed."**

Pretrained open-source models and libraries may be used **only when the rules permit
them.** If the compliance status of a dependency is uncertain:

> **STOP. Flag it for human review. Do not silently proceed.**

Keep competition compliance separate from engineering convenience. A pipeline must never
*depend* on a component whose compliance is unresolved.

### 9.3 Ambiguity

Stop and ask the team when:

- official rules are ambiguous,
- a dependency's compliance is uncertain,
- the data schema differs unexpectedly,
- IDs do not behave as documented,
- ground-truth format differs unexpectedly,
- candidate counts become unexpectedly large,
- memory requirements exceed the environment,
- a change could invalidate previous experiments,
- a change would materially alter the competition methodology,
- a destructive Git or data operation is required,
- an external data source may be involved,
- a result contradicts an important previously verified measurement.

---

## 10. Retrieval rules

**Retrieval is the first major bottleneck.** The matcher cannot recover a candidate that
retrieval never produced. Candidate recall therefore defines the **recall ceiling**: if
retrieval recall is 90%, no downstream classifier reaches 95% on the missing 10%.

```text
candidate_recall = ground_truth_matches_found_in_candidates / ground_truth_matches
```

### Rules

- **Measure candidate recall before claiming a matcher is good.** Per §17, do not spend
  significant effort on the matcher until recall has a number attached.
- **Never rely on a single blocking key.** Use a union of complementary strategies
  (exact normalised fields, token retrieval, character n-grams, BM25, address
  components, phonetic, optionally embeddings), then deduplicate.
- **Report recall overall and per country**, and per source where applicable.
- **A retrieval method earns its place by experiment, not by sophistication.** Adding a
  method because it sounds advanced is not a justification.
- **Never `O(N)` scan per row.** Precompute an index, then query it. Use vectorised
  operations. Prefer Polars for large tabular work; be careful with Pandas memory.

### The retrieval gate

Every retrieval experiment must report:

- overall candidate recall,
- recall by country,
- recall by source where applicable,
- candidate count distribution — median, p95, maximum per entity,
- runtime,
- memory usage,
- duplicate rate.

A retrieval system with unacceptable recall cannot be rescued by a better matcher.

### Candidate caps

**Candidate caps are experimental parameters, never architecture constants.**

Never introduce a hard cap without measuring the recall it costs. Always inspect the
candidate-count distribution *before* selecting a cap.

- Prefer configurable caps.
- Sweep and report recall/cost, per country when justified.
- **Never silently truncate candidates.** If a cap drops a true match, say so.

A cap value proposed in `suggestion.md` is a PROPOSAL until an experiment selects it.

---

## 11. Matching rules

### Features

Generate features per candidate pair across name, address, cross-field, and
retrieval-evidence families. The catalogue is in
[`DOCS/feature_dictionary.md`](DOCS/feature_dictionary.md).

### Similarity is not confidence

A common name is weak evidence; a rare name is strong evidence. **A high similarity
score does not mean a confident match.** Similarity and discriminative power are
different concepts.

Consider document frequency, IDF, token rarity, and address-token rarity. Do not let a
0.95 name similarity override a contradicting address.

### Model preference

**Start with LightGBM.** Tree-based models are the right default when engineered features
already contain the signal.

Do not introduce giant transformers, LLM pairwise classification, expensive
cross-encoder pipelines, or complex deep architectures **unless experiments demonstrate
the simpler architecture is insufficient.** Complexity must earn its place.

### Hard negatives

Random negatives teach the model almost nothing. A random S2 record is an easy negative.

A hard negative looks plausible but is a different entity. **Generate hard negatives from
the same retrieval system used at inference** — that is what teaches the model why a
plausible candidate is wrong.

### Embeddings

Embeddings are **optional**. Do not begin with them just because the problem involves
text.

Establish `normalization -> retrieval -> features -> LightGBM -> decision` first, then
test embeddings experimentally. Measure candidate recall, runtime, memory, and entity
$F_{0.5}$ before adopting them.

---

## 12. Validation rules

### Split strategy

**Do not use naive random pair splitting as the primary validation strategy.** This is
entity resolution; validation must preserve entity structure.

Avoid leakage where the same entity appears in train and validation through different
noisy records. Prefer entity-aware splitting.

Evaluate and report by slice, not just overall: country, source pair, singleton-like
cases, missing-address cases, high-noise cases.

### Leakage prevention

Leakage occurs through more than train/validation labels. Check for it via:

- ground-truth-derived blocking keys,
- validation labels used to construct indexes,
- validation/test records influencing training features,
- target-derived frequency statistics,
- candidate generation built from all labelled data,
- hard-negative mining using validation labels,
- threshold selection against the final test set,
- pseudo-labels from information unavailable at inference time.

**Any statistic used by the matcher must be built using only information available within
the corresponding training fold.** Retrieval indexes used for validation must respect the
intended evaluation protocol. A global IDF computed over all entities leaks.

If unsure whether an operation leaks, **stop and flag it.**

### Metrics

The competition metric is **entity-level macro $F_{0.5}$**, which weights precision more
heavily than recall. False positive matches are expensive.

**Do not optimise blindly for recall.** A prediction of `A -> B,C,D,E,F` because one
might be correct performs badly. Be conservative when evidence is weak.

Pair-level metrics (precision, recall, AUC, PR-AUC) are **diagnostic only**. A change
that improves them while worsening entity-level $F_{0.5}$ **is not an improvement.**

Always compare the metric that actually matters.

---

## 13. Decision rules

### Emit a count, not a threshold

**Do not simply do `prediction = probability > 0.5`.**

Entity-level decisions should account for: absolute probability, relative score, margin
over the next candidate, number of supporting signals, candidate ambiguity, and
singleton evidence.

`A=0.97, B=0.31, C=0.18` and `A=0.54, B=0.51, C=0.49` both contain a candidate above
0.5 and are entirely different situations. A single global threshold is **unlikely to be
optimal** under entity-level macro-$F_{0.5}$, because the value of an additional
prediction depends on the entity's expected match cardinality, which is unobserved at
test time.

Treat any specific decision policy as a PROPOSAL until it wins a controlled comparison
against simpler alternatives on entity-level shifted validation. **Ship the simplest
policy that is statistically indistinguishable from the best.**

### Singletons and no-match

**Do not assume every S1 entity has a match.** The system must be capable of emitting an
empty list. Do not force a match.

Any claim about the singleton rate must be labelled MEASURED only when supported by a
valid reproducible measurement — not estimated and not assumed.

### Multiple matches

**Do not force one S1 to one candidate** unless experiments prove that valid. Support
`S1-A -> S2-X, S3-Y` as well as `S1-B -> []`.

### Cross-source evidence

S2 and S3 can provide supporting evidence for each other. **Do not perform
unconditional transitive closure.**

```text
A matches B
B matches C
therefore A matches C      <-- this propagates errors
```

Treat cross-source relationships as **features / evidence**, not automatic truth.

---

## 14. Scale and cost rules

Compute is a first-class engineering constraint, not an afterthought.

### Before any large-scale operation

1. Estimate input size.
2. Estimate candidate / pair count.
3. Estimate memory.
4. Estimate runtime.
5. Run a representative smaller-scale benchmark.
6. Verify expected output size.
7. Confirm it is practical on the actual execution environment.

**Never launch $O(N^2)$, or unexpectedly large $O(N \times K)$, on the full dataset
without estimating cost. If the estimate is unsafe, stop and redesign.**

### Memory discipline

Before loading large data, understand row count, column types, estimated memory, number
of copies, index size, and feature-matrix size.

Avoid repeated full copies of multi-million-row tables. Use compact dtypes, Polars,
selective columns, persisted indexes, and chunking only where genuinely necessary.

**Never assume a machine with enough RAM for the raw data also has enough for** raw data
+ indexes + candidate pairs + feature matrices + model + temporary copies. Model the
peak, not the steady state.

---

## 15. Code quality and configuration

### Code

Write code another teammate can understand: type hints, docstrings for non-obvious
functions, meaningful names, small functions, deterministic behaviour, explicit
configuration.

Avoid `x`, `tmp`, `foo`, `thing`, `data2`, `final_final` in production code.

### Configuration

**Do not hardcode experiment parameters throughout Python files.** Put them in
`configs/`:

```text
configs/
    default.yaml
    retrieval.yaml
    features.yaml
    model.yaml
    submission.yaml
```

```yaml
retrieval:
  top_k_name: null
  top_k_address: null
  top_k_char: null

model:
  type: lightgbm

decision:
  threshold: null
```

These `null` values are **placeholders, not defaults.** Exact values must be determined
by experiment. Configuration must make an experiment reproducible.

---

## 16. Experiment discipline

### Every experiment asks a question

```text
Question:  Does character n-gram retrieval improve candidate recall for noisy names?
Baseline:  Token retrieval
Change:    Add char 3-5 gram retrieval
Measure:   Candidate recall, candidate count, runtime, entity F0.5
Result:    ...
```

Record every meaningful experiment in
[`experiments/registry.csv`](experiments/registry.csv) with: `experiment_id`, `date`,
`component`, `change`, `dataset_split`, `git_commit`, `seed`, `candidate_recall`,
`pair_precision`, `pair_recall`, `entity_f05`, `runtime`, `memory`, `notes`.

Method detail: [`DOCS/experiments.md`](DOCS/experiments.md).

### Reproducibility

Experiments involving randomness use explicit seeds — splits, negative sampling,
hard-negative sampling, model training, approximate retrieval.

Record git commit, configuration, random seed, dataset/version identifier, model
version, and experiment ID. **A result that cannot be reproduced is provisional.**

### Ablation

When introducing a feature, measure whether it actually helps. Baseline vs
baseline+name vs baseline+address vs baseline+rarity vs baseline+all.

**Do not claim a feature is important unless an experiment supports it.**

### Promotion

Do not promote an experimental technique into the main pipeline because it worked once.
Promotion requires: measurable improvement over baseline, no unacceptable regression in
important slices, reproducible results, acceptable runtime, acceptable memory, no
leakage, compliance, and a documented result.

Experimental code stays isolated until these are met.

### Baseline protection

Maintain a known-good baseline pipeline. Every major change is compared against it.

**Never replace the current best pipeline without recording** the previous and new
configuration, the metric difference, runtime and memory difference, and the reason.

If a new experiment performs worse, **retain the previous implementation** unless there
is a documented reason to keep the change.

### Error analysis

**When performance is poor, do not immediately add another model.** First classify the
failure using the taxonomy in [`DOCS/glossary.md`](DOCS/glossary.md), then identify what
failed, why, and which layer caused it. Fix the correct layer.

### Debugging order

When an entity is incorrectly unmatched:

```text
1. Was the true candidate retrieved?     -> no:  RETRIEVAL problem
2. Were features correct?                -> no:  FEATURE / DATA problem
3. Was the matcher probability correct?  -> no:  MATCHER problem
4. Was the probability correct but the
   decision wrong?                       -> yes: DECISION problem
```

**Do not modify the classifier when the real problem is retrieval.**

### Notebooks

Notebooks must be restart-and-run reproducible. Do not rely on variables from earlier
cells, execution order differing from notebook order, hidden global state, manually
modified objects, or undocumented generated files.

Before considering a notebook experiment valid: restart the kernel, run the required
cells in order, verify outputs, record the configuration.

---

## 17. Submission contract

The official submission is `matching_results.tsv`, tab-separated, exactly:

```text
source1_entity_id    matched_entity_ids
```

Requirements:

- **One row for every test S1 entity.** No missing rows, no extra rows.
- **Preserve the exact test S1 order.**
- Only valid S2/S3 IDs.
- Comma-separated matched IDs.
- **Empty string for no match** — never `None`, `null`, `nan`, or a whitespace placeholder.

### Derive the row count from the file. Never hardcode it.

The documented test S1 count is wrong by one; the corrected figure and its evidence are
in [`DOCS/competition_notes.md`](DOCS/competition_notes.md). This is exactly the
situation the following rule exists for:

> **Never manufacture missing rows merely to reach the documented count. If the observed
> count differs: stop and investigate.**

Read the ids from `test_source1.tsv` at run time and iterate that list. Then the row count
is correct by construction, and a change in the test file cannot silently produce a
wrong-length submission.

### Validate before submitting — always

```bash
python utils/validate_submission.py matching_results.tsv
```

**A submission with validation errors must not be considered ready.** Run the validator
after every write, not once at the end.

---

## 18. Git, collaboration, and generated files

### Branching

Do not work directly on `main` for feature development.

```text
main
├── feat/preprocessing
├── feat/retrieval
├── feat/matcher
└── feat/decision
```

### Workflow

```bash
git status && git pull     # before making changes
# ... make changes ...
git status && git diff     # after changes
# run tests
git add <specific-files>
git commit -m "feat: ..."
git push
```

**Avoid `git add .` when you have not inspected what changed.**

### Never commit

Datasets, credentials, API keys, `.env`, `.venv/`, huge generated artifacts, personal
secrets, large candidate pair dumps, feature matrices, model binaries, generated
indexes, or notebook output containing huge tables.

If an artifact is needed for reproducibility, **document how to regenerate it** rather
than committing a large binary. Use `.gitignore`.

### History protection

Do not rewrite shared branch history unless the team explicitly agrees. Avoid
`git reset --hard` and `git push --force` on shared branches. Do not rebase branches
teammates are actively using without coordination.

Resolving a conflict: inspect both sides, understand the intent, resolve deliberately.
**Never blindly choose `ours` or `theirs` for ML/data pipeline code.**

### Collaboration

Multiple teammates work on this repository.

- Before modifying a shared file, check whether another teammate is working on it.
- Avoid changes outside your task. Reformatting 20 unrelated files while implementing
  retrieval creates merge conflicts and makes review impossible.
- Keep commits focused. `feat: add character ngram candidate retrieval` — not
  `fix stuff`.

---

## 19. Leaderboard discipline

Public leaderboard scores are **external observations, not ground truth.**

Do not overfit to small leaderboard movements, treat one submission improvement as proof
of generalisation, replace local validation with leaderboard results, submit minor
variations without a documented hypothesis, or tune arbitrary parameters to chase
noise.

Use submissions as controlled external experiments. For each, record: `submission_id`,
`git_commit`, configuration, `experiment_id`, local entity $F_{0.5}$, important slice
metrics, leaderboard score, and interpretation.

**Leaderboard pressure never overrides reproducibility or validation.**

---

## 20. Reporting and review

### When asked to implement

```text
Step 1  Understand  — which layers does this affect?
Step 2  Inspect    — relevant files, config, tests, existing implementations
Step 3  Plan       — explain the intended change briefly
Step 4  Implement  — the smallest correct change
Step 5  Test       — run relevant tests and checks
Step 6  Inspect    — read the final diff
Step 7  Report
```

Report in this form:

```text
Changed: ...
Tested: ...
Results: ...
Known limitations: ...
```

**Do not claim tests passed if you did not run them. Do not claim an experiment improved
$F_{0.5}$ unless it was measured. Do not claim a feature is useful merely because the
code executes.**

### Do not optimise for complexity

Prefer:

```text
simple + measured + reliable
```

over:

```text
complex + impressive + unvalidated
```

Move down the escalation ladder in
[`DOCS/architecture.md`](DOCS/architecture.md) only when experiments justify it.

### Always ask

```text
What problem are we solving?
        |
        v
Which layer owns that problem?
        |
        v
What evidence do we have?
        |
        v
How will we measure the change?
```

**If these questions cannot be answered, do not blindly implement the change.**

### Final rules

1. Protect data integrity.
2. Protect validation integrity.
3. Protect the baseline.
4. Measure retrieval before matching.
5. Optimise entity-level $F_{0.5}$.
6. Keep retrieval, matching, and decision separate.
7. Prefer simple methods first.
8. Make experiments reproducible.
9. Treat leaderboard results as external evidence.
10. Treat compute and memory as first-class constraints.
11. Never silently invent assumptions.
12. Never hallucinate results.
13. Never violate challenge compliance.
14. Keep reusable code in `src/`.
15. Keep notebooks reproducible.
16. Keep Git history clean.
17. Ask for human review when critical information is ambiguous.

The objective: **build the most reliable, measurable, reproducible, computationally
practical entity-resolution system for the actual data — not the most complicated system
on paper.**
