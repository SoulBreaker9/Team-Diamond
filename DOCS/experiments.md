# Experiments — Team Diamond

> **Reference material, not rules.** This file is explanatory.
> Binding engineering rules live in [`AGENTS.md`](../AGENTS.md).
> Current strategy, hypotheses and experiments live in [`suggestion.md`](../suggestion.md).

**Measured evidence lives in [`experiments/registry.csv`](../experiments/registry.csv),
not in prose.** This file explains how to run and record an experiment. `suggestion.md`
holds the experiment *queue* and its hypotheses; this file holds the *method*.

---

## Reading the registry

### A blank cell means "not measured". It never means zero.

This is the single most important thing to know about the file. `E000`–`E006` are
**data audits**, not pipeline experiments: they established the row counts, the ground-truth
cardinality, and the singleton rate. They never measured candidate recall or $F_{0.5}$,
because no retrieval or matcher existed yet. Those cells are blank on purpose.

Filling them with `0` would assert a measurement that was never taken, which is exactly the
failure mode `AGENTS.md` §Evidence Discipline exists to prevent.

### Writing a row

| Column | Rule |
|:--|:--|
| `experiment_id` | Stable, sequential (`E007`, `E008`, …). Never reused or renumbered. |
| `date` | `YYYY-MM-DD`. |
| `component` | The layer: `retrieval`, `features`, `model`, `decision`, `data-audit`, … |
| `change` | One line: what was different from the baseline. |
| `dataset_split` | The split the number was computed on. A test-set number must say so. |
| `git_commit` | The commit the run was made from. `uncommitted` is acceptable but makes the row provisional. |
| `seed` | Required whenever anything was random. |
| `candidate_recall` | Overall **and** per country in `notes`. See the retrieval gate in `AGENTS.md` §10. |
| `pair_precision` / `pair_recall` | Diagnostic only. Never the headline. |
| `entity_f05` | **The number that decides whether the change was an improvement.** |
| `runtime` / `memory` | Measured, not estimated, if the experiment ran at scale. |
| `notes` | Slices, caveats, and what the result does *not* license you to claim. |

### Two rules that are easy to get wrong

1. **Do not overwrite a row.** Correcting a mistake means adding a new row that supersedes
   the old one and says so. An experiment log that silently edits history is not evidence.
2. **A number without a `git_commit`, a `dataset_split`, and a `seed` (where relevant) is
   provisional**, however confident it looks. See `AGENTS.md` §Reproducibility.

---

## 41. Experiments

Every meaningful experiment should answer a question.

Bad:

```text
Experiment 17
Changed some things
Score improved
```

Good:

```text
Question:
Does character n-gram retrieval improve candidate recall
for noisy business names?

Baseline:
Token retrieval

Change:
Add char 3-5 gram retrieval

Measure:
Candidate recall
Candidate count
Runtime
Entity F0.5

Result:
...
```

Track experiments in:

```text
experiments/registry.csv
```

Useful columns:

```text
experiment_id
date
component
change
dataset_split
git_commit
seed
candidate_recall
pair_precision
pair_recall
entity_f05
runtime
memory
notes
```

---

---

## 43. Ablation Testing

When introducing a feature, measure whether it actually helps.

Example:

```text
Baseline
Baseline + name similarity
Baseline + address similarity
Baseline + rarity
Baseline + retrieval agreement
Baseline + all features
```

Do not claim:

> "Feature X is important"

unless an experiment supports it.

---

---

## 46. Error Analysis

When performance is poor, do not immediately add another model.

First classify failures.

Useful categories:

```text
NAME_NOISE
ADDRESS_NOISE
MISSING_ADDRESS
MULTILINGUAL
TRANSLITERATION
COMMON_NAME
COMMON_ADDRESS
WRONG_RETRIEVAL
FALSE_POSITIVE
FALSE_NEGATIVE
SINGLETON_ERROR
MULTI_MATCH_ERROR
COUNTRY_SHIFT
```

Then identify:

```text
What failed?
Why did it fail?
Which layer caused the failure?
```

The objective is to fix the correct layer.

---
