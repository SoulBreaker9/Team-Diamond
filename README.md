# Team Diamond

**Amazon ML Challenge 2026** — multi-source business entity resolution.

Given a reference business record from S1, identify the corresponding record(s)
in S2 and/or S3. Scored by **entity-level macro $F_{0.5}$**, which weights
precision more heavily than recall.

> **Status: scaffold.** The evaluation metric, dataset loading, and text
> normalisation are implemented and tested. Retrieval, features, the matcher,
> and the decision layer are declared boundaries with no implementation.
> **Nothing here has produced a competition score.**

---

## Read these first

The agent and team context is layered. Know which layer you are reading.

| Layer | File | What it is | Authority |
|:--|:--|:--|:--|
| Rules | [`AGENTS.md`](AGENTS.md) | engineering contract, invariants, workflow | **OBEY** |
| Strategy | [`suggestion.md`](suggestion.md) | current thinking, hypotheses, priorities | **VERIFY BEFORE TREATING AS FACT** |
| Reference | [`DOCS/`](DOCS/) | glossary, architecture, dataset facts, experiment method | **LOOK UP** |
| Evidence | [`experiments/registry.csv`](experiments/registry.csv) | measured results | **TRUST, if it exists** |
| Code | `src/` | the implementation | **THE TRUTH** |

Every substantive claim carries an evidence status: **VERIFIED**, **MEASURED**,
**HYPOTHESIS**, **PROPOSAL**, or **PRELIMINARY**. A number without a status is
not a result.

---

## Setup

Requires **Python 3.12** and **uv**. The dataset is *not* in this repository.

```bash
uv sync                       # create/sync .venv from uv.lock

export TEAM_DIAMOND_DATA_ROOT=/path/to/dataset   # contains train/ and test/
```

The dataset is located at run time and **verified to exist**. A missing dataset
raises `DatasetNotFoundError` listing every path tried — it never returns a
plausible-looking nonexistent path.

```bash
uv run pytest                 # 77 tests
```

---

## Layout

```text
src/team_diamond/
    data/            locate + load the dataset, enforce schema
    preprocessing/   null-safe name/address normalisation
    indexing/        [ ] build multi-view indexes
    retrieval/       [ ] candidate generation
    features/        [ ] pair feature vectors
    training/        [ ] fit + calibrate the matcher
    models/          [ ] model definitions
    decision/        [ ] how many ids to emit
    evaluation/      entity-level macro F0.5   <- implemented
    pipeline/        [ ] end-to-end wiring

configs/            experiment parameters (null = not yet measured)
tests/              pytest suite
experiments/        registry.csv — the measured-evidence record
```

`[ ]` = declared boundary, not implemented.

The dependency direction is fixed. Higher layers must not secretly modify
lower-layer behaviour — see `AGENTS.md` §7.

---

## Three questions, three owners

The most common confusion in this problem is collapsing these:

| Question | Layer | Output |
|:--|:--|:--|
| *Could* this candidate be the match? | retrieval | a candidate set |
| *How likely* is it to be the match? | models | a calibrated probability |
| *Should* we emit it? | decision | a final id list |

A good matcher over a candidate set that never contained the true match is
still wrong. **Measure retrieval recall before tuning the matcher.**

---

## Measured facts

Reproduce with `suggestion.md` Appendix C. Full detail in
[`DOCS/competition_notes.md`](DOCS/competition_notes.md).

| Fact | Value | Source |
|:--|:--|:--|
| Train entities (S1) | 2,206,821 | E000 |
| Test entities (S1) | **1,732,544** | E001 |
| Mean matches per entity | 3.46125 | E002 |
| True singletons | 123,247 (5.58%) | E002 |
| Ground truth | strict partial matching (0 shared vendor ids) | E003 |
| Test S1 country mix | France 14.98% / India 46.75% / US 38.27% | E004 |
| Missing addresses | S1 0.00% / S2 3.36% / S3 3.33% | — |

Three consequences that shape the design:

1. **France appears only in test.** No hardcoded country branches
   (`AGENTS.md` §8.5). Country artefacts belong in normalisation or features.
2. **The ground truth is a strict partial matching** — every vendor id is
   claimed by at most one S1 entity. Global 1:1 enforcement is therefore safe.
3. **5.58% of entities are true singletons.** The system must be able to emit
   an empty list, and that case is a *success* when predicted correctly.

> The documented test S1 count (1,732,545) is **wrong** — it is a `wc -l` line
> count including the header. Never hardcode a row count; read the ids from the
> file (`AGENTS.md` §17).

---

## Before you change anything

```text
Understand  ->  Measure  ->  Modify  ->  Test  ->  Inspect diff
```

not

```text
Rewrite  ->  Hope
```

- Measure retrieval quality before optimising the matcher.
- Optimise entity-level $F_{0.5}$. Pair-level precision/recall are **diagnostic
  only** — a change that improves them while worsening $F_{0.5}$ is not an
  improvement.
- Never present a HYPOTHESIS or PROPOSAL as a measured result.
- Do not `pip install` in a notebook. Update `pyproject.toml` and run `uv sync`.
- Ask before anything that would materially change the methodology, or whose
  competition compliance is unresolved.
