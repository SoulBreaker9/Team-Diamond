# Architecture — Team Diamond

> **Reference material, not rules.** This file is explanatory.
> Binding engineering rules live in [`AGENTS.md`](../AGENTS.md).
> Current strategy, hypotheses and experiments live in [`suggestion.md`](../suggestion.md).

**Status: the layer model and its dependency direction are architectural *invariants*
(binding, `AGENTS.md` §Architecture Invariants). The concrete pipeline design is
PROPOSAL in [`suggestion.md`](../suggestion.md) §4.2 and has not been built.**

This file holds the long-form explanations. The invariants themselves stay in
`AGENTS.md` so they cannot be missed.

---

## Layer model

```text
Raw S1 / S2 / S3
        |
        v
Data Contract
        |
        v
Canonical Record Layer
        |
        v
Multi-View Indexes
        |
        v
Candidate Retrieval
        |
        v
Candidate Union + Deduplication
        |
        v
Pair Feature Generation
        |
        v
Pairwise Match Model
        |
        v
Calibrated Match Probability
        |
        v
Entity-Level Decision Engine
        |
        v
Submission Builder
        |
        v
Official Validator
```

### Ownership of the three questions

These three questions are asked constantly and are constantly confused. Each has a
different owner:

| Question | Layer | Output |
|:--|:--|:--|
| *Could* this candidate be the match? | **Retrieval** | a candidate set |
| *How likely* is this candidate to be the match? | **Matching** | a calibrated probability |
| *Should* we actually emit this candidate? | **Decision** | a final id list |

Retrieval failing is not a matching bug. A good matcher over a set that never contained
the true match is still wrong.

---

## Source layout

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

Notebooks orchestrate, call into this package, visualise, and record findings. They do
not hold reusable implementation. See `AGENTS.md` §Source of Truth for the binding rule.

---

## Retrieval layers available

Multi-view retrieval is the mechanism. The specific views:

```text
Exact normalized fields
        +
Token retrieval
        +
Character n-gram retrieval
        +
BM25
        +
Address-component retrieval
        +
Phonetic retrieval
        +
Optional embeddings
```

Then union, then deduplicate. The binding rule is in `AGENTS.md` §Retrieval Strategy:
**never rely on one blocking key.**

Whether embeddings earn a place is PROPOSAL and compliance-gated
([`suggestion.md`](../suggestion.md) §11.4).

---

## Configuration

Experiment parameters do not live in Python files.

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

The `null` values are deliberate. **They are placeholders, not defaults.** Exact values
must come from an experiment. A cap written into a Python file instead of a config, with
no measurement behind it, is a violation of `AGENTS.md` §Configuration and
§Candidate Caps.

---

## Order of escalation

When deciding how to improve the system, climb this ladder only as far as measurement
demands:

```text
Exact / normalized retrieval
        v
Character/token retrieval
        v
Feature engineering
        v
LightGBM
        v
Decision engine
        v
More advanced retrieval
        v
Embeddings / reranking
```

Each step down must be justified by a recorded experiment. Complexity must earn its
place. See `AGENTS.md` §Do Not Optimize for Complexity for the binding version.
