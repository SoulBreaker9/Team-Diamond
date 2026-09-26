# Competition Notes — Team Diamond

> **Reference material, not rules.** This file is explanatory.
> Binding engineering rules live in [`AGENTS.md`](../AGENTS.md).
> Current strategy, hypotheses and experiments live in [`suggestion.md`](../suggestion.md).

**This file records what the dataset actually is.** Rules that depend on these facts
stay in `AGENTS.md` as *constraints* ("verify the count from the file", "never assume a
non-null address") so they survive even when a number here changes.

## Corrections applied to this file

The figures and claims below were inherited from `AGENTS.md` §10/§12/§14 and the DOCS
folder. Some were wrong. Corrections are listed here rather than made silently, per
`AGENTS.md` §Evidence Hierarchy.

| Claim as documented | Status | Corrected value / finding |
|:--|:--|:--|
| Test S1 = 1,732,545 | **WRONG** | **1,732,544** data rows. `wc -l` = 1,732,545 including header. Confirmed by running the official validator's own `read_ids()`. |
| Test S2 = 4,887,274 / S3 = 5,082,317 | **WRONG** | 4,887,273 / 5,082,316. Same off-by-one. |
| Test total = 11,702,136 | **WRONG** | 11,702,133. |
| Training counts | **CORRECT** | Already record counts. 14,733,861. |
| "Training ground truth does not contain singletons" | **WRONG** | It contains **123,247 singletons (5.58%)** — measured, uniformly split US 5.58% / India 5.59%. |
| Ground-truth mean matches = 3.66 | **WRONG** | **3.46125** (7,638,365 pairs / 2,206,821). |
| DOCS cardinality table | **WRONG** | Accounts for only 61.2% of entities. The real histogram is in `suggestion.md` §1.1. |
| "S2/S3 contain 50,000+ Indian PIN codes" | **WRONG** | 459 in S2, 410 in S3. `raw_dataset_specification.md` §4 is the source of the error. |
| Test country mix from a 100k prefix | **IMPROVED** | Full files: S1 France 14.98% / India 46.75% / US 38.27%. |
| Ground truth is a strict partial matching | **NEW** | 7,638,365 pairs → 7,638,365 distinct vendor ids, 0 shared. Global 1:1 enforcement is safe. |

The original files (`DOCS/dataset_and_eda_summary.md`, `DOCS/raw_dataset_specification.md`,
`DOCS/eda.md`, `DOCS/eda_analysis_and_behaviour.md`, `AGENTS.md` §10) are **left
unedited** pending a team decision, so the discrepancy stays visible rather than being
patched away on one side only.

---

## Task

## 1. Project Identity

You are working on **Team-Diamond**, a 4-person team project for the **Amazon ML Challenge 2026**.

The project solves a **multi-source business entity resolution / record linkage** problem.

The core task is:

> Given a reference business record from S1, identify the corresponding business record(s) in S2 and/or S3.

The system must handle:

- Noisy business names
- Noisy addresses
- Missing addresses
- Formatting differences
- Abbreviations
- Unicode differences
- Multilingual data
- Multiple valid matches
- Singleton / no-match entities
- Country distribution shift
- Large datasets
- Precision-sensitive evaluation

The primary evaluation metric is **entity-level macro F0.5**, where precision is weighted more heavily than recall.

The objective is **not** to build the fanciest entity-resolution system.

The objective is:

> Build the most reliable, measurable, reproducible, computationally practical system that performs well on the actual challenge data.



#

**Why:** the objective is a *reliable, measurable, reproducible, computationally
practical* system — not the most sophisticated one. Optimise for the system that can be
built, measured, and defended inside the window.

---

## Dataset facts

## 10. Dataset Facts and Assumptions

The current challenge materials specify approximately:

### Training

| Dataset | Records |
|---|---:|
| S1 | 2,206,821 |
| S2 | 5,034,616 |
| S3 | 5,285,603 |
| Ground truth | 2,206,821 |
| Total | 14,733,861 |

### Test

| Dataset | Records (corrected) | As documented |
|---|---:|---:|
| S1 | **1,732,544** | 1,732,545 |
| S2 | **4,887,273** | 4,887,274 |
| S3 | **5,082,316** | 5,082,317 |
| Total | **11,702,133** | 11,702,136 |

The documented test figures are `wc -l` line counts, which include the header row.
The training figures in the same table are true record counts, so the convention is
inconsistent within one table. See `suggestion.md` §1.7.

The combined corpus is therefore approximately **26.4 million records**.

These numbers are documented challenge facts, but the actual loaded dataset must always be verified.

Before large-scale processing, profile:

- row counts,
- columns,
- data types,
- null rates,
- country distribution,
- source distribution,
- ID uniqueness,
- duplicate rates,
- ground-truth cardinality,
- malformed records.

Never silently assume the documentation and local files are identical.

Do not design algorithms that require:

```text
S1 × S2
```

or:

```text
S1 × (S2 + S3)
```

exhaustive pairwise comparisons.

The system must use:

```text
candidate generation / blocking / retrieval
```

---

---

## Schema

## 12. Data Schema

Entity records contain:

```text
entity_id
business_name
business_address
country
```

Ground truth contains:

```text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` is a comma-separated list.

An empty value represents a singleton / no-match entity according to the current project interpretation and must be verified against the official scorer.

Important:

- S1 addresses are documented as complete.
- S2/S3 addresses can be missing.
- Never assume addresses are non-null.
- Normalize null addresses safely before string processing.

Example:

```python
df = df.with_columns(
    pl.col("business_address")
      .fill_null("")
      .str.strip_chars()
)
```

---

---

## Critical dataset behaviour

These explain *why* the rules in `AGENTS.md` exist. The binding constraints —
"address == '' means evidence unavailable, not no-match", "do not hardcode country
branches", "PIN codes are not a universal blocking key" — remain in `AGENTS.md`.

## 14. Critical Dataset Behavior

### Missing addresses

S2 and S3 contain missing addresses.

Therefore:

```text
address == ""
```

must not automatically mean:

```text
not a match
```

It means:

```text
address evidence unavailable
```

The matcher should combine name, address, country, and structural evidence.

---

### Country distribution shift

Training contains US and India but no France according to the current challenge materials.

Test contains:

- US
- India
- France

Therefore:

> Do not build a system that assumes every test entity belongs to the training country distribution.

Avoid hardcoded assumptions such as:

```python
if country == "India":
    ...
elif country == "US":
    ...
```

unless there is a measured and documented reason.

---

### Indian PIN-code behavior

Current project analysis indicates that S1 does not contain the relevant Indian six-digit PIN code consistently enough to use it as a universal S1→S2/S3 blocking key.

Therefore:

```text
PIN code exact blocking
```

must not be treated as a universal retrieval strategy.

PIN/postal information may still be useful as:

- pair evidence,
- structural evidence,
- address-component retrieval,

when present.

---

### Text noise

Expect:

- abbreviations,
- punctuation changes,
- casing differences,
- whitespace changes,
- Unicode differences,
- transliteration,
- business suffixes,
- address formatting differences,
- symbols such as `&`, `/`, `-`, `|`,
- multilingual text.

Normalization must reduce superficial differences without destroying useful information.

---

---

## Singletons in the training ground truth

> **Correction.** The original text here read: *"the training ground truth does not
> contain them in the same way"*, implying training singletons are absent. They are not.
> Train contains **123,247 singletons (5.58%)**, uniformly distributed across US (5.58%)
> and India (5.59%). This makes singleton handling a *supervised* problem with 123k
> labelled examples rather than a guess, and it invalidates any plan that assumed
> $\pi$ was unidentifiable from train.
>
> The binding rule is unchanged and still valid: the system must be able to emit an
> empty match list, and must not assume every S1 entity has a match.

## 31. Singleton / No-Match Handling

Do not assume:

```text
every S1 has a match
```

The current challenge materials indicate that test contains singleton/no-match entities, while the training ground truth does not contain them in the same way.

Therefore the model must be capable of producing:

```text
[]
```

when evidence is insufficient.

Do not force every S1 entity to match something.

False positives are especially harmful under F0.5.

Any claim about the exact singleton rate must be treated as:

```text
MEASURED
```

only after being supported by a valid estimation/experiment.

---

---

## Test data

## 15. Test Data Rule

The final test labels are unavailable.

Never:

- infer labels from prohibited external sources,
- manually identify businesses through web search,
- use external business databases,
- use geocoding or lookup services for matching unless explicitly permitted,
- tune thresholds using hidden test labels,
- hardcode test-specific matches,
- inspect prohibited external information.

The test set may be used for:

- inference,
- candidate generation,
- performance benchmarking,
- submission generation,

subject to official challenge rules.

Any test-set experiment must be clearly labeled as a test experiment.

---
