# Pre-Training Readiness Report

**Date**: 2026-09-26  
**Branch**: `feat/candidate-retrieval` (HEAD, uncommitted)  
**Team**: Team Diamond — Amazon ML Challenge 2026  
**Status**: Code complete, all 114 tests pass. **Not cleared for SageMaker training.**

---

## Executive Summary

The implementation is **functionally complete** across all layers (preprocessing, retrieval, features, labels, split, negatives, model, decision, pipeline, submission). All 114 unit/synthetic tests pass.

**However**, three critical measurements are missing and **must be completed on SageMaker before any training launch**:

1. **7-strategy union candidate recall** — the arithmetic ceiling on achievable F₀.₅
2. **Retrieval cap sweep** — to produce a validated `ScaleEstimate` fitting in 32 GiB
3. **S3 access verification** — from the actual SageMaker instance with its IAM role

Without (1), we cannot know if 95% F₀.₅ is even possible. Without (2), the run will OOM. Without (3), the data cannot be read.

**Recommendation**: Run the retrieval recall experiment on SageMaker first. Only after it produces a measured union recall number and a validated `ScaleEstimate` should training be requested.

---

## 1. Unmeasured Configuration Parameters

From `Config.unmeasured()` across `configs/*.yaml` (22 placeholders):

| Config Path | Status | Required For |
|-------------|--------|--------------|
| `model.params.n_estimators` | NULL | Model capacity |
| `model.params.learning_rate` | NULL | Convergence |
| `model.params.num_leaves` | NULL | Tree complexity |
| `model.params.min_data_in_leaf` | NULL | Overfitting guard |
| `model.params.feature_fraction` | NULL | Feature subsampling |
| `model.params.bagging_fraction` | NULL | Row subsampling |
| `model.params.bagging_freq` | NULL | Bagging schedule |
| `model.params.lambda_l1` | NULL | L1 regularization |
| `model.params.lambda_l2` | NULL | L2 regularization |
| `model.params.n_jobs` | NULL | Must = 8 (ml.r5.xlarge vCPU) |
| `model.hard_negatives.negatives_per_positive` | NULL | Training class balance |
| `retrieval.caps.us` | NULL | US candidate cap |
| `retrieval.caps.in` | NULL | India candidate cap |
| `retrieval.caps.fr` | NULL | France candidate cap (France absent from train!) |
| `retrieval.union_cap` | NULL | Post-dedup cap |
| `retrieval.char_ngram.jaccard_threshold` | NULL | Char-ngram acceptance |
| `features.cascade.top_k_before_stage2b` | NULL | Cascade design |
| `decision.policy` | NULL | Entity-level decision rule |
| `decision.threshold` | NULL | Threshold fallback |
| `project.data_root` | NULL | Path resolution |
| `runtime.dev_machine_forensics.*` | NULL | Machine specs |

**Critical for training**: All model params, `negatives_per_positive`, all retrieval caps, `decision.policy`.

---

## 2. Dataset Locations & S3 Access

### Local (gitignored)
```
/home/hetm/Desktop/Hackathon/6ab10eb3b23ba_student_resource/student_resource/dataset/
├── train/
│   ├── train_source1.tsv       2,206,822 lines
│   ├── train_source2.tsv       5,034,617 lines
│   ├── train_source3.tsv       5,285,604 lines
│   └── train_ground_truth.tsv  2,206,822 lines
└── test/
    ├── test_source1.tsv        1,732,545 lines
    ├── test_source2.tsv        4,887,274 lines
    └── test_source3.tsv        5,082,317 lines
```

### S3 (Challenge specification)
```
s3://sagemaker-ap-south-1-115775806821/student_resource/*
```

**S3 access NOT VERIFIED**: Local AWS credentials are invalid (`InvalidAccessKeyId`). Must probe from SageMaker instance using its IAM role.

---

## 3. Real Data Schema & Integrity (Measured on 5K-row samples)

| Property | Value | Evidence |
|----------|-------|----------|
| S1/S2/S3 columns | `entity_id, business_name, business_address, country` | MEASURED |
| GT columns | `source1_entity_id, matched_entity_ids` (comma-separated) | MEASURED |
| Train S1 rows | 2,206,821 | wc -l - 1 |
| Train S2 rows | 5,034,616 | wc -l - 1 |
| Train S3 rows | 5,285,603 | wc -l - 1 |
| Test S1 rows | **1,732,544** | MEASURED (doc says 1,732,545 — off by header) |
| Null addresses (S2/S3) | ~3.4% | MEASURED |
| Train country mix (S1) | US ~60%, India ~40% | MEASURED |
| **Test country mix (S1)** | **France ~15%, India ~47%, US ~38%** | COMPETITION_NOTES.md |
| S2/S3 ID overlap | 0 in 5K sample | MEASURED |
| GT = strict partial matching | 17,523 pairs → 17,251 unique vendors (0 shared) | MEASURED |

**Distribution shift confirmed**: France appears **only in test**. Any country-branching logic will fail unless confined to normalisation/features.

---

## 4. Preprocessing Pipeline (Verified on Real Data)

| Stage | Status | Notes |
|-------|--------|-------|
| `normalize_columns` (vectorised) | ✅ Works | 13–23× speedup vs reference |
| Null handling | ✅ | `fill_null("")` before string ops |
| Output columns | ✅ | `name_norm, addr_norm, name_tokens_norm, addr_tokens_norm, has_address, has_name` |
| No nulls in output | ✅ | Verified |

---

## 5. Candidate Retrieval on Real Sample

**Sample**: 2,000 S1 queries, **40,000 vendors (20K S2 + 20K S3)**, all 7 strategies

| Metric | Value |
|--------|-------|
| Vendor pool | 40,000 |
| Candidates (uncapped) | 268,459 |
| Candidates per S1 (capped at 50) | median=1, p95=50, max=50 |
| **Pair recall (200 S1 slice)** | **0.0031** (2/653 GT pairs) |
| Top strategies by pair count | `address_token` 8,157, `numeric_token` 2,706, `name_token` 1,539 |
| Name exact/alnum/sorted | Only 8 pairs each |

**Critical finding**: Recall is extremely low on this sample because only exact-match strategies fire for the true pairs. Token-based strategies generate massive candidate sets but may not contain the true matches for this small sample.

---

## 6. Recall Ceiling Arithmetic (The Primary Blocker)

From AGENTS.md §12 derivation:

- F₀.₅ = 1.25·t / (0.25·m + k) where m=|true|, k=|emitted|, t=|true∩emitted|
- At perfect precision (t=k), max F₀.₅ = 1.25·R / (0.25 + R) where R = pair recall
- **For F₀.₅ ≥ 0.95 → requires pair recall R ≥ 0.7917**
- Best single-strategy recall (E004): **0.5220** (sorted_name)
- **7-strategy union recall: UNMEASURED on full corpus** (previous attempts OOM-killed)

**No classifier can exceed the retrieval ceiling.** The union recall experiment (`experiments/scripts/measure_union_recall.py`) must complete on SageMaker first.

---

## 7. Pair Features on Real Data (Verified)

| Check | Result |
|-------|--------|
| Feature count | 63 (60 `FEATURE_COLUMNS` + `country_pair` + 2 IDs) |
| Nulls | **0** across all columns |
| Dtypes | Float32/Float64/Int32/String — all consistent |
| Categorical | `country_pair` only (1 unique in US sample: `US\|US`) |
| FEATURE_COLUMNS match | ✅ Exact |
| Leakage boundary | ✅ IDF built from vendor pool only |
| Categorical vocab frozen | ✅ In `ModelMatrix`, re-used for validation |

---

## 8. Entity-Aware Train/Validation Split (Verified)

```python
split = entity_aware_split(s1_n, gt, validation_fraction=0.2, seed=20260926)
```

| Property | Value |
|----------|-------|
| Train entities | 1,601 (from 2,000 sample) |
| Validation entities | 399 |
| Partial matching verified | ✅ (0 shared vendors in GT) |
| Strata | `country\|m<=bucket` — balanced within strata |
| Singletons handled | ✅ Empty match list retained |

Strata balance example:
- `India|m<=5`: train=0.712, val=0.288
- `US|m<=8`: train=0.778, val=0.222
- Rare strata (`m>8`): 0/100 or 100/0 (expected for small N)

---

## 9. Positive / Hard-Negative Counts (Real Sample)

| Set | Pairs | Positives | Pos Rate | Hard Negatives (3:1) |
|-----|-------|-----------|----------|----------------------|
| Train (uncapped, 7 strategies) | 268,459 | 21 | 0.0078% | 21 pos → 63 neg (25% pos rate) |
| Validation | 73,467 | 7 | 0.0095% | Not sampled |

**Key insight**: Extreme class imbalance (~1:12,000). Hard negatives from retrieval candidates raise positive rate to **25%** — essential. Random negatives would be useless.

---

## 10. F₀.₅ Decision Policy Math (Derived & Implemented)

**Formula** (in `decision/policy.py`):
- Marginal value of emitting candidate with probability p: **p > 0.25·k / (m + 0.25·n)**
- k̂ (expected correct) = sum of emitted probabilities
- At n=0, k=0: bound = 0 (always emit first)
- At n=1, k=1, m=1: bound = 0.20

**Policies implemented**:
- `MarginalValuePolicy` (derived, PROPOSAL)
- `FixedThresholdPolicy` (baseline, PROPOSAL)
- `TopKPolicy` (baseline, PROPOSAL)
- `EmitNothingPolicy` (floor — 5.58% singletons)

**Comparison harness**: `compare_policies()` scores all policies on entity-level macro F₀.₅. **Not yet run on real validation fold.**

---

## 11. ScaleEstimate for Full Corpus (ml.r5.xlarge, 32 GiB)

| Parameter | Estimate | Basis |
|-----------|----------|-------|
| n_s1 (train) | 2,206,821 | MEASURED |
| n_vendors | 10,320,219 | MEASURED |
| candidates_per_query | **UNMEASURED** (cap sweep needed) |
| bytes_per_pair | ~512 B | 63 cols × 8 B + overhead |
| peak_multiplier | 2.5 | Explode/join intermediates |
| **Peak memory @ 50 cand/q** | **~140 GiB** | **EXCEEDS 32 GiB** |
| **Peak memory @ 20 cand/q** | **~56 GiB** | Still exceeds |
| **Peak memory @ 10 cand/q** | **~28 GiB** | Fits with headroom |
| Estimated runtime | 180 min | Unmeasured |

**The `guard_full_corpus_run()` in `pipeline/scale.py` will reject any run without a validated `ScaleEstimate`.**

---

## 12. CatBoost vs LightGBM Selection

| Factor | CatBoost | LightGBM | Evidence |
|--------|----------|----------|----------|
| Native categorical (`country_pair`) | ✅ | ❌ Manual encoding | DESIGN |
| Ordered target statistics | ✅ Reduces leakage | ❌ Standard GOSS | DESIGN |
| GPU support | ✅ | ✅ | Both |
| **Measured on challenge data** | **NO** | **NO** | **Neither trained** |

**Status**: CatBoost chosen by **explicit user instruction**, not measured evidence. `configs/model.yaml` still says `type: lightgbm` — **documented contradiction** per AGENTS.md §3. LightGBM retained in `pyproject.toml` for reversibility.

---

## 13. Proposed Training Configuration

| Item | Value |
|------|-------|
| **Git commit** | Current HEAD (uncommitted on `feat/candidate-retrieval`) |
| **Branch** | `feat/candidate-retrieval` (not pushed) |
| **Local dataset** | `/home/hetm/Desktop/Hackathon/6ab10eb3b23ba_student_resource/student_resource/dataset/` |
| **S3 dataset** | `s3://sagemaker-ap-south-1-115775806821/student_resource/` (unverified) |
| **Instance** | `ml.r5.xlarge` (8 vCPU, 32 GiB, **no GPU**) |
| **Expected runtime** | 3–4 hours (unmeasured) |
| **Estimated cost** | ~$0.75/hr × 4 = **~$3.00** (unmeasured) |
| **ScaleEstimate** | **REQUIRED before launch** |
| **Config files** | model + retrieval + features + submission + decision (missing) |
| **Experiment ID** | E009 (next in `experiments/registry.csv`) |

---

## 14. Readiness Checklist

| Check | Status | Blocker |
|-------|--------|---------|
| Code compiles / tests pass | ✅ 114 tests | — |
| Data loading from real paths | ✅ | — |
| Normalization | ✅ | — |
| Retrieval key builders | ✅ | — |
| Candidate generation | ✅ (small scale) | Full recall unmeasured |
| Feature generation | ✅ (no nulls, correct dtypes) | — |
| Entity-aware split | ✅ | — |
| Labelling / hard negatives | ✅ | — |
| CatBoost fit / predict | ✅ (tiny sample) | — |
| Policy comparison | ⚠️ Not run on real data | Need validation scores |
| F₀.₅ metric | ✅ Implemented & tested | — |
| Submission validator | ✅ Implements §17 contract | — |
| **Scale estimate** | ❌ **MISSING** | Cap sweep + memory profile |
| **S3 access verified** | ❌ **MISSING** | Probe from SageMaker |
| **Retrieval caps measured** | ❌ **MISSING** | Full recall experiment |
| **CatBoost params** | ❌ **ALL NULL** | Sweep or literature defaults |

---

## Required Before SageMaker Training Approval

1. **Complete `measure_union_recall.py` on SageMaker** — measure 7-strategy union pair recall (overall + per-country) to establish the F₀.₅ ceiling.
2. **Sweep retrieval caps** (100–400 grid in `retrieval.yaml`) and record recall/cost per country.
3. **Validate S3 access and layout** from the SageMaker instance (IAM role, path structure).
4. **Select CatBoost params** from literature defaults + quick local sweep on subsampled data.
5. **Choose decision policy** via `compare_policies` on real validation fold.
6. **Record validated ScaleEstimate** that fits in 32 GiB with 2 GiB headroom (via `pipeline/scale.py`).

Only after (1)–(6) are recorded in `experiments/registry.csv` should SageMaker training be requested with **CONFIRMED**.

---

**Report location**: `DOCS/reports/pre_training_readiness.md`  
**Next action**: Human approval to run SageMaker retrieval recall experiment (not training).