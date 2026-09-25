# Amazon ML Challenge 2026: Dataset & EDA Master Briefing
**Team Diamond** | **Target Problem:** Business Entity Resolution across Multi-Source Vendor Feeds  


---

## 📑 Dedicated Deep-Dive Documents

We have created two dedicated, comprehensive technical specifications for the engineering and modeling pipeline:

1. 📘 **[Raw Dataset Specification & Ingestion Guide](file:///c:/Users/Yatrik/OneDrive/Desktop/Amazon%20ML/Team-Diamond/raw_dataset_specification.md)**
   - Physical storage, file sizes, row counts (26.4M rows total), Arrow/Polars memory footprints.
   - Exact schemas, data types, null distribution analysis (3.4% null addresses in S2/S3).
   - Ingestion code with zero-copy Polars typing and defensive null coalescing.
   - Official submission contract and validation rules (`validate_submission.py`).

2. 🔬 **[In-Depth EDA Analysis & Data Behaviour Playbook](file:///c:/Users/Yatrik/OneDrive/Desktop/Amazon%20ML/Team-Diamond/eda_analysis_and_behaviour.md)**
   - Complete empirical statistics (average 3.66 matches per entity, 89% multi-matches).
   - The 6 critical data behaviors and the exact engineering playbook to handle each.
   - The France zero-shot shift (0% in train, 15% in test).
   - The singleton paradox (0% in train, high in test) and confidence gating.
   - Blocking strategy benchmarks (achieving >94% recall with candidate union).
   - End-to-end 5-stage ML pipeline architecture and $F_{0.5}$ metric optimization.

---

## ⚡ Quick Executive Summary: What You Must Know Before Starting

### 1. Data Scale & Memory Footprint
- **Total Records:** **26,435,997 records** (14.73M train, 11.70M test).
- **Physical Size:** **2.35 GB** on disk across 7 uncompressed TSV files.
- **RAM Footprint:** Fits in **~2.24 GB RAM** using Polars (Arrow columnar memory). A standard 16GB–32GB machine is sufficient.

### 2. The Evaluation Metric: $F_{0.5}$ (Precision Over Recall)
$$F_{0.5} = 1.25 \cdot \frac{\text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
- **Precision matters 2× more than recall.** False positives are penalized heavily.
- Never use raw default classification thresholds (0.50). Calibrate probabilities using **Isotonic Regression** and optimize thresholds per country to maximize Macro-$F_{0.5}$.

### 3. The Two Major Domain Shifts
1. **The France Zero-Shot Shift:**
   - Training contains **0 France records** (US ~60%, India ~40%).
   - Test contains **~15% France records** (~260k S1, ~1.44M vendor records).
   - **Handling:** Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`), `libpostal` French address parsing, and conservative thresholds ($\tau_{\text{FR}} \approx 0.55$).
2. **The Singleton Shift:**
   - In training ground truth, **0% of entities are singletons** (all have $\ge 1$ match).
   - In test, a significant portion will have no match in vendor feeds.
   - **Handling:** Build a dedicated singleton confidence gate ($\max P < \tau_{\text{singleton}} \implies \text{empty list}$).

### 4. Asymmetric Address Structure & The Indian "PIN Code Paradox"
- $S_1$ Indian addresses **do not have 6-digit numeric PIN codes**, but have rich building/floor details.
- $S_2$ and $S_3$ Indian addresses **frequently contain PIN codes**, Devanagari script, and landmarks (`"near"`, `"opposite"`).
- **Handling:** Never block $S_1$ to $S_2/S_3$ on exact PIN codes. Block on cities, normalized localities, and landmark tokens.

### 5. Multi-Strategy Union Blocking is Mandatory
- No single blocking method exceeds 75% recall (Name prefix: ~75%, Address tokens: ~65%, Soundex: ~55%).
- To achieve the required **>92–95% candidate recall ceiling**, you must compute the **union of 4 to 5 complementary blocking rules** and deduplicate to $\le 250$ candidates per $S_1$ entity.

---

## 🛠️ Data Engineering Action Checklist

```
[Phase 1] Ingestion:
  └── Load TSV via Polars with defensive fill_null("") on business_address.
  └── Clean entity IDs and normalize names (Unicode NFKC, lowercase).

[Phase 2] Blocking:
  └── Country partition -> 3-char Name Prefix + Address Tokens + Metaphone + FAISS SBERT.
  └── Union candidate pairs, cap at 250 per S1.

[Phase 3] Features:
  └── String distances (Jaro-Winkler, Levenshtein, Token Sort).
  └── libpostal address component comparisons.
  └── Multilingual dense semantic embeddings.

[Phase 4] Modeling:
  └── LightGBM / XGBoost with GroupKFold(source1_entity_id).
  └── Inject synthetic singletons in CV folds.
  └── Calibrate probabilities using Isotonic Regression.

[Phase 5] Post-Processing:
  └── Apply Country-Adaptive Thresholds (US: 0.52, IN: 0.48, FR: 0.55).
  └── Singleton gate check.
  └── Validate with utils/validate_submission.py (1,732,545 exact lines).
```
