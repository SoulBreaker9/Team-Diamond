# EDA SUMMARY & FEATURE ENGINEERING PRIORITIES
# Based on completed analysis of 2.2M S1, 5M S2, 5.3M S3, 2.2M GT records

print("=" * 80)
print("COMPREHENSIVE EDA SUMMARY - AMAZON ML CHALLENGE 2026")
print("=" * 80)

print("""
DATA STATISTICS
---------------
Train:
  Source 1 (Reference):     2,206,821 records  |  0.20 GB  |  0% nulls
  Source 2 (Vendor 1):      5,034,616 records  |  0.46 GB  |  3.4% null addresses
  Source 3 (Vendor 2):      5,285,603 records  |  0.47 GB  |  3.3% null addresses
  Ground Truth:             2,206,821 records  |  0.12 GB  |  100% S1 coverage

Test:
  Source 1:                 1,732,545 records  |  0.16 GB  |  3 countries (US/India/France)
  Source 2:                 4,887,274 records  |  0.47 GB  |  2.7% null addresses
  Source 3:                 5,082,317 records  |  0.47 GB  |  2.7% null addresses

COUNTRY DISTRIBUTION
--------------------
Train:  US ~60%, India ~40%  (NO France)
Test:   US ~38%, India ~46%, France ~15%  (France is ZERO-SHOT)

GROUND TRUTH INSIGHTS
---------------------
- NO SINGLETONS in training (every S1 has >=1 match)
- Avg matches per S1: 3.66
- Match cardinality: 1-match=5.4%, 2-5 matches=~60%, 6+=~30%
- S2/S3 split: ~50/50
- This means: training is biased toward matched entities, test WILL have singletons

NOISE PATTERNS IDENTIFIED
-------------------------
1. LEGAL SUFFIX VARIATIONS (Massive)
   US: Inc/Incorporated (21k), LLC (32k), Corp/Corporation (3k), Ltd (260)
   India: Pvt/Private (10k), Ltd/Limited (13k), LLP (900)
   S2/S3 have similar but different distributions

2. SPECIAL CHARACTERS IN NAMES
   S1: &=10k, .=13k, ()=4k, /=463
   S2: &=8k, .=24k, ()=10k, |=594, /=1.5k
   S3: &=8k, .=23k, ()=10k, |=665, /=3.2k

3. ADDRESS ABBREVIATIONS (Everywhere)
   St/Street (28k/13k), Rd/Road (42k/26k), Ave/Avenue (17k/8k), Dr/Drive (20k/9k)
   Plus: Blvd, Ln, Cir, Ct, Pl, Hwy, Pkwy

4. INDIA-SPECIFIC (Critical)
   - Landmarks: "Near SBI ATM", "Opposite Mall" (6k+ occurrences)
   - Devanagari script: 10k+ names in S2, 6k+ in S3; 11k+ addrs in S2, 10k+ in S3
   - Floor/Unit: 15k+ in S1, 13k+ in S2, 10k+ in S3
   - NO PIN codes in S1 (0 found) - but present in S2/S3

5. FRANCE ZERO-SHOT (Test only)
   - Accented chars: e-acute, e-grave, a-grave, c-cedilla, o-circumflex, u-circumflex
   - Legal forms: SARL, SAS, SASU, EURL, SCI, SA
   - Address format: "5 bis Rue...", "18 RUE JEN ZAY", "NO. 5 ALLEE DES HETRES"
   - Mixed case, French abbreviations (R. for Rue, AV for Avenue)

6. ENCODING
   S1: Clean ASCII (0 non-ascii names)
   S2: 30k non-ascii names, 19k non-ascii addresses
   S3: 23k non-ascii names, 18k non-ascii addresses

BLOCKING STRATEGY RECALL CEILING (Estimated)
--------------------------------------------
Strategy                          Recall    Candidates/S1  Use Case
Country only                      100%      ~2.5M          Baseline (useless)
Country + Name Prefix (3)         ~75%      50-100         Core blocking
Country + Address Tokens          ~65%      100-200        Complementary
Country + Phonetic (Soundex)      ~55%      50-100         Transliterations
Union (Prefix+Addr+Phonetic)      ~90-95%   200-500        RECOMMENDED BASE
Exact Postcode/ZIP                ~25%      1-5            High-precision anchor
Exact Normalized Address          ~20%      1-3            High-precision anchor
Embedding ANN (SBERT/ReFinED)     ~85%      50-100         Semantic/abbreviation
TF-IDF Char n-grams               ~75%      50-150         Typos/partial

TARGET: >95% recall at blocking stage (union of 5+ strategies)
""")

print("=" * 80)
print("FEATURE ENGINEERING PRIORITIES (Ranked by Impact)")
print("=" * 80)

features = [
    ("TIER 1 - CRITICAL (Must Have)", [
        "libpostal parsed address components: house_number, road, unit, level, postcode, city, state, country, suburb, near, po_box",
        "Jaro-Winkler on business_name (handles abbreviations, typos)",
        "Jaro-Winkler on normalized address components (road, city, postcode)",
        "Levenshtein/Damerau-Levenshtein on name (transpositions)",
        "TF-IDF Cosine (char 3-grams + word tokens) on name + address",
        "Exact match on country (mandatory filter)",
        "Exact match on postcode (ZIP5 for US, PIN for India)",
        "Soundex/Metaphone on first name token (transliteration)",
        "SBERT embedding cosine (all-MiniLM-L6-v2 for US/India, paraphrase-multilingual for France)",
        "Name token Jaccard (word-level overlap)",
    ]),
    
    ("TIER 2 - HIGH VALUE (Strong Signal)", [
        "libpostal expansion overlap (canonical address forms)",
        "Legal suffix equivalence (Corp/Corporation, Pvt/Private, Ltd/Limited, SARL/SAS)",
        "Landmark Jaccard (India: 'near', 'opposite', 'behind')",
        "Devanagari/Latin transliteration similarity (India)",
        "French accent normalization similarity (France)",
        "French legal form match (SARL/SAS/SA/EURL/SCI)",
        "Name-in-address check (does business name appear in address?)",
        "Address component Jaccard (set overlap of parsed tokens)",
        "House number exact/Levenshtein match",
        "Cross-source features: S1-S2 vs S1-S3 source indicator",
    ]),
    
    ("TIER 3 - COUNTRY-SPECIFIC", [
        "US: State abbreviation match (CA<->California), ZIP+4 match",
        "US: 'Inc' vs 'Incorporated', 'LLC' vs 'L.L.C.' normalization",
        "India: PIN code exact, State match, Floor/Unit match",
        "India: Mixed script handling (Devanagari name + Latin address)",
        "France: bis/ter handling (12 bis vs 12), cedex codes",
        "France: Accent removal comparison (e<->e, a<->a)",
        "France: Department code from postcode (75=Paris, 69=Rhone)",
    ]),
    
    ("TIER 4 - STATISTICAL / LEARNED", [
        "Splink Fellegi-Sunter match weights per comparison",
        "Splink EM posterior probability (as feature)",
        "Term frequency weights (rare tokens = higher weight)",
        "Cross-encoder score (if compute budget allows, e.g., MiniLM reranker)",
        "ReFinED entity type compatibility (ORG<->ORG, ORG<->LOC)",
        "ReFinED entity description similarity (if linked)",
    ]),
    
    ("TIER 5 - ENSEMBLE / META", [
        "Singleton probability (binary: has_any_match)",
        "Max candidate probability per S1 entity",
        "Number of candidates from blocking",
        "Country-specific threshold calibration",
        "Connected component transitive closure features",
    ]),
]

for tier, feats in features:
    print(f"\n{tier}")
    for i, f in enumerate(feats, 1):
        print(f"  {i}. {f}")

print("""
TOTAL ESTIMATED FEATURES: ~70-100 per candidate pair
""")

print("=" * 80)
print("END-TO-END PIPELINE ARCHITECTURE")
print("=" * 80)

pipeline = """
+---------------------------------------------------------------------------------+
|                        ENSEMBLE PIPELINE (RECOMMENDED)                           |
+---------------------------------------------------------------------------------+
|                                                                                  |
|  STAGE 0: DATA LOADING & NORMALIZATION                                          |
|  +- Polars read_csv(separator="\\t") - 10x faster than pandas                   |
|  +- libpostal.parse_address() on ALL addresses (parallel, 10-30 min)            |
|  |   -> Extract: house_number, road, unit, level, postcode, city, state,        |
|  |     country, suburb, near, category, po_box, etc.                            |
|  +- libpostal.expand_address() -> canonical forms for exact matching             |
|  +- Unicode normalize (NFKC), lowercase, deduplicate whitespace                 |
|  +- Abbreviation expansion dictionaries (country-specific)                      |
|  +- Features: record_id, source, country, text_length, token_counts             |
|                                                                                  |
|  STAGE 1: MULTI-STRATEGY BLOCKING (Target: >95% Recall)                         |
|  +- Strategy 1: Exact match on normalized address (country, postcode, road,    |
|  |   house_number) -> High precision anchors                                    |
|  +- Strategy 2: TF-IDF char 3-grams + word tokens (name + address)            |
|  +- Strategy 3: Phonetic blocking (Metaphone/Double Metaphone on name)        |
|  +- Strategy 4: Embedding ANN (SBERT all-MiniLM-L6-v2 + multilingual,         |
|  |   FAISS HNSW, top-100)                                                      |
|  +- Strategy 5: Splink learned blocking (auto-generated rules)                |
|  +- Strategy 6: Country + Name prefix (coarse filter for large countries)     |
|  +- UNION all strategies -> deduplicate -> Cap at 200 candidates/S1              |
|  +- OUTPUT: candidate_pairs.tsv (audit file)                                   |
|                                                                                  |
|  STAGE 2: FEATURE ENGINEERING (~70 features per candidate pair)                 |
|  +- String Similarity (Name): 15 features (JW, Jaro, Levenshtein, D-L, LCS,   |
|  |   Jaccard word/char, TF-IDF cosine, prefix/suffix, abbrev equiv, metaphone)|
|  +- String Similarity (Address): 12 features (component-level, libpostal      |
|  |   expansion overlap, full addr JW/Levenshtein, landmark, PO box)           |
|  +- Semantic Features: 8 features (SBERT cosine x2, ReFinED type match,       |
|  |   ReFinED desc sim, cross-encoder optional)                                 |
|  +- Cross Features: 10 features (name-in-addr, country agreement, source,     |
|  |   length ratios, token overlap)                                             |
|  +- Country-Specific: 15 features (US/India/France separate)                   |
|  +- Statistical (Splink): 8 features (TF weights, FS weights, EM posterior)    |
|  +- OUTPUT: Feature matrix (n_candidates x ~70 features)                       |
|                                                                                  |
|  STAGE 3: MODEL TRAINING & ENSEMBLE                                             |
|  +- Training Data Construction:                                                |
|  |   +- Positives: ALL ground truth pairs (2.2M S1 x 3.66 avg = ~8M pairs)   |
|  |   +- Negatives: Sampled from candidates (1:10 ratio, stratified by country)|
|  |   +- Pseudo-positives: Splink EM prob > 0.99 (semi-supervised boost)      |
|  |   +- Singleton negatives: S1 with empty match list (from validation split) |
|  +- Validation: GroupKFold(n=5, group=source1_entity_id) - NO LEAKAGE         |
|  +- Base Models:                                                               |
|  |   +- XGBoost (primary): scale_pos_weight=2.0, max_depth=6, lr=0.05        |
|  |   +- LightGBM: is_unbalance=True, num_leaves=63, min_data_in_leaf=50      |
|  |   +- TabNet (neural): attention-based, learns feature interactions         |
|  |   +- Splink posterior prob (as baseline feature + standalone)              |
|  +- Calibration: Isotonic regression on validation fold (CRITICAL for F0.5)   |
|  +- Threshold Optimization: Grid search maximizing MACRO-F0.5 per country     |
|  +- Singleton Detector: Binary XGBoost per S1 entity (features: num_cands,   |
|  |   max_sim, mean_sim, country, addr_null_flag)                             |
|  +- Country-Adaptive Thresholds: Per-country tuning on validation            |
|                                                                                  |
|  STAGE 4: INFERENCE & POST-PROCESSING                                          |
|  +- Generate candidates for TEST set (same blocking pipeline)                  |
|  +- Compute features for all test candidate pairs                              |
|  +- Ensemble prediction (mean of calibrated probabilities)                     |
|  +- Apply country-specific thresholds (US: 0.52, India: 0.48, France: 0.55)  |
|  +- Singleton detection (if max_prob < singleton_thresh -> empty list)         |
|  +- Connected components (transitive closure on predictions)                  |
|  +- Deduplication: One S2/S3 record -> at most one S1 match                    |
|  +- Format validation: utils/validate_submission.py                            |
|  +- OUTPUT: matching_results.tsv                                               |
|                                                                                  |
+---------------------------------------------------------------------------------+
"""

print(pipeline)

print("=" * 80)
print("PROS/CONS OF KEY DESIGN DECISIONS")
print("=" * 80)

pros_cons = {
    "libpostal for address parsing": {
        "pros": [
            "Only library handling US/India/France addresses natively (trained on OSM global)",
            "Parses landmarks, mixed scripts, missing components, transliterations",
            "Returns structured components for exact matching",
            "MIT license - competition compliant"
        ],
        "cons": [
            "C library dependency - need system install (libpostal-dev)",
            "Slower than regex (but 10-30 min for full dataset is acceptable)",
            "Python bindings (pypostal) can be tricky to install on Windows"
        ]
    },
    "Polars over Pandas": {
        "pros": [
            "10-50x faster for large datasets (multi-threaded, streaming)",
            "Handles 10M+ rows in memory efficiently",
            "Lazy evaluation for memory efficiency",
            "Native support for list/struct columns"
        ],
        "cons": [
            "Different API from pandas (learning curve)",
            "Some sklearn integrations need .to_pandas()"
        ]
    },
    "Multi-strategy blocking union": {
        "pros": [
            "Maximizes recall ceiling (>95%) - critical for F0.5",
            "Each strategy catches different match types",
            "Redundancy protects against single-strategy failures",
            "Exact postcode/address anchors give high-precision seeds"
        ],
        "cons": [
            "More complex pipeline (more failure points)",
            "Higher candidate count -> more feature computation",
            "Need careful deduplication and capping"
        ]
    },
    "Ensemble (XGBoost + LightGBM + TabNet)": {
        "pros": [
            "Reduces variance, improves generalization",
            "Different inductive biases catch different patterns",
            "XGBoost/LightGBM dominate tabular competitions",
            "TabNet adds neural attention for complex interactions"
        ],
        "cons": [
            "3x training time (but parallelizable)",
            "More hyperparameters to tune",
            "TabNet needs GPU for reasonable speed"
        ]
    },
    "Isotonic calibration + F0.5 threshold optimization": {
        "pros": [
            "Raw probabilities are miscalibrated - calibration FIXES this",
            "Direct optimization for competition metric",
            "Country-specific thresholds handle distribution shift"
        ],
        "cons": [
            "Isotonic needs sufficient validation data per fold",
            "Threshold optimization adds compute",
            "Risk of overfitting to validation if not careful"
        ]
    },
    "Singleton detection as binary classifier": {
        "pros": [
            "F0.5 rewards correct singleton prediction (score=1.0)",
            "Training has 0% singletons but test will have many",
            "Explicit modeling prevents false merges on singletons"
        ],
        "cons": [
            "No positive singleton examples in training (domain shift)",
            "Must rely on validation split or pseudo-labeling",
            "Country-adaptive thresholds needed (France more conservative)"
        ]
    },
    "ReFinED (Amazon's EL) for embeddings/features": {
        "pros": [
            "Amazon-native - organizer alignment",
            "Entity descriptions handle 'same business, different wording'",
            "Fine-grained types (Company vs Organization vs LocalBusiness)",
            "Zero-shot capable for France entities",
            "Apache 2.0 license, <8B params - COMPLIANT"
        ],
        "cons": [
            "Designed for Wikipedia/Wikidata, not business records",
            "Entity linking != Record linkage (different task)",
            "Complex fine-tuning pipeline needed for adaptation",
            "Less proven for this specific problem than SBERT"
        ]
    },
    "Country-adaptive approach for France zero-shot": {
        "pros": [
            "France has NO training data - must generalize",
            "Multilingual SBERT (paraphrase-multilingual-MiniLM-L12-v2) covers 50+ langs",
            "libpostal handles French address format natively",
            "Higher thresholds for France = more conservative (precision)"
        ],
        "cons": [
            "Risk of lower recall on France if too conservative",
            "No validation data for France threshold tuning",
            "Legal form variations (SARL/SAS/SA/EURL) need explicit handling"
        ]
    }
}

for topic, pc in pros_cons.items():
    print(f"\n{topic.upper()}")
    print("  PROS:")
    for p in pc["pros"]:
        print(f"    + {p}")
    print("  CONS:")
    for c in pc["cons"]:
        print(f"    - {c}")

print("""
================================================================================
CRITICAL SUCCESS FACTORS (Ranked)
================================================================================
1. BLOCKING RECALL >95% - This is the hard ceiling. If blocking misses matches,
   no model can recover them. Invest heavily here.

2. libpostal ADDRESS PARSING - Non-negotiable. Only way to handle India landmarks,
   Devanagari, France bis/ter, missing components consistently.

3. CALIBRATION + F0.5 THRESHOLD - Raw model probs are useless. Must calibrate
   (isotonic) and optimize threshold per country for MACRO-F0.5.

4. SINGLETON MODELING - Training has 0% singletons. Test WILL have them.
   Explicit binary classifier + conservative thresholds essential.

5. ENSEMBLE DIVERSITY - Different blocking -> different candidates -> different
   feature distributions -> diverse models -> better generalization.

6. FRANCE ZERO-SHOT - Multilingual embeddings + libpostal + conservative
   threshold (0.55) + French legal form features.

7. VALIDATION RIGOR - GroupKFold by source1_entity_id (NO LEAKAGE).
   Macro-F0.5 per entity, then average. Not micro-F1.

8. SUBMISSION VALIDATION - Run utils/validate_submission.py EVERY TIME.
   Format rejection = 0 score.

================================================================================
RECOMMENDED NEXT STEPS
================================================================================
1. Install libpostal + pypostal (system dependency - do this FIRST)
2. Build Stage 0 normalization pipeline (libpostal + cleaning)
3. Implement 5 blocking strategies + union + candidate_pairs.tsv output
4. Build feature engineering module (~70 features)
5. Construct training pairs (pos from GT, neg from blocking samples)
6. Train XGBoost + LightGBM + calibrate + threshold optimize
7. Add singleton detector + country thresholds
8. Run full pipeline on test set -> matching_results.tsv
9. Validate -> Submit -> Iterate

TIME ESTIMATE: 2-3 days for full implementation with 1 GPU
""")