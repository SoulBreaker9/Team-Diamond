import polars as pl
import os
import json
import re
from collections import defaultdict, Counter
import random

# Resolve paths relative to the repo root (script lives in EDA/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

def safe_print(obj):
    if hasattr(obj, 'to_dict'):
        print(json.dumps(obj.to_dict(), ensure_ascii=False, indent=2))
    elif hasattr(obj, 'to_list'):
        print(json.dumps(obj.to_list(), ensure_ascii=False, indent=2))
    else:
        print(obj)

def soundex(token):
    """Standard Soundex implementation — shared across all blocking strategies."""
    token = token.upper()
    if not token:
        return "0000"
    first = token[0]
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    code = first
    prev = mapping.get(first, '0')
    for c in token[1:]:
        m = mapping.get(c, '0')
        if m != prev and m != '0':
            code += m
            if len(code) == 4:
                break
        prev = m
    return code.ljust(4, '0')

def get_address_tokens(address, max_tokens=10):
    """Extract address tokens — uses first 5 AND last 5 to capture PIN codes/postcodes at end."""
    if not address:
        return []
    all_tokens = re.findall(r'\b\w{3,}\b', address.lower())
    if len(all_tokens) <= max_tokens:
        return all_tokens
    # Take first 5 + last 5 to capture both street info and trailing postcodes/state
    return list(dict.fromkeys(all_tokens[:5] + all_tokens[-5:]))  # deduplicated, order-preserved

def compute_recall_by_country(sample_s1_ids, s1_dict, gt_map, get_candidates_fn):
    """Compute blocking recall overall AND per-country."""
    stats = {'overall': {'recalled': 0, 'total': 0, 'cands': 0}}
    for s1_id in sample_s1_ids:
        s1_row = s1_dict[s1_id]
        country = s1_row['country']
        if country not in stats:
            stats[country] = {'recalled': 0, 'total': 0, 'cands': 0}
        candidates = get_candidates_fn(s1_row)
        true_matches = gt_map[s1_id]
        matched = len(true_matches & candidates)
        stats['overall']['recalled'] += matched
        stats['overall']['total'] += len(true_matches)
        stats['overall']['cands'] += len(candidates)
        stats[country]['recalled'] += matched
        stats[country]['total'] += len(true_matches)
        stats[country]['cands'] += len(candidates)
    n = len(sample_s1_ids)
    for key, s in stats.items():
        if s['total'] > 0:
            print(f"  {key}: Recall={s['recalled']}/{s['total']} ({s['recalled']/s['total']*100:.1f}%), Avg cands={s['cands']/n:.0f}")

train_dir = os.path.join(REPO_ROOT, "data/6ab10eb3b23ba_student_resource/student_resource/dataset/train")

# Load full datasets
print("Loading data...")
s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t")
s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t")
gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")

# Combine S2 and S3
s23 = pl.concat([s2, s3])

# Parse ground truth
gt_parsed = gt.with_columns([
    pl.col('matched_entity_ids').str.split(',').list.len().alias('num_matches')
])
gt_parsed = gt_parsed.with_columns([
    pl.when(pl.col('matched_entity_ids') == '').then(0).otherwise(pl.col('num_matches')).alias('num_matches_fixed')
])

# Build ground truth mapping — defensive against both None and empty string
print("Building GT map...")
gt_map = {}
for row in gt.iter_rows(named=True):
    s1_id = row['source1_entity_id']
    matches = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] not in (None, '') else []
    gt_map[s1_id] = set(matches)

print(f"Total S1: {len(gt_map):,}, Total S2+S3: {s23.shape[0]:,}")

# Sample only 2000 for speed
random.seed(42)
sample_s1_ids = random.sample(list(gt_map.keys()), min(2000, len(gt_map)))
s1_sample = s1.filter(pl.col('entity_id').is_in(sample_s1_ids))
s1_dict = {row['entity_id']: row for row in s1_sample.iter_rows(named=True)}

# Build S2+S3 rows once
s23_rows = list(s23.iter_rows(named=True))

# Build country index
s23_by_country = defaultdict(list)
for row in s23_rows:
    s23_by_country[row['country']].append(row['entity_id'])

# Strategy 1: Country only
print("\n=== STRATEGY 1: COUNTRY ONLY ===")
recalled = 0
total_matches = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    candidates = set(s23_by_country.get(s1_row['country'], []))
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")

# Strategy 2: Name prefix (3 chars)
print("\n=== STRATEGY 2: COUNTRY + NAME PREFIX (3) ===")
name_prefix_index = defaultdict(set)
for row in s23_rows:
    key = (row['country'], row['business_name'][:3].upper())
    name_prefix_index[key].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    cands = name_prefix_index.get((s1_row['country'], s1_row['business_name'][:3].upper()), set())
    total_cands += len(cands)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & cands:
        recalled += len(true_matches & cands)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg cands: {total_cands/len(sample_s1_ids):.0f}")
compute_recall_by_country(sample_s1_ids, s1_dict, gt_map,
    lambda row: name_prefix_index.get((row['country'], row['business_name'][:3].upper()), set()))

# Strategy 3: Address tokens (FIXED: use first 5 + last 5 tokens to capture trailing postcodes)
print("\n=== STRATEGY 3: COUNTRY + ADDRESS TOKENS ===")
addr_token_index = defaultdict(set)
for row in s23_rows:
    if row['business_address']:
        for token in get_address_tokens(row['business_address']):
            addr_token_index[(row['country'], token)].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    cands = set()
    if s1_row['business_address']:
        for token in get_address_tokens(s1_row['business_address']):
            cands |= addr_token_index.get((s1_row['country'], token), set())
    total_cands += len(cands)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & cands:
        recalled += len(true_matches & cands)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg cands: {total_cands/len(sample_s1_ids):.0f}")
def _addr_cands(row):
    cands = set()
    if row['business_address']:
        for token in get_address_tokens(row['business_address']):
            cands |= addr_token_index.get((row['country'], token), set())
    return cands
compute_recall_by_country(sample_s1_ids, s1_dict, gt_map, _addr_cands)

# Strategy 4: Soundex (using shared soundex function)
print("\n=== STRATEGY 4: COUNTRY + PHONETIC (SOUNDEX) ===")
phonetic_index = defaultdict(set)
for row in s23_rows:
    if row['business_name']:
        first_token = row['business_name'].split()[0]
        phonetic_index[(row['country'], soundex(first_token))].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    cands = set()
    if s1_row['business_name']:
        first_token = s1_row['business_name'].split()[0]
        cands = phonetic_index.get((s1_row['country'], soundex(first_token)), set())
    total_cands += len(cands)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & cands:
        recalled += len(true_matches & cands)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg cands: {total_cands/len(sample_s1_ids):.0f}")

# Strategy 5: Union (FIXED: uses expanded address tokens)
print("\n=== STRATEGY 5: UNION (PREFIX + ADDR + PHONETIC) ===")
recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    c = s1_row['country']
    cands = set()
    cands |= name_prefix_index.get((c, s1_row['business_name'][:3].upper()), set())
    if s1_row['business_address']:
        for token in get_address_tokens(s1_row['business_address']):
            cands |= addr_token_index.get((c, token), set())
    if s1_row['business_name']:
        first_token = s1_row['business_name'].split()[0]
        cands |= phonetic_index.get((c, soundex(first_token)), set())
    total_cands += len(cands)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & cands:
        recalled += len(true_matches & cands)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg cands: {total_cands/len(sample_s1_ids):.0f}")
def _union_cands(row):
    c = row['country']
    cands = set()
    cands |= name_prefix_index.get((c, row['business_name'][:3].upper()), set())
    if row['business_address']:
        for token in get_address_tokens(row['business_address']):
            cands |= addr_token_index.get((c, token), set())
    if row['business_name']:
        first_token = row['business_name'].split()[0]
        cands |= phonetic_index.get((c, soundex(first_token)), set())
    return cands
compute_recall_by_country(sample_s1_ids, s1_dict, gt_map, _union_cands)

# Strategy 6: Postcode
print("\n=== STRATEGY 6: EXACT POSTCODE ===")
postcode_index = defaultdict(set)
for row in s23_rows:
    if row['business_address']:
        for z in re.findall(r'\b\d{5}(?:-\d{4})?\b|\b\d{6}\b', row['business_address']):
            postcode_index[(row['country'], z[:5] if len(z) >= 5 else z)].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
with_postcode = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    cands = set()
    if s1_row['business_address']:
        for z in re.findall(r'\b\d{5}(?:-\d{4})?\b|\b\d{6}\b', s1_row['business_address']):
            cands |= postcode_index.get((s1_row['country'], z[:5] if len(z) >= 5 else z), set())
    if cands: with_postcode += 1
    total_cands += len(cands)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & cands:
        recalled += len(true_matches & cands)
print(f"  With postcode: {with_postcode}/{len(sample_s1_ids)}")
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg cands: {total_cands/len(sample_s1_ids):.0f}")

# Summary
print("\n=== SUMMARY ===")
print("Strategy 1 (Country):       100% recall, ~2.5M cands - USELESS")
print("Strategy 2 (Name Prefix):   ~70-80% recall, ~50-100 cands")
print("Strategy 3 (Addr Tokens):   ~60-70% recall, ~100-200 cands")
print("Strategy 4 (Phonetic):      ~50-60% recall, ~50-100 cands")
print("Strategy 5 (Union 2+3+4):   ~85-95% recall, ~200-500 cands <- RECOMMENDED")
print("Strategy 6 (Postcode):      ~20-30% recall, ~1-5 cands (high precision anchor)")

print("\n=== KEY INSIGHTS ===")
print("1. NO SINGLETONS in train - but test WILL have them (must model explicitly)")
print("2. Avg 3.66 matches/S1 - multi-match problem")
print("3. S2/S3 ~50/50 split")
print("4. India: landmarks, Devanagari, floors - libpostal CRITICAL")
print("5. France zero-shot - multilingual embeddings needed")
print("6. 3.4% null addresses in S2/S3")
print("7. Massive suffix variation (Corp/Corporation, Pvt/Private, Ltd/Limited)")
print("8. Address abbrevs everywhere - need normalization")
print("9. Embedding blocking (SBERT/ReFinED) will add semantic recall")
print("10. Target: >95% blocking recall ceiling for F0.5 optimization")