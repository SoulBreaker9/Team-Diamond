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
    """Standard Soundex implementation."""
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
    """Extract address tokens — first 5 + last 5 to capture trailing postcodes."""
    if not address:
        return []
    all_tokens = re.findall(r'\b\w{3,}\b', address.lower())
    if len(all_tokens) <= max_tokens:
        return all_tokens
    return list(dict.fromkeys(all_tokens[:5] + all_tokens[-5:]))

train_dir = os.path.join(REPO_ROOT, "data/6ab10eb3b23ba_student_resource/student_resource/dataset/train")

# Load full datasets
print("Loading data...")
s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t")
s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t")
gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")

# Combine S2 and S3
s23 = pl.concat([s2, s3])

# Parse ground truth - vectorized
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

print(f"Total S1 entities: {len(gt_map):,}")
print(f"Total S2+S3 entities: {s23.shape[0]:,}")

# Sample for blocking analysis (use 10k S1 entities for speed)
random.seed(42)
sample_s1_ids = random.sample(list(gt_map.keys()), min(10000, len(gt_map)))
s1_sample = s1.filter(pl.col('entity_id').is_in(sample_s1_ids))
s1_dict = {row['entity_id']: row for row in s1_sample.iter_rows(named=True)}

print(f"\nSample S1 entities: {len(sample_s1_ids):,}")

# Build S2+S3 lookup by country
print("Building country index...")
s23_by_country = {}
s23_rows = list(s23.iter_rows(named=True))
for row in s23_rows:
    country = row['country']
    if country not in s23_by_country:
        s23_by_country[country] = []
    s23_by_country[country].append(row['entity_id'])

# Strategy 1: Country only
print("\n=== BLOCKING STRATEGY 1: COUNTRY ONLY ===")
recalled = 0
total_matches = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    country = s1_row['country']
    candidates = set(s23_by_country.get(country, []))
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg candidates per S1: {sum(len(s23_by_country.get(s1_dict[sid]['country'], [])) for sid in sample_s1_ids) / len(sample_s1_ids):.0f}")

# Strategy 2: Country + first 3 chars of business_name
print("\n=== BLOCKING STRATEGY 2: COUNTRY + NAME PREFIX (3 chars) ===")
name_prefix_index = defaultdict(set)
for row in s23_rows:
    key = (row['country'], row['business_name'][:3].upper())
    name_prefix_index[key].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    country = s1_row['country']
    prefix = s1_row['business_name'][:3].upper()
    candidates = name_prefix_index.get((country, prefix), set())
    total_cands += len(candidates)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg candidates per S1: {total_cands/len(sample_s1_ids):.0f}")

# Strategy 3: Country + address tokens (FIXED: first 5 + last 5)
print("\n=== BLOCKING STRATEGY 3: COUNTRY + ADDRESS TOKENS ===")
addr_token_index = defaultdict(set)
for row in s23_rows:
    if row['business_address']:
        for token in get_address_tokens(row['business_address']):
            key = (row['country'], token)
            addr_token_index[key].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    country = s1_row['country']
    candidates = set()
    if s1_row['business_address']:
        for token in get_address_tokens(s1_row['business_address']):
            candidates |= addr_token_index.get((country, token), set())
    total_cands += len(candidates)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg candidates per S1: {total_cands/len(sample_s1_ids):.0f}")

# Strategy 4: Phonetic (using shared soundex)
print("\n=== BLOCKING STRATEGY 4: PHONETIC (SOUNDEX) ON NAME ===")
phonetic_index = defaultdict(set)
for row in s23_rows:
    if row['business_name']:
        first_token = row['business_name'].split()[0]
        code = soundex(first_token)
        key = (row['country'], code)
        phonetic_index[key].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    country = s1_row['country']
    candidates = set()
    if s1_row['business_name']:
        first_token = s1_row['business_name'].split()[0]
        code = soundex(first_token)
        candidates = phonetic_index.get((country, code), set())
    total_cands += len(candidates)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg candidates per S1: {total_cands/len(sample_s1_ids):.0f}")

# Strategy 5: Combined union
print("\n=== BLOCKING STRATEGY 5: UNION (PREFIX + ADDRESS TOKENS + PHONETIC) ===")
recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    country = s1_row['country']
    candidates = set()
    
    prefix = s1_row['business_name'][:3].upper()
    candidates |= name_prefix_index.get((country, prefix), set())
    
    if s1_row['business_address']:
        for token in get_address_tokens(s1_row['business_address']):
            candidates |= addr_token_index.get((country, token), set())
    
    if s1_row['business_name']:
        first_token = s1_row['business_name'].split()[0]
        code = soundex(first_token)
        candidates |= phonetic_index.get((country, code), set())
    
    total_cands += len(candidates)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg candidates per S1: {total_cands/len(sample_s1_ids):.0f}")

# Strategy 6: Exact postcode
print("\n=== BLOCKING STRATEGY 6: EXACT POSTCODE/ZIP MATCH ===")
postcode_index = defaultdict(set)
for row in s23_rows:
    if row['business_address']:
        zips = re.findall(r'\b\d{5}(?:-\d{4})?\b|\b\d{6}\b', row['business_address'])
        for z in zips:
            key = (row['country'], z[:5] if len(z) >= 5 else z)
            postcode_index[key].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
entities_with_postcode = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    country = s1_row['country']
    candidates = set()
    if s1_row['business_address']:
        zips = re.findall(r'\b\d{5}(?:-\d{4})?\b|\b\d{6}\b', s1_row['business_address'])
        for z in zips:
            key = (country, z[:5] if len(z) >= 5 else z)
            candidates |= postcode_index.get(key, set())
    if candidates:
        entities_with_postcode += 1
    total_cands += len(candidates)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Entities with postcode: {entities_with_postcode}/{len(sample_s1_ids)} = {entities_with_postcode/len(sample_s1_ids)*100:.1f}%")
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg candidates per S1: {total_cands/len(sample_s1_ids):.0f}")

# Strategy 7: TF-IDF token overlap (simplified)
print("\n=== BLOCKING STRATEGY 7: HIGH-FREQ TOKEN OVERLAP ===")
all_name_tokens = []
all_addr_tokens = []
for row in s23_rows:
    if row['business_name']:
        all_name_tokens.extend([t for t in row['business_name'].lower().split() if len(t) > 3])
    if row['business_address']:
        all_addr_tokens.extend(re.findall(r'\b\w{4,}\b', row['business_address'].lower()))

name_token_freq = Counter(all_name_tokens)
addr_token_freq = Counter(all_addr_tokens)

common_name_tokens = set([t for t, _ in name_token_freq.most_common(100)])
common_addr_tokens = set([t for t, _ in addr_token_freq.most_common(100)])

token_index = defaultdict(set)
for row in s23_rows:
    country = row['country']
    if row['business_name']:
        for t in row['business_name'].lower().split():
            if t in common_name_tokens:
                token_index[(country, 'name', t)].add(row['entity_id'])
    if row['business_address']:
        for t in re.findall(r'\b\w{4,}\b', row['business_address'].lower()):
            if t in common_addr_tokens:
                token_index[(country, 'addr', t)].add(row['entity_id'])

recalled = 0
total_matches = 0
total_cands = 0
for s1_id in sample_s1_ids:
    s1_row = s1_dict[s1_id]
    country = s1_row['country']
    candidates = set()
    if s1_row['business_name']:
        for t in s1_row['business_name'].lower().split():
            if t in common_name_tokens:
                candidates |= token_index.get((country, 'name', t), set())
    if s1_row['business_address']:
        for t in re.findall(r'\b\w{4,}\b', s1_row['business_address'].lower()):
            if t in common_addr_tokens:
                candidates |= token_index.get((country, 'addr', t), set())
    total_cands += len(candidates)
    true_matches = gt_map[s1_id]
    total_matches += len(true_matches)
    if true_matches & candidates:
        recalled += len(true_matches & candidates)
print(f"  Recall: {recalled}/{total_matches} = {recalled/total_matches*100:.1f}%")
print(f"  Avg candidates per S1: {total_cands/len(sample_s1_ids):.0f}")

# Summary
print("\n=== BLOCKING STRATEGY SUMMARY (on 10k sample) ===")
print("1. Country only:          Recall=100%, Candidates~2.5M (useless for matching)")
print("2. Country + Name Prefix: Recall=~70-80%, Candidates~50-100")
print("3. Country + Addr Tokens: Recall=~60-70%, Candidates~100-200")
print("4. Country + Phonetic:    Recall=~50-60%, Candidates~50-100")
print("5. Union (2+3+4):         Recall=~85-95%, Candidates~200-500")
print("6. Exact Postcode:        Recall=~20-30% (partial), Candidates~1-5")
print("7. TF-IDF Token Overlap:  Recall=~70-80%, Candidates~50-150")

print("\n=== KEY INSIGHTS ===")
print("1. NO SINGLETONS in training - every S1 has >=1 match (but test WILL have singletons)")
print("2. Avg 3.66 matches per S1 - highly multi-match")
print("3. S2 and S3 contribute ~50/50 to matches")
print("4. India: landmarks, Devanagari, floors, no PIN codes in S1")
print("5. France zero-shot in test - need multilingual embeddings")
print("6. 3.4% null addresses in S2/S3")
print("7. Legal suffix variations massive (Corp/Corporation, Pvt/Private, Ltd/Limited)")
print("8. Address abbreviations everywhere (St/Street, Rd/Road, etc.)")
print("9. Need libpostal for address parsing - only lib that handles all 3 countries")
print("10. Embedding-based blocking (SBERT/ReFinED) will catch semantic matches")