import polars as pl
import os
import json
import re
from collections import Counter

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

train_dir = os.path.join(REPO_ROOT, "data/6ab10eb3b23ba_student_resource/student_resource/dataset/train")
s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t", n_rows=200000)
s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t", n_rows=200000)
s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t", n_rows=200000)

print("=== NOISE PATTERN ANALYSIS ===")

# 1. Legal suffix variations in names
print("\n1. LEGAL SUFFIX VARIATIONS (business_name)")
suffix_patterns = {
    'corp_variants': r'\b(corp|corporation|inc|incorporated|llc|llp|ltd|limited|pvt|private|plc)\b',
    'india_suffixes': r'\b(pvt|private|ltd|limited|llp|llc|corp|corporation)\b',
    'us_suffixes': r'\b(inc|incorporated|corp|corporation|llc|llp|ltd|limited|co|company)\b',
}

for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    for country in ['US', 'India']:
        subset = df.filter(pl.col('country') == country)
        if subset.shape[0] == 0:
            continue
        names = subset['business_name'].to_list()
        # Count suffix occurrences
        corp_count = sum(1 for n in names if re.search(r'\bcorp\b', n, re.I))
        corporation_count = sum(1 for n in names if re.search(r'\bcorporation\b', n, re.I))
        inc_count = sum(1 for n in names if re.search(r'\binc\b', n, re.I))
        llc_count = sum(1 for n in names if re.search(r'\bllc\b', n, re.I))
        pvt_count = sum(1 for n in names if re.search(r'\bpvt\b', n, re.I))
        ltd_count = sum(1 for n in names if re.search(r'\bltd\b', n, re.I))
        print(f"  {name} {country}: corp={corp_count}, corporation={corporation_count}, inc={inc_count}, llc={llc_count}, pvt={pvt_count}, ltd={ltd_count}")

# 2. Special characters and punctuation in names
print("\n2. SPECIAL CHARACTERS IN NAMES")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    names = df['business_name'].to_list()
    amp_count = sum(1 for n in names if '&' in n)
    dash_count = sum(1 for n in names if '-' in n)
    dot_count = sum(1 for n in names if '.' in n)
    paren_count = sum(1 for n in names if '(' in n or ')' in n)
    pipe_count = sum(1 for n in names if '|' in n)
    slash_count = sum(1 for n in names if '/' in n)
    print(f"  {name}: &= {amp_count}, -= {dash_count}, .= {dot_count}, ()= {paren_count}, |= {pipe_count}, /= {slash_count}")

# 3. Address abbreviations
print("\n3. ADDRESS ABBREVIATIONS")
addr_abbrevs = {
    'street': ['st\\.', 'street'],
    'road': ['rd\\.', 'road'],
    'avenue': ['ave\\.', 'avenue', 'av\\.'],
    'boulevard': ['blvd\\.', 'boulevard'],
    'drive': ['dr\\.', 'drive'],
    'lane': ['ln\\.', 'lane'],
    'circle': ['cir\\.', 'circle'],
    'court': ['ct\\.', 'court'],
    'place': ['pl\\.', 'place'],
    'highway': ['hwy\\.', 'highway'],
    'parkway': ['pkwy\\.', 'parkway'],
}

for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    addrs = [a for a in df['business_address'].to_list() if a is not None]
    print(f"\n  {name} address abbrevs:")
    for full, abbrevs in addr_abbrevs.items():
        count = sum(1 for a in addrs for ab in abbrevs if re.search(rf'\b{ab}\b', a, re.I))
        if count > 0:
            print(f"    {full}: {count}")

# 4. India-specific patterns
print("\n4. INDIA-SPECIFIC PATTERNS")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    india = df.filter(pl.col('country') == 'India')
    if india.shape[0] == 0:
        continue
    addrs = [a for a in india['business_address'].to_list() if a is not None]
    names = india['business_name'].to_list()
    
    # PIN code pattern (6 digits)
    pincode_count = sum(1 for a in addrs if re.search(r'\b\d{6}\b', a))
    # Landmark patterns
    landmark_count = sum(1 for a in addrs if re.search(r'\b(near|opposite|beside|behind|next to|nearby)\b', a, re.I))
    # Devanagari script
    devanagari_count = sum(1 for n in names if re.search(r'[\u0900-\u097F]', n))
    devanagari_addr = sum(1 for a in addrs if re.search(r'[\u0900-\u097F]', a))
    # Floor/unit patterns
    floor_count = sum(1 for a in addrs if re.search(r'\b(floor|flr|fl\.)\b', a, re.I))
    
    print(f"  {name} India: pincodes={pincode_count}, landmarks={landmark_count}, devanagari_names={devanagari_count}, devanagari_addrs={devanagari_addr}, floors={floor_count}")

# 5. Name variations - token level
print("\n5. NAME TOKEN ANALYSIS")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    names = [n for n in df['business_name'].to_list() if n is not None]
    token_counts = Counter()
    for n in names[:50000]:  # sample
        tokens = n.lower().split()
        for t in tokens:
            # Only ASCII tokens for printing
            if all(ord(c) < 128 for c in t):
                token_counts[t] += 1
    top_tokens = token_counts.most_common(20)
    # Safe print
    safe_str = str([(t, c) for t, c in top_tokens if all(ord(c2) < 128 for c2 in t)])
    print(f"  {name} top 20 tokens: {safe_str}")

# 6. Character encoding issues
print("\n6. ENCODING / UNICODE ISSUES")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    names = [n for n in df['business_name'].to_list() if n is not None]
    addrs = [a for a in df['business_address'].to_list() if a is not None]
    # Non-ASCII
    non_ascii_names = sum(1 for n in names if any(ord(c) > 127 for c in n))
    non_ascii_addrs = sum(1 for a in addrs if any(ord(c) > 127 for c in a))
    # Replacement chars
    repl_names = sum(1 for n in names if '\ufffd' in n)
    repl_addrs = sum(1 for a in addrs if '\ufffd' in a)
    print(f"  {name}: non_ascii_names={non_ascii_names}, non_ascii_addrs={non_ascii_addrs}, repl_names={repl_names}, repl_addrs={repl_addrs}")

# 7. Sample cross-source comparisons for same entities (via ground truth)
print("\n7. CROSS-SOURCE NAME COMPARISONS (via ground truth sample)")
gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t", n_rows=1000)
# Get a few examples with matches
gt_parsed = gt.with_columns([
    pl.col('matched_entity_ids').str.split(',').list.get(0).alias('first_match')
])
sample_matches = gt_parsed.filter(pl.col('first_match').is_not_null()).head(10)

for row in sample_matches.iter_rows(named=True):
    s1_id = row['source1_entity_id']
    match_ids = row['matched_entity_ids'].split(',')
    s1_row = s1.filter(pl.col('entity_id') == s1_id)
    if s1_row.shape[0] > 0:
        s1_name = s1_row['business_name'][0]
        s1_addr = s1_row['business_address'][0] or ""
        print(f"\n  S1: {s1_id} | {s1_name} | {s1_addr}")
        for mid in match_ids[:3]:
            prefix = mid[:2]
            src_df = s2 if prefix == 'S2' else s3
            m_row = src_df.filter(pl.col('entity_id') == mid)
            if m_row.shape[0] > 0:
                m_name = m_row['business_name'][0]
                m_addr = m_row['business_address'][0] or ""
                print(f"    {mid} | {m_name} | {m_addr}")

# 8. Address component analysis
print("\n8. ADDRESS COMPONENT FREQUENCY")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    addrs = [a for a in df['business_address'].to_list() if a is not None]
    # Count commas (component separators)
    comma_counts = [a.count(',') for a in addrs]
    avg_commas = sum(comma_counts) / len(comma_counts)
    # Has state-like patterns (2-letter uppercase for US)
    us_state = sum(1 for a in addrs if re.search(r'\b[A-Z]{2}\b', a))
    print(f"  {name}: avg_commas={avg_commas:.1f}, us_state_pattern={us_state}")

# 9. Country-specific name patterns
print("\n9. COUNTRY-SPECIFIC NAME PATTERNS")
for country in ['US', 'India']:
    print(f"\n  {country} patterns:")
    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        subset = df.filter(pl.col('country') == country)
        if subset.shape[0] == 0:
            continue
        names = [n for n in subset['business_name'].to_list() if n is not None]
        # Common words
        word_counts = Counter()
        for n in names[:20000]:
            for w in n.lower().split():
                if len(w) > 2 and all(ord(c) < 128 for c in w):
                    word_counts[w] += 1
        top = [(t, c) for t, c in word_counts.most_common(10) if all(ord(c2) < 128 for c2 in t)]
        print(f"    {name} top 10: {top}")