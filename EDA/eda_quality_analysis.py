import polars as pl
import os
import json

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
s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t")
s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t")
s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t")
gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t")

print("=== FULL DATASET QUALITY ANALYSIS ===")
print(f"S1: {s1.shape[0]:,} rows")
print(f"S2: {s2.shape[0]:,} rows")
print(f"S3: {s3.shape[0]:,} rows")
print(f"GT: {gt.shape[0]:,} rows")

# 1. Missing values - full dataset
print("\n1. MISSING VALUES (FULL DATASET)")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    nc = df.null_count()
    total = df.shape[0]
    safe_print({name: {col: f"{nc[col][0]:,} ({nc[col][0]/total*100:.1f}%)" for col in nc.columns}})

# 2. Duplicate entity_ids
print("\n2. DUPLICATE ENTITY_ID CHECK")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    unique_count = df.unique(subset=['entity_id']).shape[0]
    total = df.shape[0]
    print(f"  {name}: total={total:,}, unique={unique_count:,}, duplicates={total-unique_count:,}")

# 3. Empty strings
print("\n3. EMPTY STRING CHECK")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    empty_name = df.filter(pl.col('business_name') == '').shape[0]
    empty_addr = df.filter(pl.col('business_address') == '').shape[0]
    empty_country = df.filter(pl.col('country') == '').shape[0]
    print(f"  {name}: empty_name={empty_name:,}, empty_addr={empty_addr:,}, empty_country={empty_country:,}")

# 4. Ground truth coverage
print("\n4. GROUND TRUTH COVERAGE")
s1_ids = set(s1['entity_id'].to_list())
gt_s1_ids = set(gt['source1_entity_id'].to_list())
print(f"  S1 entities in GT: {len(gt_s1_ids):,} / {len(s1_ids):,} ({len(gt_s1_ids)/len(s1_ids)*100:.1f}%)")
print(f"  S1 entities NOT in GT: {len(s1_ids - gt_s1_ids):,}")

# Check if GT has S1 IDs not in S1
print(f"  GT S1 IDs not in S1: {len(gt_s1_ids - s1_ids):,}")

# 5. Match ID validation - check if matched IDs exist in S2/S3
print("\n5. MATCHED ID EXISTENCE CHECK (sample 10k)")
gt_sample = gt.head(10000)
s2_ids = set(s2['entity_id'].to_list())
s3_ids = set(s3['entity_id'].to_list())

missing_s2 = 0
missing_s3 = 0
total_matched = 0
for row in gt_sample.iter_rows(named=True):
    matches = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] not in (None, '') else []
    total_matched += len(matches)
    for mid in matches:
        if mid.startswith('S2-') and mid not in s2_ids:
            missing_s2 += 1
        elif mid.startswith('S3-') and mid not in s3_ids:
            missing_s3 += 1

print(f"  Total matched IDs in sample: {total_matched:,}")
print(f"  Missing in S2: {missing_s2:,}")
print(f"  Missing in S3: {missing_s3:,}")

# 6. Singleton analysis (S1 entities with no matches)
print("\n6. SINGLETON ANALYSIS")
gt_parsed = gt.with_columns([
    pl.col('matched_entity_ids').str.split(',').list.len().alias('num_matches')
])
# Empty string -> 0 matches
gt_parsed = gt_parsed.with_columns([
    pl.when(pl.col('matched_entity_ids') == '').then(0).otherwise(pl.col('num_matches')).alias('num_matches_fixed')
])

singleton_count = gt_parsed.filter(pl.col('num_matches_fixed') == 0).shape[0]
multi_match = gt_parsed.filter(pl.col('num_matches_fixed') > 1).shape[0]
single_match = gt_parsed.filter(pl.col('num_matches_fixed') == 1).shape[0]

print(f"  Singletons (0 matches): {singleton_count:,} ({singleton_count/gt.shape[0]*100:.1f}%)")
print(f"  Single match (1): {single_match:,} ({single_match/gt.shape[0]*100:.1f}%)")
print(f"  Multi-match (>1): {multi_match:,} ({multi_match/gt.shape[0]*100:.1f}%)")

# By country
gt_with_country = gt_parsed.join(s1.select(['entity_id', 'country']), left_on='source1_entity_id', right_on='entity_id', how='left')
for country in ['US', 'India']:
    subset = gt_with_country.filter(pl.col('country') == country)
    if subset.shape[0] > 0:
        s = subset.filter(pl.col('num_matches_fixed') == 0).shape[0]
        m1 = subset.filter(pl.col('num_matches_fixed') == 1).shape[0]
        mm = subset.filter(pl.col('num_matches_fixed') > 1).shape[0]
        print(f"  {country}: singletons={s:,} ({s/subset.shape[0]*100:.1f}%), 1-match={m1:,}, multi={mm:,}")

# 7. Source 2/3 match distribution
print("\n7. MATCH SOURCE DISTRIBUTION (sample)")
s2_match_count = 0
s3_match_count = 0
for row in gt_sample.iter_rows(named=True):
    matches = row['matched_entity_ids'].split(',') if row['matched_entity_ids'] else []
    for mid in matches:
        if mid.startswith('S2-'):
            s2_match_count += 1
        elif mid.startswith('S3-'):
            s3_match_count += 1
print(f"  S2 matches: {s2_match_count:,} ({s2_match_count/total_matched*100:.1f}%)")
print(f"  S3 matches: {s3_match_count:,} ({s3_match_count/total_matched*100:.1f}%)")

# 8. Memory footprint estimation
print("\n8. MEMORY ESTIMATION")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    mem_mb = df.estimated_size() / (1024**2)
    print(f"  {name}: ~{mem_mb:.0f} MB in memory")

# 9. Address null handling - how many null addresses per country
print("\n9. NULL ADDRESSES BY COUNTRY")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    for country in ['US', 'India']:
        subset = df.filter(pl.col('country') == country)
        null_addr = subset.filter(pl.col('business_address').is_null()).shape[0]
        total_c = subset.shape[0]
        if total_c > 0:
            print(f"  {name} {country}: {null_addr:,} null addresses ({null_addr/total_c*100:.1f}%)")

# 10. Name/address length distributions by country
print("\n10. LENGTH DISTRIBUTIONS BY COUNTRY")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    for country in ['US', 'India']:
        subset = df.filter(pl.col('country') == country)
        if subset.shape[0] == 0:
            continue
        name_len = subset['business_name'].str.len_chars()
        addr_len = subset['business_address'].str.len_chars()
        print(f"  {name} {country}: name_len median={name_len.median()}, addr_len median={addr_len.median()}")