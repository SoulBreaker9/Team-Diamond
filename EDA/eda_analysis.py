import polars as pl
import os
import sys
import json

def safe_print(obj):
    """Convert polars objects to JSON-serializable and print"""
    if hasattr(obj, 'to_dict'):
        print(json.dumps(obj.to_dict(), ensure_ascii=False, indent=2))
    elif hasattr(obj, 'to_list'):
        print(json.dumps(obj.to_list(), ensure_ascii=False, indent=2))
    else:
        print(obj)

# Resilient dataset path resolution
def find_dataset_dir():
    candidates = [
        os.path.join(REPO_ROOT, "data/6ab10eb3b23ba_student_resource/student_resource/dataset"),
        os.path.join(os.path.dirname(REPO_ROOT), "6ab10eb3b23ba_student_resource/student_resource/dataset"),
        os.path.join(os.path.dirname(REPO_ROOT), "data/6ab10eb3b23ba_student_resource/student_resource/dataset"),
        os.path.join(REPO_ROOT, "dataset"),
        os.path.join(os.path.dirname(REPO_ROOT), "dataset")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train")):
            return c
    return candidates[0]

DATASET_DIR = find_dataset_dir()
train_dir = os.path.join(DATASET_DIR, "train")
test_dir = os.path.join(DATASET_DIR, "test")

print("=== FILE SIZES ===")
for f in os.listdir(train_dir):
    path = os.path.join(train_dir, f)
    size = os.path.getsize(path) / (1024**3)
    print(f"Train: {f} - {size:.2f} GB")

for f in os.listdir(test_dir):
    path = os.path.join(test_dir, f)
    size = os.path.getsize(path) / (1024**3)
    print(f"Test: {f} - {size:.2f} GB")

# Load sample for analysis
print("\n=== SAMPLE LOADING (first 100k rows each) ===")
s1 = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t", n_rows=100000)
s2 = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t", n_rows=100000)
s3 = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t", n_rows=100000)
gt = pl.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), separator="\t", n_rows=100000)

print(f"\nSource1 shape: {s1.shape}")
print(f"Source2 shape: {s2.shape}")
print(f"Source3 shape: {s3.shape}")
print(f"Ground truth shape: {gt.shape}")

print("\n=== COLUMNS ===")
print(f"S1: {s1.columns}")
print(f"S2: {s2.columns}")
print(f"S3: {s3.columns}")
print(f"GT: {gt.columns}")

print("\n=== COUNTRY DISTRIBUTION (sample) ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    vc = df['country'].value_counts()
    safe_print({name: vc.to_dict()})

print("\n=== MISSING VALUES (sample) ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    nc = df.null_count()
    safe_print({name: nc.to_dict()})

print("\n=== ENTITY_ID PREFIX CHECK ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    vc = df['entity_id'].str.slice(0, 3).value_counts()
    safe_print({name: vc.to_dict()})

print("\n=== NAME LENGTH STATS ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    lengths = df['business_name'].str.len_chars()
    stats = {
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "min": int(lengths.min()),
        "max": int(lengths.max())
    }
    safe_print({name: stats})

print("\n=== ADDRESS LENGTH STATS ===")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    lengths = df['business_address'].str.len_chars()
    stats = {
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "min": int(lengths.min()),
        "max": int(lengths.max())
    }
    safe_print({name: stats})

print("\n=== GROUND TRUTH ANALYSIS ===")
print(f"GT rows: {gt.shape[0]}")
gt_parsed = gt.with_columns([
    pl.col('matched_entity_ids').str.split(',').list.len().alias('num_matches')
])
vc = gt_parsed['num_matches'].value_counts().sort('num_matches')
safe_print({"match_cardinality": vc.to_dict()})
singletons = int((gt_parsed['num_matches'] == 0).sum())
avg_matches = float(gt_parsed['num_matches'].mean())
print(f"Singletons (0 matches): {singletons}")
print(f"Avg matches per S1: {avg_matches:.2f}")

# Country breakdown in ground truth
print("\n=== GT COUNTRY BREAKDOWN ===")
gt_with_country = gt.join(s1.select(['entity_id', 'country']), left_on='source1_entity_id', right_on='entity_id', how='left')
vc_country = gt_with_country['country'].value_counts()
safe_print({"gt_by_country": vc_country.to_dict()})

# Match cardinality by country
for country in ['US', 'India']:
    subset = gt_parsed.filter(gt_with_country['country'] == country)
    if subset.shape[0] > 0:
        print(f"  {country}: avg_matches={float(subset['num_matches'].mean()):.2f}, singletons={(subset['num_matches'] == 0).sum()}")