import polars as pl
import os
import json

def safe_print(obj):
    if hasattr(obj, 'to_dict'):
        print(json.dumps(obj.to_dict(), ensure_ascii=False, indent=2))
    elif hasattr(obj, 'to_list'):
        print(json.dumps(obj.to_list(), ensure_ascii=False, indent=2))
    else:
        print(obj)

test_dir = "data/6ab10eb3b23ba_student_resource/student_resource/dataset/test"

print("=== TEST DATA ANALYSIS ===")
s1_test = pl.read_csv(os.path.join(test_dir, "test_source1.tsv"), separator="\t", n_rows=100000)
s2_test = pl.read_csv(os.path.join(test_dir, "test_source2.tsv"), separator="\t", n_rows=100000)
s3_test = pl.read_csv(os.path.join(test_dir, "test_source3.tsv"), separator="\t", n_rows=100000)

print(f"Test S1 shape: {s1_test.shape}")
print(f"Test S2 shape: {s2_test.shape}")
print(f"Test S3 shape: {s3_test.shape}")

print("\n=== TEST COUNTRY DISTRIBUTION ===")
for name, df in [("S1_test", s1_test), ("S2_test", s2_test), ("S3_test", s3_test)]:
    vc = df['country'].value_counts()
    safe_print({name: vc.to_dict()})

print("\n=== TEST MISSING VALUES ===")
for name, df in [("S1_test", s1_test), ("S2_test", s2_test), ("S3_test", s3_test)]:
    nc = df.null_count()
    safe_print({name: nc.to_dict()})

print("\n=== TEST NAME LENGTH STATS ===")
for name, df in [("S1_test", s1_test), ("S2_test", s2_test), ("S3_test", s3_test)]:
    lengths = df['business_name'].str.len_chars()
    stats = {"mean": float(lengths.mean()), "median": float(lengths.median()), "min": int(lengths.min()), "max": int(lengths.max())}
    safe_print({name: stats})

print("\n=== TEST ADDRESS LENGTH STATS ===")
for name, df in [("S1_test", s1_test), ("S2_test", s2_test), ("S3_test", s3_test)]:
    lengths = df['business_address'].str.len_chars()
    stats = {"mean": float(lengths.mean()), "median": float(lengths.median()), "min": int(lengths.min()), "max": int(lengths.max())}
    safe_print({name: stats})

# France-specific analysis
print("\n=== FRANCE ZERO-SHOT ANALYSIS ===")
for name, df in [("S1_test", s1_test), ("S2_test", s2_test), ("S3_test", s3_test)]:
    france = df.filter(pl.col('country') == 'France')
    print(f"{name} France count: {france.shape[0]}")
    if france.shape[0] > 0:
        # Show some French business names
        names = france['business_name'].head(10).to_list()
        addrs = france['business_address'].head(10).to_list()
        print(f"  Sample names: {names}")
        print(f"  Sample addrs: {addrs}")

# Compare train vs test distributions
print("\n=== TRAIN vs TEST DISTRIBUTION COMPARISON ===")
train_dir = "data/6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1_train = pl.read_csv(os.path.join(train_dir, "train_source1.tsv"), separator="\t", n_rows=100000)
s2_train = pl.read_csv(os.path.join(train_dir, "train_source2.tsv"), separator="\t", n_rows=100000)
s3_train = pl.read_csv(os.path.join(train_dir, "train_source3.tsv"), separator="\t", n_rows=100000)

for src_name, train_df, test_df in [("S1", s1_train, s1_test), ("S2", s2_train, s2_test), ("S3", s3_train, s3_test)]:
    train_countries = train_df['country'].value_counts().to_dict()['count']
    test_countries = test_df['country'].value_counts().to_dict()['count']
    train_dict = dict(zip(train_df['country'].value_counts().to_dict()['country'], train_countries))
    test_dict = dict(zip(test_df['country'].value_counts().to_dict()['country'], test_countries))
    print(f"{src_name} train: {train_dict}")
    print(f"{src_name} test:  {test_dict}")

# Check for duplicates in test
print("\n=== TEST DUPLICATE CHECK ===")
for name, df in [("S1_test", s1_test), ("S2_test", s2_test), ("S3_test", s3_test)]:
    dup_count = df.shape[0] - df.unique(subset=['entity_id']).shape[0]
    print(f"{name} duplicate entity_ids: {dup_count}")