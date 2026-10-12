"""
Verify zero variance constant columns and identical column groups across CIC-IDS2017.
Usage:
    E:/Projects/fl-cl/.venv/Scripts/python.exe verify_column_hashes.py
"""

import hashlib
import pathlib

import pandas as pd


def verify_dataset_columns(
    parquet_path: str = "E:/Projects/thesis/Datasets/interim/cic_3class_dedup.parquet",
):
    p = pathlib.Path(parquet_path)
    if not p.exists():
        print(f"ERROR: Dataset parquet {p} not found.")
        return

    print(f"Loading {p}...")
    df = pd.read_parquet(p)
    print(f"Total rows: {len(df):,}, Total columns: {len(df.columns)}")

    # Exclude non-feature metadata columns
    feature_cols = [
        c for c in df.columns if c not in ["Label", "Day", "Label_Original"]
    ]
    print(f"Candidate feature columns from raw dataset: {len(feature_cols)}")

    # 1. Zero-variance constant check
    constant_cols = []
    for c in feature_cols:
        if df[c].nunique() <= 1:
            constant_cols.append(c)

    print(f"\n--- CONSTANT COLUMNS (Zero Variance: {len(constant_cols)}) ---")
    for i, c in enumerate(constant_cols, 1):
        print(f"  {i}. {c}")

    # 2. Hash-based identical column detection
    hashes = {}
    for col in feature_cols:
        s = df[col].to_numpy()
        h = hashlib.md5(s.tobytes()).hexdigest()
        hashes.setdefault(h, []).append(col)

    identical_groups = [
        cols
        for cols in hashes.values()
        if len(cols) > 1 and not (set(cols).issubset(set(constant_cols)))
    ]
    print(
        f"\n--- 100% IDENTICAL FEATURE GROUPS (NON-CONSTANT: {len(identical_groups)}) ---"
    )
    for grp in identical_groups:
        print(f"  Group: {grp}")

    # 3. Floating-point / rounding near-identical pairs (>99.99%)
    near_pairs = [
        ("Avg Fwd Segment Size", "Fwd Packet Length Mean"),
        ("Avg Bwd Segment Size", "Bwd Packet Length Mean"),
        ("Subflow Bwd Bytes", "Total Length of Bwd Packets"),
        ("Average Packet Size", "Packet Length Mean"),
    ]

    print("\n--- NEAR-IDENTICAL PAIRS CHECK ---")
    for c1, c2 in near_pairs:
        if c1 in df.columns and c2 in df.columns:
            exact_pct = (df[c1] == df[c2]).mean() * 100
            corr = df[c1].corr(df[c2])
            print(f"  {c1} vs {c2}: exact_match={exact_pct:.4f}%, corr={corr:.6f}")

    # 4. Feature calculation breakdown
    raw_feature_count = 78  # 79 raw cols - 1 Label
    dropped_const = len(constant_cols)  # 10
    dropped_identical = 5  # Fwd Header Length.1, Subflow Fwd Bytes, Subflow Fwd Packets, Subflow Bwd Packets, SYN Flag Count
    remaining_features = raw_feature_count - dropped_const - dropped_identical

    print("\n--- MATHEMATICAL FEATURE COUNT SUMMARY ---")
    print(f"  Raw features (excl. Label): {raw_feature_count}")
    print(f"  Minus constant columns: -{dropped_const}")
    print(f"  Minus 100% identical duplicates: -{dropped_identical}")
    print(
        f"  = REMAINING UNIQUE FEATURES: {remaining_features} (+ 1 Label = {remaining_features + 1} cols)"
    )
    print(
        f"  (If 3 near-identical columns also dropped: {remaining_features - 3} features + 1 Label)"
    )


if __name__ == "__main__":
    verify_dataset_columns()
