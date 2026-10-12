#!/usr/bin/env python3
"""
Column Pair Verification - Requested Pairs + Rosay Pairs
Checks value-identical and correlation for specific column pairs
"""

import numpy as np
import pandas as pd

cache_file = "E:/Projects/thesis/Datasets/interim/cic_3class_dedup.parquet"
print(f"Loading {cache_file}...")
df = pd.read_parquet(cache_file)

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Requested pairs + Rosay pairs
pairs_to_check = [
    # Requested pairs
    ("Average Packet Size", "Packet Length Mean"),
    ("Fwd Segment Size Avg", "Fwd Packet Length Mean"),
    ("Bwd Segment Size Avg", "Bwd Packet Length Mean"),
    ("Fwd Header Length", "Fwd Header Length.1"),
    # Rosay pairs
    ("Flow Bytes/s", "Flow Packets/s"),
    ("Subflow Fwd Bytes", "Total Length of Fwd Packets"),
    ("Subflow Bwd Bytes", "Total Length of Bwd Packets"),
]

print("\n" + "=" * 100)
print("COLUMN PAIR VERIFICATION - Value Identical + Correlation")
print("=" * 100)

results = []

for col1, col2 in pairs_to_check:
    print(f"\n{'-'*100}")
    print(f"Pair: '{col1}' vs '{col2}'")

    # Check if columns exist
    if col1 not in df.columns:
        print(f"  ❌ '{col1}' NOT FOUND")
        results.append((col1, col2, "NOT_FOUND_1", None, None))
        continue
    if col2 not in df.columns:
        print(f"  ❌ '{col2}' NOT FOUND")
        results.append((col1, col2, "NOT_FOUND_2", None, None))
        continue

    v1 = df[col1].copy()
    v2 = df[col2].copy()

    # Value-identical check
    identical_mask = (v1 == v2) | (pd.isna(v1) & pd.isna(v2))
    n_identical = identical_mask.sum()
    pct_identical = 100.0 * n_identical / len(df)

    # Correlation (numeric only)
    if pd.api.types.is_numeric_dtype(v1) and pd.api.types.is_numeric_dtype(v2):
        mask_valid = np.isfinite(v1) & np.isfinite(v2)
        if mask_valid.sum() > 10:
            corr = np.corrcoef(v1[mask_valid], v2[mask_valid])[0, 1]
        else:
            corr = np.nan
    else:
        corr = np.nan

    print(f"  Identical values: {n_identical:,} / {len(df):,} ({pct_identical:.2f}%)")
    if not np.isnan(corr):
        print(f"  Pearson correlation: {corr:.6f}")

    # Classification
    if pct_identical == 100.0:
        status = "100% DUPLICATE"
        print(f"  ✅ {status} - Remove one")
    elif pct_identical > 99.9:
        status = "99.9%+ DUPLICATE"
        print(f"  ⚠️  {status} - Rounding artifacts")
    elif not np.isnan(corr) and abs(corr) > 0.99:
        status = f"HIGHLY CORRELATED (r={corr:.4f})"
        print(f"  ⚠️  {status} - Related but distinct")
    else:
        status = "DISTINCT"
        print(f"  ✅ {status}")

    results.append(
        (col1, col2, status, pct_identical, corr if not np.isnan(corr) else None)
    )

# Summary table
print("\n" + "=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(f"\n{'Col1':<40} {'Col2':<40} {'Status':<30} {'Identical%':>10}")
print("-" * 100)
for col1, col2, status, pct, corr in results:
    pct_str = f"{pct:.2f}" if pct is not None else "N/A"
    print(f"{col1:<40} {col2:<40} {status:<30} {pct_str:>10}")

# Count duplicates
n_duplicates = sum(1 for _, _, status, _, _ in results if "DUPLICATE" in status)
print(f"\nTotal duplicate pairs: {n_duplicates}")

print("\n✅ Column pair verification complete")
