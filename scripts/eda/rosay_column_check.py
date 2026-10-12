#!/usr/bin/env python3
"""
Rosay Column Pair Check - Are these duplicate columns?
Checks 4 pairs for value-identical (not just correlation)
"""

import pandas as pd
import numpy as np

cache_file = "E:/Projects/thesis/Datasets/interim/cic_3class_dedup.parquet"
print(f"Loading {cache_file}...")
df = pd.read_parquet(cache_file)

# Rosay 2022 reported 4 duplicate column pairs
pairs = [
    ("Flow Bytes/s", "Flow Packets/s"),
    ("Fwd Header Length.1", "Fwd Header Length"),
    ("Subflow Fwd Bytes", "Total Length of Fwd Packets"),
    ("Subflow Bwd Bytes", "Total Length of Bwd Packets"),
]

print("\n" + "="*80)
print("ROSAY COLUMN PAIR VERIFICATION")
print("="*80)

for col1, col2 in pairs:
    print(f"\n{'='*80}")
    print(f"Pair: '{col1}' vs '{col2}'")
    
    if col1 not in df.columns:
        print(f"  ❌ '{col1}' NOT FOUND in dataset")
        continue
    if col2 not in df.columns:
        print(f"  ❌ '{col2}' NOT FOUND in dataset")
        continue
    
    # Get values
    v1 = df[col1].copy()
    v2 = df[col2].copy()
    
    # Count identical values
    identical_mask = (v1 == v2) | (pd.isna(v1) & pd.isna(v2))
    n_identical = identical_mask.sum()
    n_total = len(df)
    pct_identical = 100 * n_identical / n_total
    
    # Correlation (for numeric)
    if pd.api.types.is_numeric_dtype(v1) and pd.api.types.is_numeric_dtype(v2):
        # Remove inf/nan for correlation
        mask_valid = np.isfinite(v1) & np.isfinite(v2)
        if mask_valid.sum() > 10:
            corr = np.corrcoef(v1[mask_valid], v2[mask_valid])[0, 1]
        else:
            corr = np.nan
    else:
        corr = np.nan
    
    print(f"  Total rows: {n_total:,}")
    print(f"  Identical values: {n_identical:,} ({pct_identical:.2f}%)")
    if not np.isnan(corr):
        print(f"  Pearson correlation: {corr:.6f}")
    
    if pct_identical == 100.0:
        print(f"  ✅ DUPLICATE - 100% identical")
    elif pct_identical > 99.9:
        print(f"  ⚠️  NEAR-DUPLICATE - {pct_identical:.2f}% identical")
    elif not np.isnan(corr) and abs(corr) > 0.99:
        print(f"  ⚠️  HIGHLY CORRELATED (r={corr:.4f}) but NOT value-identical")
    else:
        print(f"  ✅ DISTINCT - Not duplicates")
    
    # Sample differences (if not 100% identical)
    if pct_identical < 100.0:
        diff_mask = ~identical_mask
        if diff_mask.sum() > 0:
            sample_idx = df[diff_mask].head(3).index
            print(f"\n  Sample differences (first 3):")
            for idx in sample_idx:
                print(f"    Row {idx}: {col1}={v1.iloc[idx]}, {col2}={v2.iloc[idx]}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n100% identical = TRUE duplicates (one should be removed)")
print(">99.9% identical = Rounding artifacts, treat as duplicates")
print("High correlation but different values = Related but distinct features")

print("\n✅ Rosay column pair check complete")
