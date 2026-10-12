#!/usr/bin/env python3
"""
Constant Columns Verification - Check zero variance on full dataset
"""

import pandas as pd
import numpy as np

cache_file = "E:/Projects/thesis/Datasets/interim/cic_3class_dedup.parquet"
print(f"Loading {cache_file}...")
df = pd.read_parquet(cache_file)

print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

# Get numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nNumeric columns: {len(numeric_cols)}")
print("\n" + "="*80)
print("CONSTANT COLUMNS (Zero Variance)")
print("="*80)

constant_cols = []
near_constant_cols = []

for col in numeric_cols:
    # Skip Label if it exists
    if col == 'Label':
        continue
    
    # Check variance
    var = df[col].var()
    
    # Also check for all-same values (including inf/nan handling)
    unique_vals = df[col].dropna().unique()
    
    if var == 0 or len(unique_vals) == 1:
        constant_cols.append((col, var, len(unique_vals)))
        print(f"  {col}: variance={var:.6f}, unique_values={len(unique_vals)}")
    elif len(unique_vals) <= 5:
        # Near-constant (very few unique values)
        near_constant_cols.append((col, var, len(unique_vals)))

print(f"\n{'='*80}")
print("NEAR-CONSTANT COLUMNS (≤5 unique values)")
print("="*80)

for col, var, n_unique in near_constant_cols:
    print(f"  {col}: variance={var:.6f}, unique_values={n_unique}")

print(f"\n{'='*80}")
print("SUMMARY")
print("="*80)

print(f"\nTotal constant columns (variance=0): {len(constant_cols)}")
print(f"Near-constant columns (≤5 unique): {len(near_constant_cols)}")
print(f"Total to remove: {len(constant_cols)}")

# Calculate remaining features
total_cols = len(df.columns)  # 79 (includes Label)
label_col = 1 if 'Label' in df.columns else 0
total_features = total_cols - label_col
remaining_features = total_features - len(constant_cols)

print(f"\nOriginal: {total_cols} columns ({total_features} features + {label_col} Label)")
print(f"After removing constants: {remaining_features} features")

# List constant column names
print(f"\nConstant column names:")
for col, var, n_unique in constant_cols:
    print(f"  - {col}")

print("\n✅ Constant columns verification complete")
