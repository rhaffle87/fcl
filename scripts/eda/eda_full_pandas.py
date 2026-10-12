#!/usr/bin/env python3
"""
Full EDA on CIC-IDS2017 with pandas - ALL columns
Date: 2026-10-02
Command: E:/Projects/fl-cl/.venv/Scripts/python.exe scripts/eda/eda_full_pandas.py
"""

import pandas as pd
import numpy as np
from collections import Counter

print("="*80)
print("FULL EDA - CIC-IDS2017 (PANDAS - ALL COLUMNS)")
print("Date: 2026-10-02 15:00 UTC+7")
print("="*80)

datasets_base = "E:/Projects/thesis/Datasets/CIC-IDS2017"

files = [
    ("Monday-WorkingHours.pcap_ISCX.csv", "Monday"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "Wednesday")
]

all_results = {}

for filename, day in files:
    filepath = f"{datasets_base}/{filename}"
    print(f"\n{'='*80}")
    print(f"{day} - {filename}")
    print("="*80)
    
    # Load with proper column stripping
    df = pd.read_csv(filepath, skipinitialspace=True)
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")
    
    # 1. Label distribution
    print(f"\n📋 Label Distribution (before deduplication):")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {label:30s}: {count:8,} ({pct:5.2f}%)")
    
    # 2. Duplicates (ALL columns)
    print(f"\n🔍 Duplicate Rows (ALL {len(df.columns)} columns):")
    dup_mask = df.duplicated(keep='first')
    dup_count = dup_mask.sum()
    print(f"  Duplicate rows: {dup_count:,} ({dup_count/len(df)*100:.2f}%)")
    
    # After dedup
    df_dedup = df.drop_duplicates()
    print(f"  After deduplication: {len(df_dedup):,} rows")
    
    print(f"\n📋 Label Distribution (after deduplication):")
    label_counts_dedup = df_dedup['Label'].value_counts()
    for label, count in label_counts_dedup.items():
        pct = (count / len(df_dedup)) * 100
        print(f"  {label:30s}: {count:8,} ({pct:5.2f}%)")
    
    # 3. NaN and Inf counts per column (top 10)
    print(f"\n⚠️  NaN and Infinity counts (top 10 columns):")
    nan_counts = df.isna().sum()
    inf_counts = np.isinf(df.select_dtypes(include=[np.number])).sum()
    
    problem_cols = pd.DataFrame({
        'NaN': nan_counts,
        'Inf': inf_counts
    })
    problem_cols['Total'] = problem_cols['NaN'] + problem_cols['Inf']
    problem_cols = problem_cols[problem_cols['Total'] > 0].sort_values('Total', ascending=False)
    
    if len(problem_cols) > 0:
        for col in problem_cols.head(10).index:
            print(f"  {col:40s}: NaN={problem_cols.loc[col, 'NaN']:8,} Inf={problem_cols.loc[col, 'Inf']:8,}")
    else:
        print("  No NaN or Inf values found")
    
    # 4. Constant columns
    print(f"\n🔍 Constant/Near-constant columns:")
    const_cols = []
    for col in df.select_dtypes(include=[np.number]).columns:
        nunique = df[col].nunique()
        if nunique <= 1:
            const_cols.append(col)
    
    if const_cols:
        print(f"  {len(const_cols)} constant columns: {const_cols}")
    else:
        print("  No constant columns")
    
    # 5. Duplicate columns by VALUE (check Rosay's 4 pairs)
    print(f"\n🔍 Duplicate Columns (by VALUE - Rosay's 4 pairs):")
    
    pairs_to_check = [
        ("Average Packet Size", "Packet Length Mean"),
        ("Fwd Segment Size Avg", "Fwd Packet Length Mean"),
        ("Bwd Segment Size Avg", "Bwd Packet Length Mean"),
        ("Fwd Header Length", "Fwd Header Length")  # Name duplicate
    ]
    
    dup_value_pairs = []
    for col1, col2 in pairs_to_check:
        if col1 in df.columns and col2 in df.columns:
            # Check if values are identical
            if df[col1].equals(df[col2]):
                print(f"  ✅ IDENTICAL: '{col1}' == '{col2}'")
                dup_value_pairs.append((col1, col2))
            else:
                # Check correlation
                corr = df[col1].corr(df[col2])
                print(f"  ⚠️  SIMILAR: '{col1}' vs '{col2}' (corr={corr:.4f})")
    
    # Check for duplicate column NAMES
    dup_names = [col for col in df.columns if df.columns.tolist().count(col) > 1]
    if dup_names:
        print(f"  ⚠️  Duplicate column names: {list(set(dup_names))}")
    
    # 6. Destination Port analysis per class
    print(f"\n🚨 Destination Port Distribution (Label Leakage Check):")
    
    for label in df['Label'].unique():
        subset = df[df['Label'] == label]
        if 'Destination Port' in df.columns:
            top_ports = subset['Destination Port'].value_counts().head(5)
            print(f"\n  {label} (n={len(subset):,}):")
            for port, count in top_ports.items():
                pct = (count / len(subset)) * 100
                print(f"    Port {port}: {count:8,} ({pct:5.2f}%)")
    
    # Store results
    all_results[day] = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'duplicates': dup_count,
        'after_dedup': len(df_dedup),
        'labels_before': dict(label_counts),
        'labels_after': dict(label_counts_dedup),
        'nan_inf': problem_cols.to_dict() if len(problem_cols) > 0 else {},
        'constant_cols': const_cols,
        'dup_value_pairs': dup_value_pairs
    }

# Summary
print(f"\n{'='*80}")
print("SUMMARY - CIC-IDS2017")
print("="*80)

total_before = sum(r['total_rows'] for r in all_results.values())
total_after = sum(r['after_dedup'] for r in all_results.values())
total_dup = sum(r['duplicates'] for r in all_results.values())

print(f"\nTotal rows BEFORE deduplication: {total_before:,}")
print(f"Total duplicates (ALL columns): {total_dup:,} ({total_dup/total_before*100:.2f}%)")
print(f"Total rows AFTER deduplication: {total_after:,}")

print(f"\n✅ Full EDA complete (pandas, all columns)")
