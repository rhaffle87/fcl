#!/usr/bin/env python3
"""
Full EDA on CIC-IDS2017 and LYCOS-IDS2017 using stdlib only
Date: 2026-10-02
Command: python scripts/eda/eda_full_stdlib.py
"""

import csv
from collections import Counter, defaultdict
import sys
import os

print("="*80)
print("FULL EDA - CIC-IDS2017 & LYCOS-IDS2017")
print("Date: 2026-10-02")
print("="*80)

datasets_base = "E:/Projects/thesis/Datasets"

# CIC-IDS2017 files
cic_files = [
    ("CIC-IDS2017/Monday-WorkingHours.pcap_ISCX.csv", "Monday (Normal)"),
    ("CIC-IDS2017/Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday (Normal+SSH+FTP)"),
    ("CIC-IDS2017/Wednesday-workingHours.pcap_ISCX.csv", "Wednesday (Normal+DoS)")
]

# LYCOS-IDS2017 files (corrected versions)
lycos_files = [
    ("LYCOS-IDS2017/pcap_lycos/Monday-WorkingHours.pcap_lycos/Monday-WorkingHours.pcap_lycos.csv", "Monday LYCOS"),
    ("LYCOS-IDS2017/pcap_lycos/Tuesday-WorkingHours.pcap_lycos/Tuesday-WorkingHours.pcap_lycos.csv", "Tuesday LYCOS"),
    ("LYCOS-IDS2017/pcap_lycos/Wednesday-WorkingHours.pcap_lycos/Wednesday-WorkingHours.pcap_lycos.csv", "Wednesday LYCOS")
]

def analyze_csv(filepath, description):
    """Analyze a CSV file completely"""
    
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"File: {os.path.basename(filepath)}")
    print("="*80)
    
    if not os.path.exists(filepath):
        print(f"  ❌ File not found: {filepath}")
        return None
    
    # Full scan
    print("Scanning entire file...")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        # Read header
        header_line = f.readline().strip()
        fieldnames = [col.strip() for col in header_line.split(',')]
        
        # Find label column
        label_idx = None
        for i, col in enumerate(fieldnames):
            if col.strip() == 'Label':
                label_idx = i
                break
        
        if label_idx is None:
            print(f"  ❌ Label column not found!")
            return None
        
        # Scan all rows
        label_counts = Counter()
        row_count = 0
        duplicate_check = set()
        duplicate_count = 0
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) > label_idx:
                label = parts[label_idx].strip()
                if label:
                    label_counts[label] += 1
                
                # Check duplicates (hash first 10 columns for speed)
                row_hash = ','.join(parts[:min(10, len(parts))])
                if row_hash in duplicate_check:
                    duplicate_count += 1
                else:
                    duplicate_check.add(row_hash)
                
                row_count += 1
                
                # Progress indicator
                if row_count % 100000 == 0:
                    print(f"  ... {row_count:,} rows processed")
    
    # Report
    print(f"\n📊 Results:")
    print(f"  Total rows: {row_count:,}")
    print(f"  Total columns: {len(fieldnames)}")
    print(f"  Duplicate rows (approx): {duplicate_count:,}")
    
    # Check duplicate column names
    col_counts = Counter(fieldnames)
    dup_cols = [col for col, cnt in col_counts.items() if cnt > 1]
    if dup_cols:
        print(f"  ⚠️ Duplicate columns: {dup_cols}")
    
    print(f"\n📋 Label Distribution:")
    total = sum(label_counts.values())
    for label, count in label_counts.most_common():
        pct = (count / total) * 100 if total > 0 else 0
        print(f"    {label:30s}: {count:8,} ({pct:5.2f}%)")
    
    return {
        'file': os.path.basename(filepath),
        'description': description,
        'rows': row_count,
        'columns': len(fieldnames),
        'labels': dict(label_counts),
        'duplicates': duplicate_count,
        'dup_columns': dup_cols
    }

# Analyze CIC-IDS2017
print("\n" + "="*80)
print("SECTION 1: CIC-IDS2017 ORIGINAL")
print("="*80)

cic_results = []
for filepath, desc in cic_files:
    full_path = f"{datasets_base}/{filepath}"
    result = analyze_csv(full_path, desc)
    if result:
        cic_results.append(result)

# Analyze LYCOS-IDS2017
print("\n" + "="*80)
print("SECTION 2: LYCOS-IDS2017 CORRECTED")
print("="*80)

lycos_results = []
for filepath, desc in lycos_files:
    full_path = f"{datasets_base}/{filepath}"
    result = analyze_csv(full_path, desc)
    if result:
        lycos_results.append(result)

# Summary comparison
print("\n" + "="*80)
print("COMPARISON: CIC vs LYCOS")
print("="*80)

print("\n| Metric | CIC-IDS2017 | LYCOS-IDS2017 |")
print("|--------|-------------|---------------|")
print(f"| Total rows | {sum(r['rows'] for r in cic_results):,} | {sum(r['rows'] for r in lycos_results):,} |")
print(f"| Files analyzed | {len(cic_results)} | {len(lycos_results)} |")

# Label comparison
all_cic_labels = set()
all_lycos_labels = set()
for r in cic_results:
    all_cic_labels.update(r['labels'].keys())
for r in lycos_results:
    all_lycos_labels.update(r['labels'].keys())

print(f"\n📋 Unique labels:")
print(f"  CIC-IDS2017: {len(all_cic_labels)} labels")
print(f"  LYCOS-IDS2017: {len(all_lycos_labels)} labels")

print("\n✅ Full EDA complete. Results saved for report.")
