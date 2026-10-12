#!/usr/bin/env python3
"""
Port leakage test - FAST version with chunk loading
Date: 2026-10-02
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report

print("="*80)
print("PORT LEAKAGE TEST - Fast Sampling Version")
print("="*80)

np.random.seed(42)
datasets_base = "E:/Projects/thesis/Datasets/CIC-IDS2017"

# Sample directly from CSVs (50K rows each, before dedup)
sample_size = 50000

print("\nLoading samples (50K per file)...")
dfs = []
for fname in ["Monday-WorkingHours.pcap_ISCX.csv", 
              "Tuesday-WorkingHours.pcap_ISCX.csv",
              "Wednesday-workingHours.pcap_ISCX.csv"]:
    df = pd.read_csv(f"{datasets_base}/{fname}", nrows=sample_size, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)
df_all = df_all.drop_duplicates()

# Filter to 3 classes + merge DoS
df_all['Label'] = df_all['Label'].replace({
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS'
})

df_3class = df_all[df_all['Label'].isin(['BENIGN', 'SSH-Patator', 'DoS'])].copy()

print(f"\nTotal after dedup: {len(df_3class):,}")
print("\nClass distribution:")
print(df_3class['Label'].value_counts())

# Port distribution
print(f"\n{'='*80}")
print("DESTINATION PORT DISTRIBUTION BY CLASS")
print("="*80)

for label in ['BENIGN', 'SSH-Patator', 'DoS']:
    subset = df_3class[df_3class['Label'] == label]
    if len(subset) == 0:
        continue
    top_ports = subset['Destination Port'].value_counts().head(5)
    print(f"\n{label} (n={len(subset):,}):")
    for port, count in top_ports.items():
        pct = (count / len(subset)) * 100
        print(f"  Port {int(port):5d}: {count:6,} ({pct:5.2f}%)")

# Balanced sample
min_count = df_3class['Label'].value_counts().min()
sample_per_class = min(3000, min_count)

print(f"\n{'='*80}")
print(f"Sampling {sample_per_class:,} per class (balanced, seed=42)")
print("="*80)

df_balanced = pd.concat([
    df_3class[df_3class['Label'] == label].sample(n=sample_per_class, random_state=42)
    for label in ['BENIGN', 'SSH-Patator', 'DoS']
])

# Prepare data
X_port = df_balanced[['Destination Port']].values
y = df_balanced['Label'].values

# Train/test split (70/30)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_port, y, test_size=0.3, random_state=42, stratify=y
)

# Decision Tree with ONLY port
print("\nTraining Decision Tree (depth=3) with ONLY Destination Port...")
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

# Metrics
f1_macro = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred, labels=['BENIGN', 'SSH-Patator', 'DoS'])

print(f"\n{'='*80}")
print("RESULTS - Port-Only Classification")
print("="*80)

print(f"\nMacro-F1: {f1_macro:.4f}")

print("\nConfusion Matrix:")
print("                Predicted")
print("Actual     BENIGN    SSH    DoS")
print(f"BENIGN     {cm[0,0]:6d} {cm[0,1]:6d} {cm[0,2]:6d}")
print(f"SSH        {cm[1,0]:6d} {cm[1,1]:6d} {cm[1,2]:6d}")
print(f"DoS        {cm[2,0]:6d} {cm[2,1]:6d} {cm[2,2]:6d}")

print("\nPer-class metrics:")
report = classification_report(y_test, y_pred, output_dict=True)
for label in ['BENIGN', 'SSH-Patator', 'DoS']:
    if label in report:
        print(f"  {label:15s}: Precision={report[label]['precision']:.4f}, Recall={report[label]['recall']:.4f}, F1={report[label]['f1-score']:.4f}")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)

if f1_macro > 0.95:
    print(f"\n⚠️  Macro-F1 = {f1_macro:.4f} (>95%)")
    print("   Port-only achieves near-perfect classification.")
    print("   Recommendation: REMOVE Destination Port for forgetting experiments.")
elif f1_macro > 0.85:
    print(f"\n⚠️  Macro-F1 = {f1_macro:.4f} (85-95%)")
    print("   Port is strong predictor but not perfect.")
    print("   Recommendation: Consider ablation study.")
else:
    print(f"\n✅ Macro-F1 = {f1_macro:.4f} (<85%)")
    print("   Port leakage exists but limited impact.")

print("\n✅ Port leakage test complete (fast version)")
