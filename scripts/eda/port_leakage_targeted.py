#!/usr/bin/env python3
"""
Port leakage test - Targeted sampling per class
Date: 2026-10-02
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split

print("="*80)
print("PORT LEAKAGE TEST - Targeted Class Sampling")
print("="*80)

np.random.seed(42)
datasets_base = "E:/Projects/thesis/Datasets/CIC-IDS2017"

# Load specific files for each class
print("\nLoading Normal from Monday (50K)...")
df_mon = pd.read_csv(f"{datasets_base}/Monday-WorkingHours.pcap_ISCX.csv", nrows=50000, skipinitialspace=True)
df_mon.columns = df_mon.columns.str.strip()

print("Loading SSH from Tuesday (all)...")
df_tue = pd.read_csv(f"{datasets_base}/Tuesday-WorkingHours.pcap_ISCX.csv", skipinitialspace=True)
df_tue.columns = df_tue.columns.str.strip()
df_ssh = df_tue[df_tue['Label'] == 'SSH-Patator'].copy()

print("Loading DoS from Wednesday (50K)...")
df_wed = pd.read_csv(f"{datasets_base}/Wednesday-workingHours.pcap_ISCX.csv", nrows=50000, skipinitialspace=True)
df_wed.columns = df_wed.columns.str.strip()
df_wed['Label'] = df_wed['Label'].replace({
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS',
    'DoS slowloris': 'DoS',
    'DoS Slowhttptest': 'DoS'
})
df_dos = df_wed[df_wed['Label'] == 'DoS'].copy()

# Combine
df_normal = df_mon[df_mon['Label'] == 'BENIGN'].copy()

print(f"\nRaw counts:")
print(f"  Normal: {len(df_normal):,}")
print(f"  SSH: {len(df_ssh):,}")
print(f"  DoS: {len(df_dos):,}")

# Deduplicate each
df_normal = df_normal.drop_duplicates()
df_ssh = df_ssh.drop_duplicates()
df_dos = df_dos.drop_duplicates()

print(f"\nAfter deduplication:")
print(f"  Normal: {len(df_normal):,}")
print(f"  SSH: {len(df_ssh):,}")
print(f"  DoS: {len(df_dos):,}")

# Port distribution
print(f"\n{'='*80}")
print("DESTINATION PORT DISTRIBUTION BY CLASS")
print("="*80)

for label, subset in [('BENIGN', df_normal), ('SSH-Patator', df_ssh), ('DoS', df_dos)]:
    top_ports = subset['Destination Port'].value_counts().head(5)
    print(f"\n{label} (n={len(subset):,}):")
    for port, count in top_ports.items():
        pct = (count / len(subset)) * 100
        print(f"  Port {int(port):5d}: {count:6,} ({pct:5.2f}%)")

# Balanced sample (limit by smallest class)
sample_per_class = min(len(df_normal), len(df_ssh), len(df_dos))

print(f"\n{'='*80}")
print(f"Balanced Sampling: {sample_per_class:,} per class (seed=42)")
print("="*80)

df_balanced = pd.concat([
    df_normal.sample(n=sample_per_class, random_state=42),
    df_ssh.sample(n=sample_per_class, random_state=42),
    df_dos  # Use all DoS (it's the smallest)
])

df_balanced['Label'] = df_balanced['Label'].replace({'BENIGN': 'BENIGN', 'SSH-Patator': 'SSH-Patator', 'DoS': 'DoS'})

# Reset index to avoid pyarrow issues
df_balanced = df_balanced.reset_index(drop=True)

print(f"\nBalanced dataset: {len(df_balanced):,} total")
print(df_balanced['Label'].value_counts())

# Prepare features - convert to numpy to avoid pyarrow indexing issues
X_port = df_balanced['Destination Port'].to_numpy().reshape(-1, 1).astype(float)
y = df_balanced['Label'].to_numpy()

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_port, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# Train Decision Tree
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

print(f"\n**Macro-F1: {f1_macro:.4f}**")

print("\nConfusion Matrix:")
print("                    Predicted")
print("Actual        BENIGN    SSH    DoS")
print(f"BENIGN        {cm[0,0]:6d} {cm[0,1]:6d} {cm[0,2]:6d}")
print(f"SSH-Patator   {cm[1,0]:6d} {cm[1,1]:6d} {cm[1,2]:6d}")
print(f"DoS           {cm[2,0]:6d} {cm[2,1]:6d} {cm[2,2]:6d}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# Calculate per-class metrics manually
print("\nPer-class Analysis:")
for i, label in enumerate(['BENIGN', 'SSH-Patator', 'DoS']):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"  {label:15s}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)

if f1_macro > 0.95:
    print(f"\n⚠️  Macro-F1 = {f1_macro:.4f} (>95%)")
    print("   Destination Port achieves near-perfect classification.")
    print("   SSH→22 and DoS→80 are deterministic mappings.")
    print("   **Recommendation: REMOVE Destination Port for experiments.**")
    print("   Forgetting experiments would measure port memorization, not attack patterns.")
elif f1_macro > 0.85:
    print(f"\n⚠️  Macro-F1 = {f1_macro:.4f} (85-95%)")
    print("   Port is strong predictor but not perfect.")
    print("   **Recommendation: Consider ablation study (E0 with/without port).**")
else:
    print(f"\n✅ Macro-F1 = {f1_macro:.4f} (<85%)")
    print("   Port leakage exists but impact is limited.")
    print("   Consider keeping with caveat in methodology.")

print("\n✅ Port leakage test complete")
