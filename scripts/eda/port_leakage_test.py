#!/usr/bin/env python3
"""
Port leakage test - Decision Tree classification using only Destination Port
Date: 2026-10-02
Command: E:/Projects/fl-cl/.venv/Scripts/python.exe scripts/eda/port_leakage_test.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("=" * 80)
print("PORT LEAKAGE TEST - Destination Port as Predictor")
print("Date: 2026-10-02 15:40 UTC+7")
print("=" * 80)

datasets_base = "E:/Projects/thesis/Datasets/CIC-IDS2017"
np.random.seed(42)

# Load all 3 files
files = [
    ("Monday-WorkingHours.pcap_ISCX.csv", "Monday"),
    ("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday"),
    ("Wednesday-workingHours.pcap_ISCX.csv", "Wednesday"),
]

all_data = []

for filename, day in files:
    filepath = f"{datasets_base}/{filename}"
    print(f"\nLoading {day}...")

    df = pd.read_csv(filepath, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    all_data.append(df)

# Combine all data
df_all = pd.concat(all_data, ignore_index=True)
print(f"\nTotal after dedup: {len(df_all):,} rows")

# Filter to 3 classes
df_filtered = df_all[
    df_all["Label"].isin(
        [
            "BENIGN",
            "SSH-Patator",
            "DoS Hulk",
            "DoS GoldenEye",
            "DoS slowloris",
            "DoS Slowhttptest",
        ]
    )
]

# Merge DoS variants
df_filtered["Label"] = df_filtered["Label"].replace(
    {
        "DoS Hulk": "DoS",
        "DoS GoldenEye": "DoS",
        "DoS slowloris": "DoS",
        "DoS Slowhttptest": "DoS",
    }
)

print("\n📋 Class Distribution (3 classes, all Normal):")
print(df_filtered["Label"].value_counts())

# Port distribution per class
print("\n🚨 Destination Port Distribution by Class:")
for label in ["BENIGN", "SSH-Patator", "DoS"]:
    subset = df_filtered[df_filtered["Label"] == label]
    top_ports = subset["Destination Port"].value_counts().head(5)
    print(f"\n{label} (n={len(subset):,}):")
    for port, count in top_ports.items():
        pct = (count / len(subset)) * 100
        print(f"  Port {port}: {count:8,} ({pct:5.2f}%)")

# Balanced sample for fair test
min_class = df_filtered["Label"].value_counts().min()
sample_size = min(5000, min_class)  # Cap at 5K per class

print(f"\n🎯 Sampling {sample_size:,} per class (balanced, seed=42)")

df_balanced = pd.concat(
    [
        df_filtered[df_filtered["Label"] == label].sample(
            n=sample_size, random_state=42
        )
        for label in ["BENIGN", "SSH-Patator", "DoS"]
    ]
)

print(f"Balanced sample: {len(df_balanced):,} total ({sample_size:,} × 3 classes)")

# Prepare features and labels
X_port_only = df_balanced[["Destination Port"]].values
X_all_except_port = df_balanced.drop(
    columns=["Label", "Destination Port"]
).select_dtypes(include=[np.number])

# Handle NaN/Inf in all features
X_all_except_port = X_all_except_port.replace([np.inf, -np.inf], np.nan)
X_all_except_port = X_all_except_port.fillna(0)

y = df_balanced["Label"].values

# Split
X_port_train, X_port_test, y_train, y_test = train_test_split(
    X_port_only, y, test_size=0.3, random_state=42, stratify=y
)

X_all_train, X_all_test, _, _ = train_test_split(
    X_all_except_port, y, test_size=0.3, random_state=42, stratify=y
)

# Test 1: Decision Tree with ONLY Destination Port
print(f"\n{'='*80}")
print("TEST 1: Decision Tree (depth=3) with ONLY Destination Port")
print("=" * 80)

dt_port = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_port.fit(X_port_train, y_train)
y_pred_port = dt_port.predict(X_port_test)

print("\nClassification Report (Port Only):")
print(classification_report(y_test, y_pred_port, digits=4))

cm_port = confusion_matrix(y_test, y_pred_port, labels=["BENIGN", "SSH-Patator", "DoS"])
print("\nConfusion Matrix (Port Only):")
print("                Predicted")
print("           BENIGN  SSH  DoS")
print(f"BENIGN     {cm_port[0,0]:6d} {cm_port[0,1]:4d} {cm_port[0,2]:4d}")
print(f"SSH        {cm_port[1,0]:6d} {cm_port[1,1]:4d} {cm_port[1,2]:4d}")
print(f"DoS        {cm_port[2,0]:6d} {cm_port[2,1]:4d} {cm_port[2,2]:4d}")

# Test 2: Decision Tree WITHOUT Destination Port
print(f"\n{'='*80}")
print(
    f"TEST 2: Decision Tree (depth=5) WITHOUT Destination Port ({X_all_except_port.shape[1]} features)"
)
print("=" * 80)

dt_no_port = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_no_port.fit(X_all_train, y_train)
y_pred_no_port = dt_no_port.predict(X_all_test)

print("\nClassification Report (No Port):")
print(classification_report(y_test, y_pred_no_port, digits=4))

cm_no_port = confusion_matrix(
    y_test, y_pred_no_port, labels=["BENIGN", "SSH-Patator", "DoS"]
)
print("\nConfusion Matrix (No Port):")
print("                Predicted")
print("           BENIGN  SSH  DoS")
print(f"BENIGN     {cm_no_port[0,0]:6d} {cm_no_port[0,1]:4d} {cm_no_port[0,2]:4d}")
print(f"SSH        {cm_no_port[1,0]:6d} {cm_no_port[1,1]:4d} {cm_no_port[1,2]:4d}")
print(f"DoS        {cm_no_port[2,0]:6d} {cm_no_port[2,1]:4d} {cm_no_port[2,2]:4d}")

print(f"\n{'='*80}")
print("CONCLUSION")
print("=" * 80)

# Extract macro-F1
from sklearn.metrics import f1_score

f1_port = f1_score(y_test, y_pred_port, average="macro")
f1_no_port = f1_score(y_test, y_pred_no_port, average="macro")

print(f"\nMacro-F1 (Port Only):    {f1_port:.4f}")
print(f"Macro-F1 (No Port):      {f1_no_port:.4f}")
print(f"Difference:              {f1_port - f1_no_port:+.4f}")

if f1_port > 0.95:
    print("\n⚠️  Port-only classifier achieves >95% macro-F1")
    print("   Keeping Destination Port makes classification trivial.")
    print("   Recommendation: REMOVE for forgetting experiments.")
else:
    print("\n✅ Port-only classifier <95% macro-F1")
    print("   Port leakage exists but not perfect predictor.")

print("\n✅ Port leakage test complete")
