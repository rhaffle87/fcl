#!/usr/bin/env python3
"""
Port Leakage Test - Classification with 3 conditions, 5 seeds
Part 2: Classification tests
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

print("="*80)
print("CLASSIFICATION TESTS - 3 Conditions, 5 Seeds")
print("="*80)

# Load cached data
cache_file = "E:/Projects/thesis/Datasets/interim/cic_3class_dedup.parquet"
print(f"\nLoading {cache_file}...")
df_all = pd.read_parquet(cache_file)
print(f"Loaded: {len(df_all):,} rows")

# Balanced sampling (limited by SSH count)
sample_size = (df_all['Label'] == 'SSH-Patator').sum()
print(f"\nBalanced sample size: {sample_size:,} per class (limited by SSH)")

seeds = [42, 43, 44, 45, 46]

# Prepare features
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'Destination Port']

# Results storage
results = {
    'port_only': {'f1_macro': [], 'f1_benign': [], 'f1_ssh': [], 'f1_dos': [], 'cm': []},
    'no_port': {'f1_macro': [], 'f1_benign': [], 'f1_ssh': [], 'f1_dos': [], 'cm': []},
    'all_features': {'f1_macro': [], 'f1_benign': [], 'f1_ssh': [], 'f1_dos': [], 'cm': []}
}

for seed in seeds:
    print(f"\n{'='*80}")
    print(f"Seed {seed}")
    print("="*80)
    
    # Balanced sample
    df_balanced = pd.concat([
        df_all[df_all['Label'] == 'BENIGN'].sample(n=sample_size, random_state=seed),
        df_all[df_all['Label'] == 'SSH-Patator'],
        df_all[df_all['Label'] == 'DoS'].sample(n=sample_size, random_state=seed)
    ]).reset_index(drop=True)
    
    y = df_balanced['Label'].to_numpy()
    
    # Split
    indices = np.arange(len(df_balanced))
    idx_train, idx_test = train_test_split(indices, test_size=0.3, random_state=seed, stratify=y)
    
    y_train, y_test = y[idx_train], y[idx_test]
    
    # Condition 1: Port only
    X_port = df_balanced['Destination Port'].to_numpy().reshape(-1, 1).astype(float)
    X_port_train, X_port_test = X_port[idx_train], X_port[idx_test]
    
    dt_port = DecisionTreeClassifier(max_depth=3, random_state=seed)
    dt_port.fit(X_port_train, y_train)
    y_pred_port = dt_port.predict(X_port_test)
    
    f1_macro = f1_score(y_test, y_pred_port, average='macro')
    f1_per_class = f1_score(y_test, y_pred_port, average=None, labels=['BENIGN', 'SSH-Patator', 'DoS'])
    cm = confusion_matrix(y_test, y_pred_port, labels=['BENIGN', 'SSH-Patator', 'DoS'])
    
    results['port_only']['f1_macro'].append(f1_macro)
    results['port_only']['f1_benign'].append(f1_per_class[0])
    results['port_only']['f1_ssh'].append(f1_per_class[1])
    results['port_only']['f1_dos'].append(f1_per_class[2])
    results['port_only']['cm'].append(cm)
    
    print(f"  Port only: macro-F1={f1_macro:.4f}")
    
    # Condition 2: No port (depth=8)
    X_no_port = df_balanced[numeric_cols].copy()
    X_no_port = X_no_port.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_no_port = X_no_port.to_numpy().astype(float)
    
    X_no_port_train, X_no_port_test = X_no_port[idx_train], X_no_port[idx_test]
    
    dt_no_port = DecisionTreeClassifier(max_depth=8, random_state=seed)
    dt_no_port.fit(X_no_port_train, y_train)
    y_pred_no_port = dt_no_port.predict(X_no_port_test)
    
    f1_macro = f1_score(y_test, y_pred_no_port, average='macro')
    f1_per_class = f1_score(y_test, y_pred_no_port, average=None, labels=['BENIGN', 'SSH-Patator', 'DoS'])
    cm = confusion_matrix(y_test, y_pred_no_port, labels=['BENIGN', 'SSH-Patator', 'DoS'])
    
    results['no_port']['f1_macro'].append(f1_macro)
    results['no_port']['f1_benign'].append(f1_per_class[0])
    results['no_port']['f1_ssh'].append(f1_per_class[1])
    results['no_port']['f1_dos'].append(f1_per_class[2])
    results['no_port']['cm'].append(cm)
    
    print(f"  No port:   macro-F1={f1_macro:.4f}")
    
    # Condition 3: All features (depth=8)
    X_all = df_balanced[numeric_cols + ['Destination Port']].copy()
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_all = X_all.to_numpy().astype(float)
    
    X_all_train, X_all_test = X_all[idx_train], X_all[idx_test]
    
    dt_all = DecisionTreeClassifier(max_depth=8, random_state=seed)
    dt_all.fit(X_all_train, y_train)
    y_pred_all = dt_all.predict(X_all_test)
    
    f1_macro = f1_score(y_test, y_pred_all, average='macro')
    f1_per_class = f1_score(y_test, y_pred_all, average=None, labels=['BENIGN', 'SSH-Patator', 'DoS'])
    cm = confusion_matrix(y_test, y_pred_all, labels=['BENIGN', 'SSH-Patator', 'DoS'])
    
    results['all_features']['f1_macro'].append(f1_macro)
    results['all_features']['f1_benign'].append(f1_per_class[0])
    results['all_features']['f1_ssh'].append(f1_per_class[1])
    results['all_features']['f1_dos'].append(f1_per_class[2])
    results['all_features']['cm'].append(cm)
    
    print(f"  All feats: macro-F1={f1_macro:.4f}")

# Summary
print(f"\n{'='*80}")
print("SUMMARY - Mean ± Std over 5 seeds")
print("="*80)

for cond_name, cond_data in results.items():
    print(f"\n{cond_name.upper().replace('_', ' ')}:")
    print(f"  Macro-F1:     {np.mean(cond_data['f1_macro']):.4f} ± {np.std(cond_data['f1_macro']):.4f}")
    print(f"  BENIGN F1:    {np.mean(cond_data['f1_benign']):.4f} ± {np.std(cond_data['f1_benign']):.4f}")
    print(f"  SSH F1:       {np.mean(cond_data['f1_ssh']):.4f} ± {np.std(cond_data['f1_ssh']):.4f}")
    print(f"  DoS F1:       {np.mean(cond_data['f1_dos']):.4f} ± {np.std(cond_data['f1_dos']):.4f}")

# Average confusion matrices
print(f"\n{'='*80}")
print("AVERAGE CONFUSION MATRICES")
print("="*80)

for cond_name, cond_data in results.items():
    cm_avg = np.mean(cond_data['cm'], axis=0).astype(int)
    print(f"\n{cond_name.upper().replace('_', ' ')}:")
    print("                Predicted")
    print("Actual     BENIGN    SSH    DoS")
    print(f"BENIGN     {cm_avg[0,0]:6d} {cm_avg[0,1]:6d} {cm_avg[0,2]:6d}")
    print(f"SSH        {cm_avg[1,0]:6d} {cm_avg[1,1]:6d} {cm_avg[1,2]:6d}")
    print(f"DoS        {cm_avg[2,0]:6d} {cm_avg[2,1]:6d} {cm_avg[2,2]:6d}")

print(f"\n✅ Classification tests complete")
