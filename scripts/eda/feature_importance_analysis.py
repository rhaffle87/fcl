#!/usr/bin/env python3
"""
Feature Importance Analysis - No-Port Decision Tree
Identifies dominant features that may be shortcuts
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

print("="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

cache_file = "E:/Projects/thesis/Datasets/interim/cic_3class_dedup.parquet"
print(f"\nLoading {cache_file}...")
df_all = pd.read_parquet(cache_file)

sample_size = 3219  # SSH limit
seeds = [42, 43, 44, 45, 46]

# Prepare features
numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != 'Destination Port']

print(f"\nTotal features (no port): {len(numeric_cols)}")

# Aggregate importance across seeds
importance_sum = np.zeros(len(numeric_cols))
importance_counts = 0

for seed in seeds:
    df_balanced = pd.concat([
        df_all[df_all['Label'] == 'BENIGN'].sample(n=sample_size, random_state=seed),
        df_all[df_all['Label'] == 'SSH-Patator'],
        df_all[df_all['Label'] == 'DoS'].sample(n=sample_size, random_state=seed)
    ]).reset_index(drop=True)
    
    X = df_balanced[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy().astype(float)
    y = df_balanced['Label'].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    dt = DecisionTreeClassifier(max_depth=8, random_state=seed)
    dt.fit(X_train, y_train)
    
    importance_sum += dt.feature_importances_
    importance_counts += 1

# Average importance
importance_avg = importance_sum / importance_counts

# Top 10 features
top_indices = np.argsort(importance_avg)[::-1][:10]
top_features = [(numeric_cols[i], importance_avg[i]) for i in top_indices]

print(f"\n{'='*80}")
print("TOP 10 FEATURES (Average Importance)")
print("="*80)

for rank, (feat, imp) in enumerate(top_features, 1):
    print(f"{rank:2d}. {feat:40s}: {imp:.4f}")

# Test single feature dominance (top 3)
print(f"\n{'='*80}")
print("SINGLE FEATURE CLASSIFICATION TEST")
print("="*80)

for rank, (feat, imp) in enumerate(top_features[:3], 1):
    f1_scores = []
    feat_idx = numeric_cols.index(feat)
    
    for seed in seeds:
        df_balanced = pd.concat([
            df_all[df_all['Label'] == 'BENIGN'].sample(n=sample_size, random_state=seed),
            df_all[df_all['Label'] == 'SSH-Patator'],
            df_all[df_all['Label'] == 'DoS'].sample(n=sample_size, random_state=seed)
        ]).reset_index(drop=True)
        
        X_all = df_balanced[numeric_cols].copy()
        X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        X_single = X_all.iloc[:, feat_idx].to_numpy().reshape(-1, 1).astype(float)
        y = df_balanced['Label'].to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_single, y, test_size=0.3, random_state=seed, stratify=y
        )
        
        dt = DecisionTreeClassifier(max_depth=3, random_state=seed)
        dt.fit(X_train, y_train)
        y_pred = dt.predict(X_test)
        
        f1 = f1_score(y_test, y_pred, average='macro')
        f1_scores.append(f1)
    
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\n{rank}. {feat}")
    print(f"   Macro-F1 (single feature): {mean_f1:.4f} ± {std_f1:.4f}")
    
    if mean_f1 > 0.90:
        print(f"   ⚠️  HIGH - Potential shortcut feature!")
    elif mean_f1 > 0.75:
        print(f"   ⚠️  MODERATE - Strong predictor")
    else:
        print(f"   ✅ LOW - Weak individual predictor")

print(f"\n{'='*80}")
print("CONCLUSION")
print("="*80)

top1_name, top1_imp = top_features[0]
print(f"\nMost important feature: {top1_name} ({top1_imp:.4f})")
print("\nIf single-feature F1 > 90%, that feature is a shortcut comparable to Destination Port.")
print("High separability (98%) may reflect attack-tool artifacts, not intrusion patterns.")

print("\n✅ Feature importance analysis complete")
