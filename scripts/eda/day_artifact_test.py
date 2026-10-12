#!/usr/bin/env python3
"""
Day-of-Week Artifact Test (Decision 2)
Tests if Normal traffic has day-specific patterns that would confound Task 1 vs Task 2
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

print("=" * 80)
print("DAY-OF-WEEK ARTIFACT TEST (Decision 2)")
print("=" * 80)

cache_file = "E:/Projects/thesis/Datasets/interim/cic_3class_dedup.parquet"
print(f"\nLoading {cache_file}...")
df_all = pd.read_parquet(cache_file)

# Normal only, Tuesday vs Wednesday
df_normal = df_all[df_all["Label"] == "BENIGN"].copy()
df_tue_wed = df_normal[df_normal["Day"].isin(["Tuesday", "Wednesday"])].copy()

print("\nNormal traffic:")
print(f"  Tuesday: {(df_tue_wed['Day']=='Tuesday').sum():,}")
print(f"  Wednesday: {(df_tue_wed['Day']=='Wednesday').sum():,}")

# Balanced sample
min_count = df_tue_wed["Day"].value_counts().min()
sample_size = min(20000, min_count)  # Cap at 20K for speed

print(f"\nBalanced sample: {sample_size:,} per day")

# Features (no port)
numeric_cols = df_tue_wed.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "Destination Port"]

seeds = [42, 43, 44, 45, 46]
accuracies = []

for seed in seeds:
    df_balanced = pd.concat(
        [
            df_tue_wed[df_tue_wed["Day"] == "Tuesday"].sample(
                n=sample_size, random_state=seed
            ),
            df_tue_wed[df_tue_wed["Day"] == "Wednesday"].sample(
                n=sample_size, random_state=seed
            ),
        ]
    ).reset_index(drop=True)

    X = df_balanced[numeric_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy().astype(float)
    y = (df_balanced["Day"] == "Wednesday").to_numpy().astype(int)  # 0=Tue, 1=Wed

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    dt = DecisionTreeClassifier(max_depth=5, random_state=seed)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"  Seed {seed}: Tue vs Wed accuracy = {acc:.4f}")

print(f"\n{'='*80}")
print("RESULT")
print("=" * 80)

mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)

print(f"\nTue vs Wed Classification (no port): {mean_acc:.4f} ± {std_acc:.4f}")

if mean_acc > 0.75:
    print("\n⚠️  Accuracy >75% - Strong day-of-week pattern detected!")
    print("   Recommendation: Random split (Decision 2 Option A)")
elif mean_acc > 0.60:
    print("\n⚠️  Accuracy 60-75% - Moderate day pattern")
    print("   Recommendation: Random split preferred")
else:
    print("\n✅ Accuracy <60% - Weak day pattern")
    print("   Temporal split (Option B) acceptable")

print("\n✅ Day artifact test complete")
