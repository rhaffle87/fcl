"""
CIC-IDS2017 / LYCOS-IDS2017 Preprocessing Script
Extract 3-class dataset (Normal, SSH-Patator, DoS) for FL-CL experiments

Input: LYCOS-IDS2017 CSV files (corrected version from Rosay et al. 2022)
Output: preprocessed_3class.csv (trainable for PyTorch)

Author: Rafli Alif
Date: 2026-10-02
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path("E:/Projects/fl-cl/data/raw/LYCOS-IDS2017")
OUTPUT_DIR = Path("E:/Projects/fl-cl/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("LYCOS-IDS2017 Preprocessing for FL-CL-IDS Thesis")
print("="*80)

# Step 1: Load CSV files
print("\n[1/8] Loading CSV files from LYCOS-IDS2017...")
csv_files = list(DATA_DIR.glob("**/*.csv"))
print(f"Found {len(csv_files)} CSV files")

if len(csv_files) == 0:
    print("ERROR: No CSV files found!")
    print("Expected path: E:/Projects/fl-cl/data/raw/LYCOS-IDS2017/")
    print("Please extract LYCOS-IDS2017.zip first")
    exit(1)

dfs = []
for csv_file in csv_files:
    print(f"  Loading {csv_file.name}...")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8', low_memory=False)
        dfs.append(df)
    except Exception as e:
        print(f"  ERROR loading {csv_file.name}: {e}")
        continue

df = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")

# Step 2: Filter 3 classes only
print("\n[2/8] Filtering 3 classes (BENIGN, SSH-Patator, DoS)...")
print(f"Original label distribution:\n{df['Label'].value_counts()}\n")

# Keep only relevant classes
target_labels = ['BENIGN', 'SSH-Patator', 'DoS Slowloris', 'DoS Hulk', 'DoS GoldenEye']
df_filtered = df[df['Label'].isin(target_labels)].copy()

# Merge DoS variants into single 'DoS' class
df_filtered['Label'] = df_filtered['Label'].replace({
    'DoS Slowloris': 'DoS',
    'DoS Hulk': 'DoS',
    'DoS GoldenEye': 'DoS'
})

print(f"After filtering: {len(df_filtered):,} rows")
print(f"Class distribution:\n{df_filtered['Label'].value_counts()}\n")

# Step 3: Handle missing values and inf
print("\n[3/8] Handling missing values and inf...")
# Replace inf with NaN
df_filtered.replace([np.inf, -np.inf], np.nan, inplace=True)

# Check NaN count
nan_counts = df_filtered.isnull().sum()
print(f"Columns with NaN: {(nan_counts > 0).sum()}")

# Drop rows with NaN in critical columns (or impute)
print(f"Rows before NaN removal: {len(df_filtered):,}")
df_filtered = df_filtered.dropna()
print(f"Rows after NaN removal: {len(df_filtered):,}")

# Step 4: Feature selection
print("\n[4/8] Feature selection...")
# Exclude non-numeric and identifier columns
exclude_cols = ['Label', 'Timestamp', 'Flow ID', 'Source IP', 'Destination IP', 
                'Source Port', 'Destination Port', 'Protocol']
feature_cols = [col for col in df_filtered.columns if col not in exclude_cols]

# Keep only numeric columns
numeric_cols = df_filtered[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric features: {len(numeric_cols)}")

# Remove features with zero variance (constant values)
zero_var_cols = []
for col in numeric_cols:
    if df_filtered[col].std() == 0:
        zero_var_cols.append(col)

numeric_cols = [col for col in numeric_cols if col not in zero_var_cols]
print(f"After removing zero-variance: {len(numeric_cols)} features")
if zero_var_cols:
    print(f"Removed: {zero_var_cols[:5]}...")

# Step 5: Encode labels
print("\n[5/8] Encoding labels...")
label_encoder = LabelEncoder()
df_filtered['Label_Encoded'] = label_encoder.fit_transform(df_filtered['Label'])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print(f"Label mapping: {label_mapping}")

# Step 6: Normalize features
print("\n[6/8] Normalizing features (StandardScaler)...")
X = df_filtered[numeric_cols].values
y = df_filtered['Label_Encoded'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Feature matrix shape: {X_scaled.shape}")
print(f"Label vector shape: {y.shape}")

# Step 7: Train/Val/Test split (stratified)
print("\n[7/8] Splitting dataset (60% train, 20% val, 20% test)...")
X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train):,} samples")
print(f"Val:   {len(X_val):,} samples")
print(f"Test:  {len(X_test):,} samples")

# Class distribution in each split
print("\nTrain class distribution:")
for label_name, label_id in label_mapping.items():
    count = (y_train == label_id).sum()
    print(f"  {label_name}: {count:,} ({count/len(y_train)*100:.2f}%)")

# Step 8: Save preprocessed data
print("\n[8/8] Saving preprocessed data...")

# Save as CSV (with feature names)
train_df = pd.DataFrame(X_train, columns=numeric_cols)
train_df['Label'] = y_train
train_df.to_csv(OUTPUT_DIR / "train_3class.csv", index=False)

val_df = pd.DataFrame(X_val, columns=numeric_cols)
val_df['Label'] = y_val
val_df.to_csv(OUTPUT_DIR / "val_3class.csv", index=False)

test_df = pd.DataFrame(X_test, columns=numeric_cols)
test_df['Label'] = y_test
test_df.to_csv(OUTPUT_DIR / "test_3class.csv", index=False)

# Save metadata
metadata = {
    'dataset': 'LYCOS-IDS2017',
    'classes': label_mapping,
    'num_features': len(numeric_cols),
    'feature_names': numeric_cols,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_std': scaler.scale_.tolist(),
    'train_samples': len(X_train),
    'val_samples': len(X_val),
    'test_samples': len(X_test),
    'preprocessing_date': '2026-10-02'
}

import json
with open(OUTPUT_DIR / "metadata.json", 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved to: {OUTPUT_DIR}")
print(f"  - train_3class.csv: {len(train_df):,} rows")
print(f"  - val_3class.csv:   {len(val_df):,} rows")
print(f"  - test_3class.csv:  {len(test_df):,} rows")
print(f"  - metadata.json")

# Summary statistics
print("\n" + "="*80)
print("PREPROCESSING COMPLETE")
print("="*80)
print(f"Original dataset: {len(df):,} rows")
print(f"After filtering: {len(df_filtered):,} rows")
print(f"Final (train+val+test): {len(X_train) + len(X_val) + len(X_test):,} rows")
print(f"Features: {len(numeric_cols)}")
print(f"Classes: 3 (Normal=0, DoS=1, SSH-Patator=2)")
print("\nImbalance ratio (train set):")
for label_name, label_id in sorted(label_mapping.items(), key=lambda x: x[1]):
    count = (y_train == label_id).sum()
    ratio = count / min([sum(y_train == i) for i in label_mapping.values()])
    print(f"  {label_name}: {ratio:.1f}x")

print("\nReady for PyTorch training!")
print("="*80)
