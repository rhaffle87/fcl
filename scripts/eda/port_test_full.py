#!/usr/bin/env python3
"""
Port Leakage Test - FULL DATA, Multiple Seeds, All Conditions
Date: 2026-10-02
Fixes: Previous test used biased 50K sample. This uses FULL deduplicated data.
"""

import os

import pandas as pd

print("=" * 80)
print("PORT LEAKAGE TEST - FULL DATA (Corrected)")
print("Date: 2026-10-02 17:00 UTC+7")
print("=" * 80)

datasets_base = "E:/Projects/thesis/Datasets/CIC-IDS2017"
interim_path = "E:/Projects/thesis/Datasets/interim"
cache_file = f"{interim_path}/cic_3class_dedup.parquet"

# Step 1: Load and deduplicate FULL data (cache for reuse)
if os.path.exists(cache_file):
    print(f"\nLoading cached dedup data from {cache_file}...")
    df_all = pd.read_parquet(cache_file)
    print(f"Loaded: {len(df_all):,} rows")
else:
    print("\nLoading FULL CSVs with efficient dtypes...")

    # Efficient dtypes
    dtype_spec = {"Destination Port": "uint16", "Flow Duration": "int64"}

    dfs = []
    for fname, day in [
        ("Monday-WorkingHours.pcap_ISCX.csv", "Monday"),
        ("Tuesday-WorkingHours.pcap_ISCX.csv", "Tuesday"),
        ("Wednesday-workingHours.pcap_ISCX.csv", "Wednesday"),
    ]:
        print(f"  Loading {day}...")
        df = pd.read_csv(
            f"{datasets_base}/{fname}",
            skipinitialspace=True,
            dtype=dtype_spec,
            low_memory=False,
        )
        df.columns = df.columns.str.strip()
        df["Day"] = day

        # Count before dedup
        print(f"    Before dedup: {len(df):,} rows")
        for label in df["Label"].unique():
            count = (df["Label"] == label).sum()
            print(f"      {label}: {count:,}")

        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total before dedup: {len(df_all):,}")

    # Deduplicate on ALL columns
    print("  Deduplicating on ALL columns...")
    df_all = df_all.drop_duplicates()
    print(
        f"  After dedup: {len(df_all):,} (dropped {len(pd.concat(dfs)) - len(df_all):,})"
    )

    # Filter to 3 classes
    print("\n  Filtering to 3 classes...")
    df_all["Label_Original"] = df_all["Label"]
    df_all["Label"] = df_all["Label"].replace(
        {
            "DoS Hulk": "DoS",
            "DoS GoldenEye": "DoS",
            "DoS slowloris": "DoS",
            "DoS Slowhttptest": "DoS",
        }
    )

    # Exclude FTP, Heartbleed
    df_all = df_all[df_all["Label"].isin(["BENIGN", "SSH-Patator", "DoS"])].copy()

    # Save cache
    print(f"\n  Saving cache to {cache_file}...")
    os.makedirs(interim_path, exist_ok=True)
    df_all.to_parquet(cache_file, index=False)
    print("  ✅ Cache saved")

print(f"\n{'='*80}")
print("AFTER DEDUPLICATION - Per Day Per Class")
print("=" * 80)

for day in ["Monday", "Tuesday", "Wednesday"]:
    df_day = df_all[df_all["Day"] == day]
    print(f"\n{day}:")
    for label in ["BENIGN", "SSH-Patator", "DoS"]:
        count = (df_day["Label"] == label).sum()
        if count > 0:
            print(f"  {label:15s}: {count:,}")

# Step 2: Destination Port distribution
print(f"\n{'='*80}")
print("DESTINATION PORT DISTRIBUTION")
print("=" * 80)

print("\n📋 Normal (per day):")
for day in ["Monday", "Tuesday", "Wednesday"]:
    df_norm_day = df_all[(df_all["Label"] == "BENIGN") & (df_all["Day"] == day)]
    print(f"\n{day} BENIGN (n={len(df_norm_day):,}):")
    top_ports = df_norm_day["Destination Port"].value_counts().head(5)
    for port, count in top_ports.items():
        pct = (count / len(df_norm_day)) * 100
        print(f"  Port {int(port):5d}: {count:8,} ({pct:5.2f}%)")

    # Port 80 and 22 specifically
    port80 = (df_norm_day["Destination Port"] == 80).sum()
    port22 = (df_norm_day["Destination Port"] == 22).sum()
    print(f"  Port 80: {port80:,} ({port80/len(df_norm_day)*100:.2f}%)")
    print(f"  Port 22: {port22:,} ({port22/len(df_norm_day)*100:.2f}%)")

print("\n📋 SSH-Patator:")
df_ssh = df_all[df_all["Label"] == "SSH-Patator"]
print(f"Total (n={len(df_ssh):,}):")
top_ports = df_ssh["Destination Port"].value_counts().head(3)
for port, count in top_ports.items():
    pct = (count / len(df_ssh)) * 100
    print(f"  Port {int(port):5d}: {count:8,} ({pct:5.2f}%)")

print("\n📋 DoS (per variant):")
for variant in ["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"]:
    df_var = df_all[df_all["Label_Original"] == variant]
    if len(df_var) > 0:
        print(f"\n{variant} (n={len(df_var):,}):")
        top_ports = df_var["Destination Port"].value_counts().head(3)
        for port, count in top_ports.items():
            pct = (count / len(df_var)) * 100
            print(f"  Port {int(port):5d}: {count:8,} ({pct:5.2f}%)")

print("\n✅ Step 1-2 complete. Proceeding to classification tests...")
print(f"\nDataset size: {len(df_all):,} rows")
print(f"  BENIGN: {(df_all['Label']=='BENIGN').sum():,}")
print(f"  SSH-Patator: {(df_all['Label']=='SSH-Patator').sum():,}")
print(f"  DoS: {(df_all['Label']=='DoS').sum():,}")
