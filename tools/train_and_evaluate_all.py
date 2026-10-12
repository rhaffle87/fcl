# tools/retrain_and_evaluate_all.py
#
# Comprehensive retraining and empirical verification suite for FL-CL.
# Trains 1D-CNN, MLP, and Transformer on the clean testbed dataset.
# Evaluates on both:
#   1) Stratified 80/20 train/test split (measuring out-of-sample multi-class generalization)
#   2) Temporal 80/20 train/test split (measuring chronological production streaming performance)
# Logs epoch loss, convergence, confusion matrices, and per-class Precision/Recall/F1.

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src" / "defender"))
sys.path.insert(0, str(repo_root / "src"))

import client
from model import get_model

LABEL_NAMES = {0: "Normal", 1: "Botnet", 2: "Exfiltration", 3: "BruteForce", 4: "DoS"}


def train_single_model(
    model_type,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=40,
    lr=0.005,
    batch_size=64,
    seed=42,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    num_classes = 5

    model = get_model(
        model_type=model_type, input_dim=input_dim, num_classes=num_classes
    )
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Compute class weights for cross-entropy to handle extreme imbalance
    counts = Counter(y_train.numpy())
    total_samples = len(y_train)
    weights = []
    for c in range(num_classes):
        cnt = counts.get(c, 1)
        w = total_samples / (num_classes * cnt)
        weights.append(w)
    class_weights = torch.tensor(weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "epoch_times": []}

    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        ep_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        t_elapsed = time.time() - t0
        avg_loss = ep_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        history["epoch_times"].append(t_elapsed)

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        logits = model(X_val)
        val_loss = criterion(logits, y_val).item()
        preds = torch.argmax(logits, dim=1).numpy()
        y_true = y_val.numpy()

    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    conf_mat = confusion_matrix(y_true, preds, labels=list(range(5)))

    # Per-class metrics
    per_class = {}
    present_classes = np.unique(y_true)
    for c in range(5):
        c_name = LABEL_NAMES[c]
        if c in present_classes:
            prec = float(precision_score(y_true == c, preds == c, zero_division=0))
            rec = float(recall_score(y_true == c, preds == c, zero_division=0))
            f1 = float(f1_score(y_true == c, preds == c, zero_division=0))
            sup = int((y_true == c).sum())
        else:
            prec, rec, f1, sup = 0.0, 0.0, 0.0, 0
        per_class[c_name] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": sup,
        }

    return {
        "model_type": model_type,
        "epochs": epochs,
        "final_train_loss": history["train_loss"][-1],
        "val_loss": val_loss,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "history": history,
    }


def run_full_retraining():
    print("=" * 80)
    print("  FL-CL EXHAUSTIVE RETRAINING & PERFORMANCE VERIFICATION PIPELINE")
    print("=" * 80)

    # Load data from physical directories
    flows_a = repo_root / "data" / "raw" / "live_flows" / "defender_a"
    flows_b = repo_root / "data" / "raw" / "live_flows" / "defender_b"

    print("[*] Ingesting physical flow CSVs...")
    X_a, y_a = client.load_ramdisk_flows(str(flows_a))
    X_b, y_b = client.load_ramdisk_flows(str(flows_b))

    X_all = torch.cat([X_a, X_b], dim=0)
    y_all = torch.cat([y_a, y_b], dim=0)
    N_total = len(X_all)
    print(
        f"[+] Loaded {N_total} total flows across dual defenders. Dimensions: {X_all.shape}"
    )

    models = ["cnn", "mlp", "transformer"]
    results = {"stratified": {}, "temporal": {}}

    # =========================================================================
    # PART 1: STRATIFIED 80/20 EVALUATION (Out-of-sample Minority Generalization)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  1. STRATIFIED 80/20 TRAIN/VAL EVALUATION")
    print("=" * 60)
    np.random.seed(42)
    train_idx, val_idx = [], []
    y_np = y_all.numpy()
    for lbl in range(5):
        c_idx = np.where(y_np == lbl)[0]
        np.random.shuffle(c_idx)
        sp = int(np.ceil(0.8 * len(c_idx)))
        train_idx.extend(c_idx[:sp])
        val_idx.extend(c_idx[sp:])

    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)

    X_train_s, y_train_s = X_all[train_idx], y_all[train_idx]
    X_val_s, y_val_s = X_all[val_idx], y_all[val_idx]

    print(
        f"[*] Stratified Split: Train N={len(X_train_s)} {dict(Counter(y_train_s.numpy()))}"
    )
    print(
        f"[*] Stratified Split: Val   N={len(X_val_s)} {dict(Counter(y_val_s.numpy()))}"
    )

    for m in models:
        print(f"\n--- Retraining {m.upper()} on Stratified Split ---")
        res = train_single_model(
            m, X_train_s, y_train_s, X_val_s, y_val_s, epochs=40, lr=0.005
        )
        results["stratified"][m] = res
        print(
            f"  [+] {m.upper()} Accuracy: {res['accuracy'] * 100:.2f}% | Macro F1: {res['macro_f1'] * 100:.2f}%"
        )
        for cname, p in res["per_class"].items():
            print(
                f"      {cname:12s} -> Prec: {p['precision'] * 100:6.2f}%, Recall: {p['recall'] * 100:6.2f}%, F1: {p['f1'] * 100:6.2f}% (N={p['support']})"
            )

    # =========================================================================
    # PART 2: TEMPORAL 80/20 EVALUATION (Chronological Streaming Split)
    # =========================================================================
    print("\n" + "=" * 60)
    print("  2. TEMPORAL 80/20 CHRONOLOGICAL SPLIT EVALUATION")
    print("=" * 60)
    # Split per defender node chronologically
    sp_a = int(np.ceil(0.8 * len(X_a)))
    sp_b = int(np.ceil(0.8 * len(X_b)))

    X_train_t = torch.cat([X_a[:sp_a], X_b[:sp_b]], dim=0)
    y_train_t = torch.cat([y_a[:sp_a], y_b[:sp_b]], dim=0)
    X_val_t = torch.cat([X_a[sp_a:], X_b[sp_b:]], dim=0)
    y_val_t = torch.cat([y_a[sp_a:], y_b[sp_b:]], dim=0)

    print(
        f"[*] Temporal Split: Train N={len(X_train_t)} {dict(Counter(y_train_t.numpy()))}"
    )
    print(
        f"[*] Temporal Split: Val   N={len(X_val_t)} {dict(Counter(y_val_t.numpy()))}"
    )

    for m in models:
        print(f"\n--- Retraining {m.upper()} on Temporal Split ---")
        res = train_single_model(
            m, X_train_t, y_train_t, X_val_t, y_val_t, epochs=40, lr=0.005
        )
        results["temporal"][m] = res
        print(
            f"  [+] {m.upper()} Accuracy: {res['accuracy'] * 100:.2f}% | Normal Recall: {res['per_class']['Normal']['recall'] * 100:.2f}%"
        )

    # Save retraining results to disk
    out_file = repo_root / "data" / "reports" / "retraining_verification_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full Retraining Results successfully exported to: {out_file}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive retraining and empirical verification suite for FL-CL."
    )
    parser.parse_args()
    run_full_retraining()
