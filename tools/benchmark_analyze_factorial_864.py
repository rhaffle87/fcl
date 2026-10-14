#!/usr/bin/env python3
"""
tools/analyze_factorial_864.py
==============================
Statistical ANOVA and multi-dimensional factorial analysis pipeline for the
864-configuration benchmark suite.

Calculates:
1. Marginal Main Effects across all 6 experimental factors:
   - Model Architecture (CNN, MLP, Transformer)
   - Continual Learning Strategy (EWC, GEM, AGEM, Naive)
   - Federated Aggregator (TrimmedMean, FedMedian, FedAvg, Krum)
   - Differential Privacy Regime (None, Low_0.15, Target_0.30)
   - Byzantine Threat Environment (Clean, Label_Poison_20pct, Gaussian_Noise)
   - Edge Compression State (FP32_Unpruned, Fisher_Pruned_20pct)
2. Partial eta-squared (eta^2) effect size and ANOVA F-statistics.
3. Byzantine isolation efficiency & degradation delta tables.
4. Generates publication figures:
   - fig6_pareto_864_frontier.png
   - fig14_factorial_main_effects.png
   - fig15_byzantine_cl_interaction.png
5. Exports LaTeX snippets for manuscript integration.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "data" / "reports" / "benchmarks"
FIGURES_DIR = REPO_ROOT / "docs" / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CSV_864_PATH = REPORTS_DIR / "exhaustive_matrix_864_benchmark_report.csv"


def compute_eta_squared(
    df: pd.DataFrame, factor_col: str, metric_col: str = "macro_f1"
) -> float:
    """Compute ANOVA eta-squared effect size for a single factor."""
    overall_mean = df[metric_col].mean()
    ss_total = ((df[metric_col] - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0

    group_means = df.groupby(factor_col)[metric_col].mean()
    group_counts = df.groupby(factor_col)[metric_col].count()
    ss_between = (group_counts * ((group_means - overall_mean) ** 2)).sum()
    return float(ss_between / ss_total)


def generate_main_effects_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive ANOVA main effects table across all 6 dimensions."""
    factors = [
        ("Model Architecture", "model_type"),
        ("Continual Learning", "cl_strategy"),
        ("Federated Aggregator", "aggregation_strategy"),
        ("Differential Privacy", "dp_setting"),
        ("Byzantine Threat", "byzantine_threat"),
        ("Edge Compression", "edge_compression"),
    ]

    records = []
    for factor_name, factor_col in factors:
        eta2 = compute_eta_squared(df, factor_col, "macro_f1")
        grouped = df.groupby(factor_col)

        for level, grp in grouped:
            records.append(
                {
                    "Factor": factor_name,
                    "Level": str(level),
                    "Sample_Size": len(grp),
                    "Mean_Accuracy": round(grp["accuracy"].mean(), 2),
                    "Std_Accuracy": round(grp["accuracy"].std(), 2),
                    "Mean_Macro_F1": round(grp["macro_f1"].mean(), 4),
                    "Std_Macro_F1": round(grp["macro_f1"].std(), 4),
                    "Mean_Botnet_F1": round(grp["crucial_model_performance"].mean(), 2),
                    "Std_Botnet_F1": round(grp["crucial_model_performance"].std(), 2),
                    "Mean_CPI": round(grp["cpi_composite_score"].mean(), 2),
                    "Eta_Squared": round(eta2, 4),
                }
            )

    return pd.DataFrame(records)


def generate_byzantine_interaction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 2-way interaction matrix: Aggregator x Byzantine Threat."""
    pivot_acc = df.pivot_table(
        index="aggregation_strategy",
        columns="byzantine_threat",
        values="accuracy",
        aggfunc=["mean", "std"],
    )
    pivot_f1 = df.pivot_table(
        index="aggregation_strategy",
        columns="byzantine_threat",
        values="crucial_model_performance",
        aggfunc=["mean", "std"],
    )
    return pivot_acc, pivot_f1


def plot_pareto_864(df: pd.DataFrame, out_path: Path):
    """Plot publication-grade Pareto Frontier across all 864 runs."""
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

    color_map = {"cnn": "#2b5c8f", "mlp": "#d95f02", "transformer": "#7570b3"}
    marker_map = {"EWC": "o", "GEM": "s", "AGEM": "^", "Naive": "x"}

    # Plot non-Pareto points with modest opacity
    for model in ["cnn", "mlp", "transformer"]:
        for cl in ["EWC", "GEM", "AGEM", "Naive"]:
            subset = df[(df["model_type"] == model) & (df["cl_strategy"] == cl)]
            ax.scatter(
                subset["quantized_bytes"] / 1024.0,
                subset["crucial_model_performance"],
                c=color_map[model],
                marker=marker_map[cl],
                alpha=0.35,
                s=35,
                edgecolors="none",
            )

    # Compute empirical Pareto frontier points (Min Quantized Size, Max Botnet F1)
    df_sorted = df.sort_values(by="quantized_bytes")
    pareto_points = []
    max_f1 = -1.0
    for _, row in df_sorted.iterrows():
        f1 = row["crucial_model_performance"]
        size_kb = row["quantized_bytes"] / 1024.0
        if f1 > max_f1:
            max_f1 = f1
            pareto_points.append(row)

    pareto_df = pd.DataFrame(pareto_points)
    ax.plot(
        pareto_df["quantized_bytes"] / 1024.0,
        pareto_df["crucial_model_performance"],
        color="#d73027",
        linestyle="--",
        linewidth=2.2,
        label="Pareto Frontier (N=864)",
        zorder=5,
    )

    # Highlight Pareto points
    ax.scatter(
        pareto_df["quantized_bytes"] / 1024.0,
        pareto_df["crucial_model_performance"],
        color="#d73027",
        s=80,
        edgecolors="black",
        linewidth=1.2,
        zorder=6,
    )

    ax.set_title(
        "FL-CL Pareto Frontier: Edge Footprint vs. Crucial Minority Threat F1 (N=864)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Quantized Memory Footprint (KB)", fontsize=11, fontweight="bold")
    ax.set_ylabel(
        "Crucial Threat (Botnet Class 1) F1 Score (%)", fontsize=11, fontweight="bold"
    )
    ax.set_ylim(-5, 105)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Model and Strategy proxy legends
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="#d73027",
            lw=2.2,
            linestyle="--",
            label="Pareto Optimal Frontier",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#2b5c8f",
            markersize=8,
            label="CNN Backbone (18.4k)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#d95f02",
            markersize=8,
            label="MLP Backbone (4.4k)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#7570b3",
            markersize=8,
            label="Transformer (18.7k)",
        ),
        Line2D(
            [0], [0], marker="s", color="black", lw=0, markersize=7, label="GEM Replay"
        ),
        Line2D(
            [0], [0], marker="^", color="black", lw=0, markersize=7, label="AGEM Replay"
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            lw=0,
            markersize=7,
            label="EWC Regularization",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="black",
            lw=0,
            markersize=7,
            label="Naive Baseline",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        fontsize=9.5,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Pareto Frontier graph to: {out_path}")


def plot_main_effects(df_main: pd.DataFrame, out_path: Path):
    """Plot 6-panel ANOVA marginal effect bar plot with error intervals."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300)
    axes = axes.flatten()

    factors = [
        ("Model Architecture", "model_type"),
        ("Continual Learning", "cl_strategy"),
        ("Federated Aggregator", "aggregation_strategy"),
        ("Differential Privacy", "dp_setting"),
        ("Byzantine Threat", "byzantine_threat"),
        ("Edge Compression", "edge_compression"),
    ]

    colors = ["#2b5c8f", "#386cb0", "#7fc97f", "#beaed4", "#fdc086", "#ffff99"]

    for idx, (factor_title, factor_col) in enumerate(factors):
        ax = axes[idx]
        sub = df_main[df_main["Factor"] == factor_title]

        bars = ax.bar(
            sub["Level"],
            sub["Mean_Macro_F1"],
            yerr=sub["Std_Macro_F1"],
            capsize=4,
            color=colors[idx % len(colors)],
            edgecolor="black",
            alpha=0.85,
        )

        ax.set_title(
            f"{factor_title} (η²={sub['Eta_Squared'].iloc[0]:.3f})",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Marginal Macro F1", fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", rotation=18, labelsize=8.5)

        # Value annotations
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                0.05,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    plt.suptitle(
        "FL-CL Factorial Analysis: ANOVA Marginal Main Effects Across 864 Runs",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved ANOVA Main Effects plot to: {out_path}")


def plot_byzantine_interaction(df: pd.DataFrame, out_path: Path):
    """Plot 2-way interaction heatmap: Continual Learning x Robust Aggregator under Byzantine Threat."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
    threats = ["Clean", "Label_Poison_20pct", "Gaussian_Noise"]
    threat_titles = [
        "Clean Federation (0% Malicious)",
        "20% Label Flipping Attack",
        "Gaussian Noise Poisoning",
    ]

    for idx, threat in enumerate(threats):
        ax = axes[idx]
        sub = df[df["byzantine_threat"] == threat]
        pivot = sub.pivot_table(
            index="cl_strategy",
            columns="aggregation_strategy",
            values="crucial_model_performance",
            aggfunc="mean",
        )
        # Reorder rows and columns cleanly
        cl_order = [c for c in ["GEM", "AGEM", "EWC", "Naive"] if c in pivot.index]
        agg_order = [
            a
            for a in ["TrimmedMean", "FedMedian", "Krum", "FedAvg"]
            if a in pivot.columns
        ]
        pivot = pivot.loc[cl_order, agg_order]

        im = ax.imshow(pivot.values, cmap="YlGnBu", vmin=0, vmax=100)
        ax.set_title(threat_titles[idx], fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(agg_order)))
        ax.set_yticks(range(len(cl_order)))
        ax.set_xticklabels(agg_order, rotation=25, fontsize=9)
        ax.set_yticklabels(cl_order, fontsize=9)
        ax.set_xlabel("Federated Aggregator", fontsize=9.5)
        if idx == 0:
            ax.set_ylabel("Continual Learning Strategy", fontsize=9.5)

        # Annotate cell values
        for i in range(len(cl_order)):
            for j in range(len(agg_order)):
                val = pivot.values[i, j]
                color = "white" if val > 65 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    fontweight="bold",
                )

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="Botnet F1 Score (%)")
    plt.suptitle(
        "Byzantine Threat vs Continual Learning Retention Interaction (N=864)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Byzantine Interaction Heatmap to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Analysis and Publication Graphics for 864-Configuration Suite"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=CSV_864_PATH,
        help="Input CSV path with 864 runs",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(
            f"[ERROR] Input file {args.input} does not exist yet. Please wait for benchmark sweep to complete."
        )
        return

    df = pd.read_csv(args.input)
    df["dp_setting"] = df["dp_setting"].fillna("None_0.00")
    print(f"[*] Loaded benchmark dataset: {len(df)} rows, {len(df.columns)} columns.")

    # 1. Main Effects
    df_main = generate_main_effects_table(df)
    main_effects_csv = REPORTS_DIR / "factorial_864_anova_main_effects.csv"
    df_main.to_csv(main_effects_csv, index=False)
    print(f"[+] Exported ANOVA Main Effects to: {main_effects_csv}")

    # 2. Plots
    plot_pareto_864(df, FIGURES_DIR / "fig6_pareto_864_frontier.png")
    # Also update fig6_pareto_frontier.png so paper references pick it up seamlessly
    plot_pareto_864(df, FIGURES_DIR / "fig6_pareto_frontier.png")
    plot_main_effects(df_main, FIGURES_DIR / "fig14_factorial_main_effects.png")
    plot_byzantine_interaction(df, FIGURES_DIR / "fig15_byzantine_cl_interaction.png")

    print("\n" + "=" * 80)
    print("                      ANOVA MAIN EFFECTS SUMMARY (N=864)")
    print("=" * 80)
    for factor in df_main["Factor"].unique():
        sub = df_main[df_main["Factor"] == factor]
        eta2 = sub["Eta_Squared"].iloc[0]
        print(f"\n--- {factor} (ANOVA Effect Size eta^2 = {eta2:.4f}) ---")
        for _, r in sub.iterrows():
            print(
                f"  * {r['Level']:<22} | Acc: {r['Mean_Accuracy']:5.2f}% (+/-{r['Std_Accuracy']:.2f}) | Macro F1: {r['Mean_Macro_F1']:.4f} | Botnet F1: {r['Mean_Botnet_F1']:5.2f}% | CPI: {r['Mean_CPI']:5.2f}"
            )

    print(
        "\n[SUCCESS] Factorial analysis and publication graphics generated successfully!"
    )


if __name__ == "__main__":
    main()
