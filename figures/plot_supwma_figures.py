"""
Generate SUP-WMA-related figures for the paper.

Produces:
  - supwma_foundational_gradient.png — bar plot of foundational rate per SUP-WMA
    cluster, sorted, mirroring the bundle-level specificity gradient figure idea.
  - supwma_vs_tractseg_coverage.png — grouped bar plot showing per-subject TractSeg /
    SUP-WMA / either / neither rates among foundational streamlines (Table 4 visualized).
  - triple_no_length_distribution.png — overlaid length histograms for the four
    diagnostic cells in tab:triple_no (atlas-orphan-foundational, atlas-orphan-redundant,
    SW-classified-redundant, TS-classified-redundant), making the
    short-range-vs-long-range contrast visible.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("TRACTO_ROOT", "/home/maia-user/tracto_work")
SUPWMA_DIR = f"{ROOT}/Data/SupWMA"
ANALYSIS = f"{SUPWMA_DIR}/analysis"
OUT = f"{ROOT}/Streamlines-redundancy/figures"
os.makedirs(OUT, exist_ok=True)


def plot_foundational_gradient():
    """Cross-subject mean foundational rate per SUP-WMA cluster, sorted."""
    df = pd.read_csv(f"{ANALYSIS}/per_cluster_averaged.csv")
    df = df.sort_values("foundational", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(df))
    ax.bar(x, df["foundational"], width=1.0, color="#3b6db5", edgecolor="none")
    # Highlight top-3 with red error bars (1 SD)
    top3 = df.head(3)
    ax.errorbar(top3.index, top3["foundational"], yerr=top3["foundational_sd"],
                fmt="none", ecolor="#b53b3b", capsize=3, linewidth=1.2)
    # Annotate the top cluster
    ax.annotate(f"{df.iloc[0]['supwma_cluster']}\n{df.iloc[0]['foundational']:.1f}%",
                xy=(0, df.iloc[0]['foundational']),
                xytext=(8, df.iloc[0]['foundational'] - 1.5),
                fontsize=9, color="#222",
                arrowprops=dict(arrowstyle="->", color="#666", lw=0.6))

    # Reference line: fornix left foundational rate
    ax.axhline(4.9, color="#888", linestyle="--", linewidth=0.8)
    ax.text(len(df) * 0.6, 5.3, "Left fornix (TractSeg, 4.9%)",
            fontsize=9, color="#555")

    ax.set_xlabel("SUP-WMA cluster (sorted by foundational rate)")
    ax.set_ylabel("Foundational rate (%)")
    ax.set_xlim(-1, len(df))
    ax.set_ylim(0, max(df["foundational"]) * 1.15)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_title("SUP-WMA cluster specificity gradient (mean across 5 subjects)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(f"{OUT}/supwma_foundational_gradient.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/supwma_foundational_gradient.png")


def plot_atlas_coverage_among_foundational():
    """Grouped bars: per-subject TS / SW / Either / Neither among foundational."""
    df = pd.read_csv(f"{ANALYSIS}/coverage_foundational.csv")
    subjects = df["subject"].astype(str).values

    cats = [
        ("pct_foundational_tractseg",        "TractSeg",        "#7a4ec5"),
        ("pct_foundational_supwma",          "SUP-WMA",         "#3b6db5"),
        ("pct_foundational_either_atlas",    "Either atlas",    "#3ba07a"),
        ("pct_foundational_doubly_unclassified", "Neither atlas", "#b53b3b"),
    ]
    x = np.arange(len(subjects))
    w = 0.20

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (col, lab, c) in enumerate(cats):
        ax.bar(x + (i - 1.5) * w, df[col], width=w, color=c, label=lab, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.set_xlabel("Subject")
    ax.set_ylabel("% of foundational streamlines")
    ax.set_ylim(0, 100)
    ax.set_title("Atlas coverage of foundational streamlines, per subject")
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    fig.savefig(f"{OUT}/supwma_vs_tractseg_coverage.png", dpi=200, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}/supwma_vs_tractseg_coverage.png")


def plot_triple_no_length_distribution():
    """Length distribution comparison of the four diagnostic cells.
    We don't have raw per-streamline lengths cached, so build from per-subject
    summary mean and percentile columns to draw stylized box-style markers.
    """
    df = pd.read_csv(f"{ANALYSIS}/triple_no_breakdown.csv")
    # Pick the four key cells
    key_cells = [
        ("ts_no_sw_no",  "foundational", "Unclassified, foundational"),
        ("ts_no_sw_no",  "neither",      "Unclassified, redundant"),
        ("ts_no_sw_yes", "neither",      "SUP-WMA only, redundant"),
        ("ts_yes_sw_no", "neither",      "TractSeg only, redundant"),
    ]
    colors = ["#3ba07a", "#b53b3b", "#3b6db5", "#7a4ec5"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, ((atlas, fil, lab), c) in enumerate(zip(key_cells, colors)):
        sub = df[(df["atlas"] == atlas) & (df["filter"] == fil)]
        if len(sub) == 0:
            continue
        means = sub["mean_length"].values
        meds = sub["median_length"].values
        p25 = sub["p25_length"].values
        p75 = sub["p75_length"].values
        p90 = sub["p90_length"].values
        # Plot one row per subject, then overall mean as a thick line
        y = i + np.linspace(-0.18, 0.18, len(means))
        for j in range(len(means)):
            ax.plot([p25[j], p75[j]], [y[j], y[j]], color=c, linewidth=2, alpha=0.55)
            ax.plot([p25[j], p90[j]], [y[j], y[j]], color=c, linewidth=0.8, alpha=0.35)
            ax.plot(meds[j], y[j], "o", color=c, markersize=5)
        # Annotate mean of means
        m = means.mean()
        ax.plot([m, m], [i - 0.3, i + 0.3], color="black", linewidth=1.5)
        ax.text(m + 4, i + 0.32, f"mean = {m:.0f} mm",
                fontsize=9, va="bottom", color="#222")

    ax.set_yticks(range(len(key_cells)))
    ax.set_yticklabels([k[2] for k in key_cells])
    ax.invert_yaxis()
    ax.set_xlabel("Streamline length (mm)")
    ax.set_xlim(0, 220)
    ax.set_title("Length distribution by atlas × filter cell (medians + IQR + p90 per subject)")
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(f"{OUT}/triple_no_length_distribution.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {OUT}/triple_no_length_distribution.png")


if __name__ == "__main__":
    plot_foundational_gradient()
    plot_atlas_coverage_among_foundational()
    plot_triple_no_length_distribution()
