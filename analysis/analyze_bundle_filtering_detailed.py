"""
Analyze per-bundle acceptance rates broken down by:
- Foundational: both COMMIT >= 1 AND SIFT >= 1
- COMMIT-only accepted: COMMIT >= 1 AND SIFT < 1
- SIFT-only accepted: SIFT >= 1 AND COMMIT < 1
- Either accepted: COMMIT >= 1 OR SIFT >= 1
- Neither accepted: both < 1
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from collections import defaultdict

INTERSECTION_DIR = "Data/Intersection_Labels/all_subjects"
BUNDLE_CLASS_PATHS = {
    "665254": "Data/Tractogram_Data/TEST/665254_bundle_classifications.csv",
    "734045": "Data/Tractogram_Data/TRAIN/734045_bundle_classifications.csv",
    "784565": "Data/Tractogram_Data/VAL/784565_bundle_classifications.csv",
    "837964": "Data/Tractogram_Data/TEST/837964_bundle_classifications.csv",
    "877168": "Data/Tractogram_Data/TEST/877168_bundle_classifications.csv",
}

subjects = list(BUNDLE_CLASS_PATHS.keys())

# Collect per-bundle rates
bundle_rates = defaultdict(lambda: defaultdict(list))
all_rows = []

for subj in subjects:
    il = pd.read_csv(os.path.join(INTERSECTION_DIR, f"{subj}.csv"))
    bc = pd.read_csv(BUNDLE_CLASS_PATHS[subj])

    df = pd.DataFrame({
        "bundle": bc["bundle"].values,
        "commit_acc": il["commit_acc"].values,
        "sift_acc": il["sift_acc"].values,
    })

    commit_yes = df["commit_acc"] >= 1.0
    sift_yes = df["sift_acc"] >= 1.0

    df["foundational"] = commit_yes & sift_yes
    df["commit_only"] = commit_yes & ~sift_yes
    df["sift_only"] = ~commit_yes & sift_yes
    df["either"] = commit_yes | sift_yes
    df["neither"] = ~commit_yes & ~sift_yes

    classified = df[df["bundle"] != "unclassified"]
    bundles = sorted(classified["bundle"].unique())

    for bundle_name in bundles:
        b = classified[classified["bundle"] == bundle_name]
        n = len(b)
        if n == 0:
            continue

        rates = {
            "foundational": b["foundational"].mean() * 100,
            "commit_only": b["commit_only"].mean() * 100,
            "sift_only": b["sift_only"].mean() * 100,
            "either": b["either"].mean() * 100,
            "neither": b["neither"].mean() * 100,
        }

        for cat, rate in rates.items():
            bundle_rates[bundle_name][cat].append(rate)

        all_rows.append({
            "subject": subj,
            "bundle": bundle_name,
            "n": n,
            **{f"pct_{k}": v for k, v in rates.items()},
        })

    # Also do unclassified
    u = df[df["bundle"] == "unclassified"]
    all_rows.append({
        "subject": subj,
        "bundle": "unclassified",
        "n": len(u),
        "pct_foundational": u["foundational"].mean() * 100,
        "pct_commit_only": u["commit_only"].mean() * 100,
        "pct_sift_only": u["sift_only"].mean() * 100,
        "pct_either": u["either"].mean() * 100,
        "pct_neither": u["neither"].mean() * 100,
    })

# Save detailed results
detail_df = pd.DataFrame(all_rows)
detail_df.to_csv("Data/Intersection_Labels/bundle_filtering_detailed.csv", index=False)

# Cross-subject averages
print(f"{'Bundle':20s} {'Found':>7s} {'COMMIT':>7s} {'SIFT':>7s} {'Either':>7s} {'Neither':>7s}")
print("-" * 60)

avg_data = []
for bundle_name in sorted(bundle_rates.keys()):
    r = bundle_rates[bundle_name]
    if len(r["foundational"]) < len(subjects):
        continue
    row = {
        "bundle": bundle_name,
        "foundational": np.mean(r["foundational"]),
        "commit_only": np.mean(r["commit_only"]),
        "sift_only": np.mean(r["sift_only"]),
        "either": np.mean(r["either"]),
        "neither": np.mean(r["neither"]),
    }
    avg_data.append(row)

# Sort by "either" rate descending
avg_data.sort(key=lambda x: -x["either"])

for row in avg_data:
    print(f"{row['bundle']:20s} {row['foundational']:6.1f}% {row['commit_only']:6.1f}% "
          f"{row['sift_only']:6.1f}% {row['either']:6.1f}% {row['neither']:6.1f}%")

# Unclassified summary
print(f"\n{'='*60}")
print("UNCLASSIFIED STREAMLINES (avg across subjects):")
u_rows = [r for r in all_rows if r["bundle"] == "unclassified"]
u_df = pd.DataFrame(u_rows)
print(f"  Foundational: {u_df['pct_foundational'].mean():.1f}%")
print(f"  COMMIT-only:  {u_df['pct_commit_only'].mean():.1f}%")
print(f"  SIFT-only:    {u_df['pct_sift_only'].mean():.1f}%")
print(f"  Either:       {u_df['pct_either'].mean():.1f}%")
print(f"  Neither:      {u_df['pct_neither'].mean():.1f}%")

# ANOVA per category
print(f"\n{'='*60}")
print("ANOVA: Do rates differ across bundles?")
common_bundles = sorted([b for b in bundle_rates if len(bundle_rates[b]["foundational"]) == len(subjects)])

for cat in ["foundational", "commit_only", "sift_only", "either", "neither"]:
    groups = [bundle_rates[b][cat] for b in common_bundles]
    f_stat, p_val = stats.f_oneway(*groups)
    print(f"  {cat:15s}: F={f_stat:6.2f}, p={p_val:.2e}")

# Kruskal-Wallis (non-parametric, better for small N)
print(f"\nKruskal-Wallis (non-parametric):")
for cat in ["foundational", "commit_only", "sift_only", "either", "neither"]:
    groups = [bundle_rates[b][cat] for b in common_bundles]
    h_stat, p_val = stats.kruskal(*groups)
    print(f"  {cat:15s}: H={h_stat:6.2f}, p={p_val:.2e}")

print(f"\nDetailed results saved to Data/Intersection_Labels/bundle_filtering_detailed.csv")
