"""
Graded redundancy analysis for TractSeg bundles.

Companion to the existing analyze_bundle_filtering_detailed.py, which uses only
the binary >=1.0 acceptance threshold. This script reports the same per-bundle
breakdown but with:
  - continuous mean / median / high-percentile weight statistics
  - foundational / either rates at thresholds 0.5, 1.0, 1.5, 2.0
  - cluster-level summed weight ("explained signal")
  - threshold-sensitivity check on the top/bottom bundle ranking
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("TRACTO_ROOT", "/home/maia-user/tracto_work")
INTERSECTION_DIR = f"{ROOT}/Data/Intersection_Labels/all_subjects"
BUNDLE_CLASS_PATHS = {
    "665254": f"{ROOT}/Data/Tractogram_Data/TEST/665254_bundle_classifications.csv",
    "734045": f"{ROOT}/Data/Tractogram_Data/TRAIN/734045_bundle_classifications.csv",
    "784565": f"{ROOT}/Data/Tractogram_Data/VAL/784565_bundle_classifications.csv",
    "837964": f"{ROOT}/Data/Tractogram_Data/TEST/837964_bundle_classifications.csv",
    "877168": f"{ROOT}/Data/Tractogram_Data/TEST/877168_bundle_classifications.csv",
}
SUBJECTS = list(BUNDLE_CLASS_PATHS.keys())
# rSIFT/rCOMMIT acceptance weights are probabilities in [0, 1] over random
# subsets, so the threshold controls cross-subset consensus strictness rather
# than a magnitude. Values above 1.0 are not meaningful.
THRESHOLDS = (0.25, 0.5, 0.75, 1.0)


def per_bundle_graded(dfs: dict) -> pd.DataFrame:
    rows = []
    for subj, df in dfs.items():
        classified = df[df["bundle"] != "unclassified"]
        for bundle, sub in classified.groupby("bundle"):
            commit = sub["commit_acc"].values
            sift = sub["sift_acc"].values
            row = {
                "subject": subj, "bundle": bundle, "n": len(sub),
                "mean_commit": commit.mean(),
                "median_commit": np.median(commit),
                "p75_commit": np.percentile(commit, 75),
                "p90_commit": np.percentile(commit, 90),
                "p95_commit": np.percentile(commit, 95),
                "mean_sift": sift.mean(),
                "median_sift": np.median(sift),
                "p75_sift": np.percentile(sift, 75),
                "p90_sift": np.percentile(sift, 90),
                "p95_sift": np.percentile(sift, 95),
                "sum_commit": commit.sum(),
                "sum_sift": sift.sum(),
            }
            for thr in THRESHOLDS:
                row[f"pct_foundational_thr{thr}"] = (
                    ((commit >= thr) & (sift >= thr)).mean() * 100
                )
                row[f"pct_either_thr{thr}"] = (
                    ((commit >= thr) | (sift >= thr)).mean() * 100
                )
                # GRADED REDUNDANCY: % of bundle's streamlines that the filters
                # both downweight below this threshold. High value = classified
                # by TractSeg but signal-redundant.
                row[f"pct_redundant_thr{thr}"] = (
                    ((commit < thr) & (sift < thr)).mean() * 100
                )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(per_subj: pd.DataFrame, n_subjects: int) -> pd.DataFrame:
    out = []
    numeric_cols = [c for c in per_subj.columns if c not in ("subject", "bundle")]
    for bundle, g in per_subj.groupby("bundle"):
        if len(g) < n_subjects:
            continue
        row = {"bundle": bundle, "avg_n": g["n"].mean()}
        for c in numeric_cols:
            if c == "n":
                continue
            row[c] = g[c].mean()
        # Per-bundle SD for the canonical metric
        row["foundational_thr1.0_sd"] = g["pct_foundational_thr1.0"].std()
        out.append(row)
    return pd.DataFrame(out)


def main():
    dfs = {}
    for subj in SUBJECTS:
        il = pd.read_csv(os.path.join(INTERSECTION_DIR, f"{subj}.csv"))
        bc = pd.read_csv(BUNDLE_CLASS_PATHS[subj])
        if len(il) != len(bc):
            print(f"{subj}: row count mismatch ({len(il)} vs {len(bc)})"); continue
        dfs[subj] = pd.DataFrame({
            "bundle": bc["bundle"].values,
            "commit_acc": il["commit_acc"].values,
            "sift_acc": il["sift_acc"].values,
        })
        print(f"  {subj}: {len(dfs[subj]):,} streamlines")
    n_sub = len(dfs)

    out_dir = f"{ROOT}/Data/Intersection_Labels/graded_bundles"
    os.makedirs(out_dir, exist_ok=True)

    print("\nPer-bundle graded metrics (per subject)")
    per_subj = per_bundle_graded(dfs)
    per_subj.to_csv(f"{out_dir}/per_bundle_per_subject.csv", index=False)
    print(f"  rows: {len(per_subj):,} (bundles × subjects)")

    print("\nCross-subject averages (bundles present in all subjects)")
    agg = aggregate(per_subj, n_sub)
    agg.to_csv(f"{out_dir}/per_bundle_averaged.csv", index=False)
    print(f"  {len(agg)} bundles present in all {n_sub} subjects")

    if len(agg) == 0:
        return

    print("\n== Continuous-mean COMMIT weight ranking ==")
    cols_show = ["bundle", "avg_n", "mean_commit", "mean_sift",
                 "median_commit", "p90_commit",
                 "pct_foundational_thr1.0"]
    cols_show = [c for c in cols_show if c in agg.columns]
    print("\nTop 10 by mean COMMIT weight:")
    print(agg.sort_values("mean_commit", ascending=False).head(10)[cols_show].to_string(index=False))
    print("\nBottom 10 by mean COMMIT weight:")
    print(agg.sort_values("mean_commit", ascending=True).head(10)[cols_show].to_string(index=False))

    print("\n== Foundational rate at multiple thresholds ==")
    thr_cols = [c for c in agg.columns if c.startswith("pct_foundational_thr")]
    print("\nTop 5 bundles by foundational rate, per threshold:")
    for c in thr_cols:
        thr = c.split("_thr")[1]
        top5 = agg.sort_values(c, ascending=False).head(5)
        names = ", ".join(f"{r['bundle']} ({r[c]:.2f}%)" for _, r in top5.iterrows())
        print(f"  thr={thr}: {names}")

    # GRADED REDUNDANCY: where in the atlas is the heavy "classified but downweighted" mass?
    print("\n== Graded redundancy across bundles ==")
    red_cols = [c for c in agg.columns if c.startswith("pct_redundant_thr")]
    for c in red_cols:
        thr = c.split("_thr")[1]
        print(f"\nTop 10 most-redundant bundles, threshold {thr} "
              f"(% with both commit<{thr} AND sift<{thr}):")
        show = ["bundle", "avg_n", c, "mean_commit", "mean_sift"]
        show = [s for s in show if s in agg.columns]
        print(agg.sort_values(c, ascending=False).head(10)[show].to_string(index=False))

    # Volume × redundancy: which bundles waste the most streamlines?
    if "pct_redundant_thr1.0" in agg.columns:
        agg["wasted_streamlines"] = agg["avg_n"] * agg["pct_redundant_thr1.0"] / 100.0
        print("\n== Largest classified-but-redundant TractSeg bundles ==")
        print("(top 10 by avg_n × pct_redundant_thr1.0 — where the redundancy concentrates)")
        print(agg.sort_values("wasted_streamlines", ascending=False).head(10)[
            ["bundle", "avg_n", "pct_redundant_thr1.0", "wasted_streamlines",
             "mean_commit", "mean_sift"]].to_string(index=False))

    print("\nThreshold sensitivity — Jaccard overlap of top-10 sets:")
    top_sets = {c: set(agg.sort_values(c, ascending=False).head(10)["bundle"]) for c in thr_cols}
    for i, c1 in enumerate(thr_cols):
        for c2 in thr_cols[i+1:]:
            t1, t2 = c1.split("_thr")[1], c2.split("_thr")[1]
            inter = top_sets[c1] & top_sets[c2]
            union = top_sets[c1] | top_sets[c2]
            j = len(inter) / len(union) if union else 0.0
            print(f"  Jaccard(thr={t1}, thr={t2}) = {j:.2f}  ({len(inter)}/{len(union)} bundles overlap)")

    print(f"\nOutputs → {out_dir}/")


if __name__ == "__main__":
    main()
