"""
Analyze SUP-WMA results across the 5 subjects.

Mirrors the structure of analyze_bundle_filtering_detailed.py, but partitions
streamlines by SUP-WMA cluster (198 superficial WM clusters + "all_others")
instead of TractSeg bundle.

Outputs:
  1. Per-subject coverage: % SWM-classified, vs TractSeg-classified.
  2. SUP-WMA × TractSeg cross-tab (one per subject + average).
  3. Per-SUP-WMA-cluster acceptance rates (foundational, commit-only, sift-only,
     either, neither) — averaged across subjects.
  4. ANOVA + Kruskal-Wallis on per-cluster rates.
  5. Top/bottom SUP-WMA clusters by foundational rate.
  6. Cross-subject reproducibility (which clusters carry foundational streamlines
     in all 5 subjects).
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("TRACTO_ROOT", "/home/maia-user/tracto_work")
INTERSECTION_DIR = f"{ROOT}/Data/Intersection_Labels/all_subjects"
SUPWMA_DIR = f"{ROOT}/Data/SupWMA"

BUNDLE_CLASS_PATHS = {
    "665254": f"{ROOT}/Data/Tractogram_Data/TEST/665254_bundle_classifications.csv",
    "734045": f"{ROOT}/Data/Tractogram_Data/TRAIN/734045_bundle_classifications.csv",
    "784565": f"{ROOT}/Data/Tractogram_Data/VAL/784565_bundle_classifications.csv",
    "837964": f"{ROOT}/Data/Tractogram_Data/TEST/837964_bundle_classifications.csv",
    "877168": f"{ROOT}/Data/Tractogram_Data/TEST/877168_bundle_classifications.csv",
}
SUBJECTS = list(BUNDLE_CLASS_PATHS.keys())


def load_subject(subj: str) -> pd.DataFrame:
    """Build a single DataFrame per subject merging weights + TractSeg + SUP-WMA."""
    il = pd.read_csv(os.path.join(INTERSECTION_DIR, f"{subj}.csv"))
    bc = pd.read_csv(BUNDLE_CLASS_PATHS[subj])
    sw = pd.read_csv(os.path.join(SUPWMA_DIR, f"{subj}_supwma_labels.csv"))
    n_il, n_bc, n_sw = len(il), len(bc), len(sw)
    if not (n_il == n_bc == n_sw):
        raise ValueError(
            f"{subj}: row count mismatch — il={n_il}, bc={n_bc}, sw={n_sw}")
    df = pd.DataFrame({
        "commit_acc": il["commit_acc"].values,
        "sift_acc": il["sift_acc"].values,
        "bundle": bc["bundle"].values,
        "supwma_cluster_name": sw["supwma_cluster_name"].values,
        "supwma_cluster_idx": sw["supwma_cluster_idx"].values,
        "supwma_rejected": sw["rejected"].values.astype(bool),
    })
    commit_yes = df["commit_acc"] >= 1.0
    sift_yes = df["sift_acc"] >= 1.0
    df["foundational"] = commit_yes & sift_yes
    df["commit_only"] = commit_yes & ~sift_yes
    df["sift_only"] = ~commit_yes & sift_yes
    df["either"] = commit_yes | sift_yes
    df["neither"] = ~commit_yes & ~sift_yes
    return df


def coverage_stats(dfs: dict) -> pd.DataFrame:
    rows = []
    for subj, df in dfs.items():
        n = len(df)
        n_swm = (~df["supwma_rejected"]).sum()
        n_tractseg = (df["bundle"] != "unclassified").sum()
        n_either = ((df["bundle"] != "unclassified") | (~df["supwma_rejected"])).sum()
        n_both_class = ((df["bundle"] != "unclassified") & (~df["supwma_rejected"])).sum()
        n_doubly_unc = ((df["bundle"] == "unclassified") & df["supwma_rejected"]).sum()
        rows.append({
            "subject": subj,
            "n_streamlines": n,
            "pct_tractseg": 100*n_tractseg/n,
            "pct_supwma": 100*n_swm/n,
            "pct_either_atlas": 100*n_either/n,
            "pct_both_atlases": 100*n_both_class/n,
            "pct_doubly_unclassified": 100*n_doubly_unc/n,
        })
    return pd.DataFrame(rows)


def coverage_among_foundational(dfs: dict) -> pd.DataFrame:
    """For foundational streamlines specifically: how many does SUP-WMA recover?"""
    rows = []
    for subj, df in dfs.items():
        f = df[df["foundational"]]
        n = len(f)
        if n == 0:
            continue
        n_swm = (~f["supwma_rejected"]).sum()
        n_tractseg = (f["bundle"] != "unclassified").sum()
        n_either = ((f["bundle"] != "unclassified") | (~f["supwma_rejected"])).sum()
        n_doubly_unc = ((f["bundle"] == "unclassified") & f["supwma_rejected"]).sum()
        rows.append({
            "subject": subj,
            "n_foundational": n,
            "pct_foundational_tractseg": 100*n_tractseg/n,
            "pct_foundational_supwma": 100*n_swm/n,
            "pct_foundational_either_atlas": 100*n_either/n,
            "pct_foundational_doubly_unclassified": 100*n_doubly_unc/n,
        })
    return pd.DataFrame(rows)


def coverage_threshold_sweep(dfs: dict,
                             thresholds=(0.0, 0.25, 0.5, 0.75, 1.0)) -> pd.DataFrame:
    """Recovery curves: vary the rCOMMIT/rSIFT foundational threshold.

    For each threshold T, redefine foundational as commit >= T AND sift >= T,
    then report what fraction of those streamlines TractSeg / SUP-WMA / either
    classifies. The 'rate' direction matters because the paper's 1.0 cutoff is
    arbitrary; this sweep shows whether the headline finding is threshold-stable.
    """
    rows = []
    for subj, df in dfs.items():
        commit = df["commit_acc"].values
        sift = df["sift_acc"].values
        ts_cls = (df["bundle"] != "unclassified").values
        sw_cls = (~df["supwma_rejected"]).values
        for T in thresholds:
            mask = (commit >= T) & (sift >= T)
            n = int(mask.sum())
            if n == 0:
                rows.append({
                    "subject": subj, "threshold": T, "n_foundational": 0,
                    "pct_tractseg": np.nan, "pct_supwma": np.nan,
                    "pct_either": np.nan, "pct_doubly_unclassified": np.nan,
                })
                continue
            rows.append({
                "subject": subj, "threshold": T, "n_foundational": n,
                "pct_tractseg": 100 * (ts_cls & mask).sum() / n,
                "pct_supwma": 100 * (sw_cls & mask).sum() / n,
                "pct_either": 100 * ((ts_cls | sw_cls) & mask).sum() / n,
                "pct_doubly_unclassified": 100 * ((~ts_cls & ~sw_cls) & mask).sum() / n,
            })
    return pd.DataFrame(rows)


def coverage_weight_quantiles(dfs: dict,
                              quantiles=(0.0, 0.5, 0.75, 0.9, 0.95, 0.99)) -> pd.DataFrame:
    """Threshold-free graded view: rank streamlines by min(commit, sift) and
    examine the top X% of streamlines — what fraction is in each atlas?

    Avoids picking a unit-meaningful threshold; instead picks the highest-
    weighted streamlines by quantile. The top 1% by min-weight is a strong
    operational definition of "most essential streamlines."
    """
    rows = []
    for subj, df in dfs.items():
        # Lowest of the two weights = streamline's "weakest case" essentialness
        min_w = np.minimum(df["commit_acc"].values, df["sift_acc"].values)
        ts_cls = (df["bundle"] != "unclassified").values
        sw_cls = (~df["supwma_rejected"]).values
        n_total = len(df)
        for q in quantiles:
            # Top (1 - q) fraction = streamlines whose min_weight >= the q-quantile cutoff
            cutoff = np.quantile(min_w, q) if q > 0 else -np.inf
            mask = min_w >= cutoff
            n = int(mask.sum())
            rows.append({
                "subject": subj, "min_weight_quantile": q,
                "weight_cutoff": float(cutoff), "n_in_band": n,
                "pct_of_total": 100 * n / n_total,
                "pct_tractseg": 100 * (ts_cls & mask).sum() / n,
                "pct_supwma": 100 * (sw_cls & mask).sum() / n,
                "pct_either": 100 * ((ts_cls | sw_cls) & mask).sum() / n,
                "pct_doubly_unclassified": 100 * ((~ts_cls & ~sw_cls) & mask).sum() / n,
            })
    return pd.DataFrame(rows)


def supwma_x_tractseg(dfs: dict) -> pd.DataFrame:
    """How SUP-WMA labels distribute across TractSeg-classified vs unclassified."""
    rows = []
    for subj, df in dfs.items():
        total = len(df)
        # TractSeg classified
        ts_c = df["bundle"] != "unclassified"
        ts_u = ~ts_c
        sw_c = ~df["supwma_rejected"]
        sw_u = df["supwma_rejected"]
        rows.append({
            "subject": subj,
            "ts_yes_sw_yes": (ts_c & sw_c).sum(),
            "ts_yes_sw_no":  (ts_c & sw_u).sum(),
            "ts_no_sw_yes":  (ts_u & sw_c).sum(),
            "ts_no_sw_no":   (ts_u & sw_u).sum(),
            "total": total,
        })
    return pd.DataFrame(rows)


def per_cluster_rates(dfs: dict) -> pd.DataFrame:
    """Compute per-cluster acceptance rates per subject, exclude 'all_others'.

    Includes both binary-threshold rates (the original metric, thr=1.0) AND
    graded weight statistics: rates at several thresholds, continuous mean,
    median and high-percentile values per cluster. The graded view sidesteps
    the arbitrariness of the 1.0 threshold called out in the paper's discussion.
    """
    # rSIFT and rCOMMIT acceptance weights are probabilities over random subsets,
    # so they live in [0, 1] by construction. The threshold therefore controls
    # how strict the cross-subset consensus must be: 1.0 = "always accepted",
    # 0.5 = "majority accepted", 0.25 = "accepted in at least 1/4 of subsets".
    # Thresholds above 1.0 are not meaningful.
    GRADED_THRESHOLDS = (0.25, 0.5, 0.75, 1.0)
    rows = []
    for subj, df in dfs.items():
        sw_classified = df[~df["supwma_rejected"]]
        for name, sub in sw_classified.groupby("supwma_cluster_name"):
            n = len(sub)
            if n == 0:
                continue
            commit = sub["commit_acc"].values
            sift = sub["sift_acc"].values
            row = {
                "subject": subj,
                "supwma_cluster": name,
                "n": n,
                # Original (binary) acceptance rates at threshold 1.0
                "pct_foundational": sub["foundational"].mean()*100,
                "pct_commit_only": sub["commit_only"].mean()*100,
                "pct_sift_only": sub["sift_only"].mean()*100,
                "pct_either": sub["either"].mean()*100,
                "pct_neither": sub["neither"].mean()*100,
                # Continuous weight statistics
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
                # Cluster-level "explained signal" — total weight summed.
                # Larger clusters can sum to more even with low per-streamline weights,
                # so this is a complementary view to the per-streamline percentages.
                "sum_commit": commit.sum(),
                "sum_sift": sift.sum(),
            }
            # Foundational rate at multiple thresholds: both commit and sift >= thr
            for thr in GRADED_THRESHOLDS:
                row[f"pct_foundational_thr{thr}"] = (
                    ((commit >= thr) & (sift >= thr)).mean() * 100
                )
                row[f"pct_either_thr{thr}"] = (
                    ((commit >= thr) | (sift >= thr)).mean() * 100
                )
                # Graded REDUNDANCY: classified by atlas yet downweighted by both
                # filters below this threshold. High value → cluster is anatomically
                # well-defined but signal-redundant.
                row[f"pct_redundant_thr{thr}"] = (
                    ((commit < thr) & (sift < thr)).mean() * 100
                )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_clusters(per_subj_rates: pd.DataFrame, n_subjects: int) -> pd.DataFrame:
    """Average per-cluster rates across subjects (only clusters present in all)."""
    by_cluster = per_subj_rates.groupby("supwma_cluster")
    # Numeric columns to average across subjects; pass through cluster name as group key.
    numeric_cols = [c for c in per_subj_rates.columns
                    if c not in ("subject", "supwma_cluster")]
    rows = []
    for cluster, g in by_cluster:
        if len(g) < n_subjects:
            continue
        row = {"supwma_cluster": cluster}
        for c in numeric_cols:
            row[c] = g[c].mean()
        # Also keep SD of the headline metric for the existing top/bottom tables
        row["foundational_sd"] = g["pct_foundational"].std()
        rows.append(row)
    df = pd.DataFrame(rows)
    if "n" in df.columns:
        df = df.rename(columns={"n": "avg_n"})
    if "pct_foundational" in df.columns:
        df = df.rename(columns={"pct_foundational": "foundational",
                                "pct_either": "either",
                                "pct_neither": "neither",
                                "pct_commit_only": "commit_only",
                                "pct_sift_only": "sift_only"})
    return df


def anova_kruskal(per_subj_rates: pd.DataFrame, n_subjects: int) -> pd.DataFrame:
    common = (per_subj_rates.groupby("supwma_cluster").size() == n_subjects)
    common = common[common].index.tolist()
    sub = per_subj_rates[per_subj_rates["supwma_cluster"].isin(common)]
    out = []
    for cat in ["pct_foundational", "pct_commit_only", "pct_sift_only", "pct_either", "pct_neither"]:
        groups = [g[cat].values for _, g in sub.groupby("supwma_cluster")]
        f_stat, p_anova = stats.f_oneway(*groups)
        h_stat, p_kw = stats.kruskal(*groups)
        out.append({
            "category": cat.replace("pct_", ""),
            "n_clusters": len(common),
            "F": f_stat, "p_anova": p_anova,
            "H": h_stat, "p_kruskal": p_kw,
        })
    return pd.DataFrame(out)


def cross_subject_reproducibility(per_subj_rates: pd.DataFrame, n_subjects: int) -> dict:
    """How many SUP-WMA clusters are present (any foundational streamlines) in all subjects?"""
    found_present = per_subj_rates[per_subj_rates["pct_foundational"] > 0]
    presence = found_present.groupby("supwma_cluster")["subject"].nunique()
    return {
        "in_all_subjects": int((presence == n_subjects).sum()),
        "in_at_least_one": int(len(presence)),
    }


def main():
    print("=" * 70)
    print("Loading subjects...")
    dfs = {}
    for subj in SUBJECTS:
        path = os.path.join(SUPWMA_DIR, f"{subj}_supwma_labels.csv")
        if not os.path.exists(path):
            print(f"  {subj}: missing → {path}, skipping")
            continue
        dfs[subj] = load_subject(subj)
        print(f"  {subj}: {len(dfs[subj]):,} streamlines")
    if not dfs:
        print("No subjects loaded. Run run_supwma_subject.py first.")
        return
    n_sub = len(dfs)

    out_dir = f"{ROOT}/Data/SupWMA/analysis"
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("1. Coverage stats (all streamlines)")
    cov = coverage_stats(dfs)
    print(cov.to_string(index=False))
    cov.to_csv(f"{out_dir}/coverage_all.csv", index=False)

    print("\n" + "=" * 70)
    print("2. Coverage among foundational streamlines (threshold=1.0, paper definition)")
    cov_f = coverage_among_foundational(dfs)
    print(cov_f.to_string(index=False))
    cov_f.to_csv(f"{out_dir}/coverage_foundational.csv", index=False)

    print("\n" + "=" * 70)
    print("2b. Graded coverage: TractSeg/SUP-WMA recovery vs. rSIFT/rCOMMIT threshold")
    print("    (foundational redefined as commit>=T AND sift>=T for each T)")
    sweep = coverage_threshold_sweep(dfs)
    print(sweep.to_string(index=False))
    sweep.to_csv(f"{out_dir}/coverage_threshold_sweep.csv", index=False)

    print("\n" + "=" * 70)
    print("2c. Threshold-free coverage: top X% of streamlines by min(commit,sift) weight")
    qview = coverage_weight_quantiles(dfs)
    print(qview.to_string(index=False))
    qview.to_csv(f"{out_dir}/coverage_weight_quantiles.csv", index=False)

    print("\n" + "=" * 70)
    print("3. SUP-WMA × TractSeg cross-tab (counts)")
    xt = supwma_x_tractseg(dfs)
    print(xt.to_string(index=False))
    xt.to_csv(f"{out_dir}/supwma_x_tractseg.csv", index=False)

    print("\n" + "=" * 70)
    print("4. Per-cluster acceptance rates (per subject)")
    per_subj = per_cluster_rates(dfs)
    per_subj.to_csv(f"{out_dir}/per_cluster_per_subject.csv", index=False)
    print(f"  saved → {out_dir}/per_cluster_per_subject.csv "
          f"({len(per_subj):,} rows across {n_sub} subjects)")

    print("\n" + "=" * 70)
    print("5. Cross-subject averages (clusters present in all subjects)")
    agg = aggregate_clusters(per_subj, n_sub)
    print(f"  {len(agg)} clusters present in all {n_sub} subjects")
    agg.to_csv(f"{out_dir}/per_cluster_averaged.csv", index=False)

    if len(agg) > 0:
        print("\n  Top 10 by foundational rate, threshold=1.0 (lowest redundancy):")
        top = agg.sort_values("foundational", ascending=False).head(10)
        print(top[["supwma_cluster", "avg_n", "foundational", "foundational_sd",
                   "either", "neither"]].to_string(index=False))

        print("\n  Bottom 10 by foundational rate, threshold=1.0 (highest redundancy):")
        bot = agg.sort_values("foundational", ascending=True).head(10)
        print(bot[["supwma_cluster", "avg_n", "foundational",
                   "either", "neither"]].to_string(index=False))

        # Graded view: top clusters by continuous mean weight (no arbitrary threshold)
        print("\n  Top 10 by MEAN commit weight (graded, no threshold):")
        cols_mean = ["supwma_cluster", "avg_n", "mean_commit", "mean_sift",
                     "median_commit", "median_sift", "p95_commit", "p95_sift"]
        cols_mean = [c for c in cols_mean if c in agg.columns]
        print(agg.sort_values("mean_commit", ascending=False).head(10)[cols_mean].to_string(index=False))

        # GRADED REDUNDANCY: where is the heavy redundancy concentrated?
        # "Classified by atlas but downweighted by both filters" — sort by
        # how many streamlines fall under each redundancy threshold.
        print("\n  Most-redundant SUP-WMA clusters (highest %% with BOTH weights below threshold):")
        for thr in (0.25, 0.5, 1.0):
            col = f"pct_redundant_thr{thr}"
            if col not in agg.columns:
                continue
            # Restrict to "real" clusters with n >= 100 avg to avoid noise from tiny clusters
            big = agg[agg["avg_n"] >= 100].copy()
            top_red = big.sort_values(col, ascending=False).head(10)
            print(f"\n  Threshold {thr} — top-10 most-redundant clusters (≥100 streamlines avg):")
            show = ["supwma_cluster", "avg_n", col, "mean_commit", "mean_sift"]
            print(top_red[show].to_string(index=False))

        # The big counterintuitive ones: high atlas confidence (LARGE avg_n)
        # AND high redundancy at thr=1.0. These dominate the "wasted" streamlines.
        if "pct_redundant_thr1.0" in agg.columns:
            print("\n  Largest classified-but-redundant clusters "
                  "(top 10 by avg_n × pct_redundant_thr1.0):")
            agg["wasted_streamlines"] = agg["avg_n"] * agg["pct_redundant_thr1.0"] / 100.0
            big_red = agg.sort_values("wasted_streamlines", ascending=False).head(10)
            print(big_red[["supwma_cluster", "avg_n", "pct_redundant_thr1.0",
                           "wasted_streamlines", "mean_commit",
                           "mean_sift"]].to_string(index=False))

        # Threshold sensitivity: rank stability across multiple thresholds
        thr_cols = [c for c in agg.columns if c.startswith("pct_foundational_thr")]
        if thr_cols:
            print("\n  Threshold sensitivity — top-10 cluster overlap across thresholds:")
            top_sets = {}
            for c in thr_cols:
                top10 = set(agg.sort_values(c, ascending=False).head(10)["supwma_cluster"])
                top_sets[c] = top10
                print(f"    {c}: top10 size {len(top10)}")
            # Pairwise Jaccard
            for i, c1 in enumerate(thr_cols):
                for c2 in thr_cols[i+1:]:
                    inter = top_sets[c1] & top_sets[c2]
                    union = top_sets[c1] | top_sets[c2]
                    j = len(inter) / len(union) if union else 0.0
                    print(f"    Jaccard({c1.split('_thr')[1]}, {c2.split('_thr')[1]}) = {j:.2f}")

    print("\n" + "=" * 70)
    print("6. ANOVA + Kruskal-Wallis: do rates differ across SUP-WMA clusters?")
    if n_sub >= 2:
        ak = anova_kruskal(per_subj, n_sub)
        print(ak.to_string(index=False))
        ak.to_csv(f"{out_dir}/anova_kruskal.csv", index=False)
    else:
        print(f"  only {n_sub} subject(s) loaded → skipping statistical tests")

    print("\n" + "=" * 70)
    print("7. Cross-subject reproducibility")
    rep = cross_subject_reproducibility(per_subj, n_sub)
    print(f"  Clusters with foundational streamlines in ALL subjects: {rep['in_all_subjects']}")
    print(f"  Clusters with foundational streamlines in ≥1 subject : {rep['in_at_least_one']}")

    print("\n" + "=" * 70)
    print(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
