"""
Characterize the redundant pool across all five HCP subjects.

Per subject:
  - bundle distribution of redundant streamlines (TractSeg + SupWMA)
  - length distribution of redundant vs foundational
  - MDF distance from each redundant streamline to nearest foundational

Aggregates into a single figure and prints a cross-subject table.

Definitions (t = 0.9):
  foundational := sift_acc $\\geq t$ AND commit_acc $\\geq t$
  redundant    := $0 < $ sift_acc $< t$ AND $0 < $ commit_acc $< t$
"""

import os
import time
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("TRACTO_ROOT", "/home/maia-user/tracto_work")
SUBJECTS = ["665254", "734045", "784565", "837964", "877168"]
OUT_DIR = f"{ROOT}/Streamlines-redundancy/figures"

T = 0.9
N_POINTS = 20
REDUNDANT_CAP_FOR_MDF = 5000
FOUNDATIONAL_CAP_FOR_MDF = 30000
LENGTH_SAMPLE = 20000          # streamlines sampled per pool for length histogram
TOP_K_BUNDLES = 15


def locate_subject_files(subj):
    """Search TEST/TRAIN/VAL for the subject's tractogram + bundle CSV."""
    for split in ("TEST", "TRAIN", "VAL"):
        trk = f"{ROOT}/Data/Tractogram_Data/{split}/{subj}.trk"
        bcsv = f"{ROOT}/Data/Tractogram_Data/{split}/{subj}_bundle_classifications.csv"
        if os.path.exists(trk) and os.path.exists(bcsv):
            return trk, bcsv
    return None, None


def resample_streamline(s, n):
    s = np.asarray(s, dtype=np.float32)
    if s.shape[0] == n:
        return s
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(s, axis=0), axis=1))])
    if cum[-1] == 0:
        return np.tile(s[:1], (n, 1)).astype(np.float32)
    ts = np.linspace(0, cum[-1], n)
    idx = np.searchsorted(cum, ts).clip(1, len(cum) - 1)
    t0, t1 = cum[idx - 1], cum[idx]
    f = np.where(t1 > t0, (ts - t0) / np.where(t1 > t0, t1 - t0, 1.0), 0.0)
    return ((1 - f)[:, None] * s[idx - 1] + f[:, None] * s[idx]).astype(np.float32)


def streamline_length(s):
    s = np.asarray(s)
    return float(np.linalg.norm(np.diff(s, axis=0), axis=1).sum()) if len(s) > 1 else 0.0


def mdf_chunked(R, F, F_flip, chunk=128):
    Nr = R.shape[0]
    out_d = np.empty(Nr, dtype=np.float32)
    for s in range(0, Nr, chunk):
        e = min(s + chunk, Nr)
        diff_d = R[s:e, None, :, :] - F[None, :, :, :]
        d_direct = np.sqrt((diff_d ** 2).sum(-1)).mean(-1)
        diff_f = R[s:e, None, :, :] - F_flip[None, :, :, :]
        d_flip = np.sqrt((diff_f ** 2).sum(-1)).mean(-1)
        out_d[s:e] = np.minimum(d_direct, d_flip).min(axis=1)
    return out_d


def analyze_subject(subj):
    print(f"\n========== Subject {subj} ==========")
    t_subj = time.time()
    trk_path, bundle_csv = locate_subject_files(subj)
    intersection_csv = f"{ROOT}/Data/Intersection_Labels/all_subjects/{subj}.csv"
    supwma_csv = f"{ROOT}/Data/SupWMA/{subj}_supwma_labels.csv"

    bc = pd.read_csv(bundle_csv)
    il = pd.read_csv(intersection_csv)
    sw = pd.read_csv(supwma_csv, usecols=["supwma_cluster_name", "rejected"])
    bundle = bc["bundle"].values
    cn = sw["supwma_cluster_name"].values
    supwma_kept = sw["rejected"].values == 0
    sa = il["sift_acc"].values
    ca = il["commit_acc"].values

    foundational = (sa >= T) & (ca >= T)
    redundant = (sa > 0) & (sa < T) & (ca > 0) & (ca < T)
    n_red = int(redundant.sum())
    n_fnd = int(foundational.sum())
    print(f"  foundational={n_fnd:,}  redundant={n_red:,}")

    # Bundle distribution
    red_bundle = pd.Series(bundle[redundant]).value_counts()
    unclass_n = int(red_bundle.get("unclassified", 0))
    top_named = red_bundle[red_bundle.index != "unclassified"].head(TOP_K_BUNDLES)
    print(f"  unclassified in redundant: {unclass_n:,}  ({100*unclass_n/n_red:.1f}%)")
    for bname, n in top_named.head(8).items():
        pool = (bundle == bname).sum()
        print(f"    {bname:<18} {n:>10,}  ({100*n/pool:.1f}% of bundle)")

    # SupWMA membership
    red_supwma_kept = int((redundant & supwma_kept).sum())
    print(f"  redundant ∩ SupWMA-kept: {red_supwma_kept:,}  ({100*red_supwma_kept/n_red:.1f}%)")

    # Load tractogram
    print("  loading tractogram...")
    t0 = time.time()
    trk = nib.streamlines.load(trk_path)
    streams = trk.streamlines
    print(f"  loaded {len(streams):,} streamlines in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(hash(subj) & 0xFFFFFFFF)

    # Length distributions
    red_idx = np.where(redundant)[0]
    fnd_idx = np.where(foundational)[0]
    red_len_sample = rng.choice(red_idx, size=min(LENGTH_SAMPLE, n_red), replace=False)
    red_lens = np.array([streamline_length(streams[int(i)]) for i in red_len_sample])
    fnd_lens = np.array([streamline_length(streams[int(i)]) for i in fnd_idx])
    print(f"  length: redundant mean={red_lens.mean():.1f}mm  "
          f"foundational mean={fnd_lens.mean():.1f}mm")

    # MDF
    red_idx_mdf = rng.choice(red_idx, size=min(REDUNDANT_CAP_FOR_MDF, n_red), replace=False)
    fnd_idx_mdf = (rng.choice(fnd_idx, size=min(FOUNDATIONAL_CAP_FOR_MDF, n_fnd), replace=False)
                   if n_fnd > FOUNDATIONAL_CAP_FOR_MDF else fnd_idx)
    print(f"  MDF: {len(red_idx_mdf):,} redundant × {len(fnd_idx_mdf):,} foundational")
    t0 = time.time()
    R = np.stack([resample_streamline(streams[int(i)], N_POINTS) for i in red_idx_mdf])
    F = np.stack([resample_streamline(streams[int(i)], N_POINTS) for i in fnd_idx_mdf])
    F_flip = F[:, ::-1, :]
    d_min = mdf_chunked(R, F, F_flip, chunk=128)
    print(f"  MDF done in {time.time()-t0:.1f}s  mean={d_min.mean():.2f}mm  "
          f"median={np.median(d_min):.2f}mm")

    print(f"  subject {subj} total time: {time.time()-t_subj:.1f}s")
    return {
        "subj": subj,
        "n_redundant": n_red,
        "n_foundational": n_fnd,
        "unclass_pct": 100 * unclass_n / n_red,
        "supwma_pct": 100 * red_supwma_kept / n_red,
        "top_bundles": top_named.to_dict(),
        "bundle_pool_sizes": {b: int((bundle == b).sum()) for b in top_named.index},
        "red_lens": red_lens,
        "fnd_lens": fnd_lens,
        "d_min": d_min,
    }


def main():
    results = []
    for subj in SUBJECTS:
        trk_path, bundle_csv = locate_subject_files(subj)
        if trk_path is None:
            print(f"\n[skip] subject {subj}: missing tractogram or bundle CSV in any split")
            continue
        print(f"[locate] {subj} -> {trk_path}")
        results.append(analyze_subject(subj))

    # Cross-subject text summary
    print("\n\n========== Cross-subject summary ==========")
    print(f"{'subj':<10} {'n_red':>12} {'n_fnd':>10} {'unclass%':>10} {'supwma%':>9} "
          f"{'redLen':>8} {'fndLen':>8} {'MDFmean':>8} {'MDFmed':>8}")
    for r in results:
        print(f"{r['subj']:<10} {r['n_redundant']:>12,} {r['n_foundational']:>10,} "
              f"{r['unclass_pct']:>9.1f}% {r['supwma_pct']:>8.1f}% "
              f"{r['red_lens'].mean():>7.1f}m {r['fnd_lens'].mean():>7.1f}m "
              f"{r['d_min'].mean():>7.2f}m {np.median(r['d_min']):>7.2f}m")

    # Cross-subject aggregated top-bundles ranking
    print("\nTop named bundles by total redundant count across subjects:")
    bundle_totals = {}
    bundle_subj_counts = {}
    for r in results:
        for b, n in r["top_bundles"].items():
            bundle_totals[b] = bundle_totals.get(b, 0) + n
            bundle_subj_counts[b] = bundle_subj_counts.get(b, 0) + 1
    top_aggregate = sorted(bundle_totals.items(), key=lambda x: -x[1])[:TOP_K_BUNDLES]
    for b, total in top_aggregate:
        print(f"  {b:<18} total={total:>10,}  in {bundle_subj_counts[b]}/5 subjects' top-{TOP_K_BUNDLES}")

    # Figure: 1 row × 3 panels, all 5 subjects overlaid
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SUBJECTS)))

    # Panel 1: cross-subject bundle distribution (aggregated bar)
    ax = axes[0]
    top_names = [b for b, _ in top_aggregate[:12]]
    y = np.arange(len(top_names))[::-1]
    width = 0.16
    for k, r in enumerate(results):
        vals = [r["top_bundles"].get(b, 0) for b in top_names]
        ax.barh(y + (k - 2) * width, vals, height=width,
                color=colors[k], alpha=0.85, label=r["subj"])
    ax.set_yticks(y); ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("# redundant streamlines (log)", fontsize=9)
    ax.set_title("Top named bundles in the redundant pool", fontsize=10)
    ax.legend(fontsize=7, frameon=False, ncol=5, loc="lower right")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 2: length distribution, redundant (one curve per subject) + foundational (one curve per subject)
    ax = axes[1]
    bins = np.linspace(0, 220, 60)
    for k, r in enumerate(results):
        ax.hist(r["red_lens"], bins=bins, color=colors[k], alpha=0.30,
                histtype="step", density=True, lw=1.2)
        ax.hist(r["fnd_lens"], bins=bins, color=colors[k], alpha=0.80,
                histtype="step", density=True, ls="--", lw=1.0)
    # Indicate which line type is which pool
    ax.plot([], [], color="#666", lw=1.5, label="redundant")
    ax.plot([], [], color="#666", lw=1.0, ls="--", label="foundational")
    ax.set_xlabel("streamline length (mm)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    # Annotate cross-subject means
    red_means = [r["red_lens"].mean() for r in results]
    fnd_means = [r["fnd_lens"].mean() for r in results]
    ax.set_title(f"Length: redundant {np.mean(red_means):.0f}±{np.std(red_means):.0f}mm  "
                  f"vs foundational {np.mean(fnd_means):.0f}±{np.std(fnd_means):.0f}mm",
                  fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 3: distance to nearest foundational, one histogram per subject overlaid
    ax = axes[2]
    bins = np.linspace(0, 25, 60)
    for k, r in enumerate(results):
        ax.hist(r["d_min"], bins=bins, color=colors[k], alpha=0.55,
                histtype="step", lw=1.2, label=f"{r['subj']} (med {np.median(r['d_min']):.1f})")
    ax.set_xlabel("MDF to nearest foundational (mm)", fontsize=9)
    ax.set_ylabel("# redundant streamlines", fontsize=9)
    means = [r["d_min"].mean() for r in results]
    ax.set_title(f"Distance to nearest foundational: {np.mean(means):.1f}±{np.std(means):.1f}mm",
                  fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.99, top=0.90, bottom=0.13, wspace=0.30)
    out_path = os.path.join(OUT_DIR, "redundant_pool_all_subjects.png")
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"\nsaved {out_path}")

    # Persist summary JSON-ish numbers for later use (small)
    summary_path = os.path.join(OUT_DIR, "redundant_pool_all_subjects_summary.json")
    summary = []
    for r in results:
        summary.append({
            "subj": r["subj"],
            "n_redundant": r["n_redundant"],
            "n_foundational": r["n_foundational"],
            "unclass_pct": r["unclass_pct"],
            "supwma_pct": r["supwma_pct"],
            "redundant_len_mean": float(r["red_lens"].mean()),
            "redundant_len_median": float(np.median(r["red_lens"])),
            "foundational_len_mean": float(r["fnd_lens"].mean()),
            "foundational_len_median": float(np.median(r["fnd_lens"])),
            "mdf_mean": float(r["d_min"].mean()),
            "mdf_median": float(np.median(r["d_min"])),
            "mdf_p25": float(np.percentile(r["d_min"], 25)),
            "mdf_p75": float(np.percentile(r["d_min"], 75)),
            "top_bundles": r["top_bundles"],
        })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
