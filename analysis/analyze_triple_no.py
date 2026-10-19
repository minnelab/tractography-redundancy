"""
Length + location breakdown of streamlines stratified by:
  - Atlas status: (TractSeg-yes/no) × (SUP-WMA-yes/no) — 4 cells
  - Filter status: foundational vs neither vs single-accepted

Headline question: what do the "triple-no" streamlines look like?
(unclassified by both atlases AND filter-redundant at the 1.0 threshold)
Are they short-range fiber orphans? Long-range crossing-fiber overgeneration?
Cerebellum residual? Each pattern has a different implication for tractography.

Outputs:
  - Data/SupWMA/analysis/triple_no_breakdown.csv (per subject × cell stats)
  - Data/SupWMA/analysis/triple_no_summary.csv (subject-averaged)
"""

import os
import time

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.environ.get("TRACTO_ROOT", "/home/maia-user/tracto_work")
INTERSECTION_DIR = f"{ROOT}/Data/Intersection_Labels/all_subjects"
SUPWMA_DIR = f"{ROOT}/Data/SupWMA"

TRK_PATHS = {
    "665254": f"{ROOT}/Data/Tractogram_Data/TEST/665254.trk",
    "734045": f"{ROOT}/Data/Tractogram_Data/TRAIN/734045.trk",
    "784565": f"{ROOT}/Data/Tractogram_Data/VAL/784565.trk",
    "837964": f"{ROOT}/Data/Tractogram_Data/TEST/837964.trk",
    "877168": f"{ROOT}/Data/Tractogram_Data/TEST/877168.trk",
}
BUNDLE_CLASS_PATHS = {
    "665254": f"{ROOT}/Data/Tractogram_Data/TEST/665254_bundle_classifications.csv",
    "734045": f"{ROOT}/Data/Tractogram_Data/TRAIN/734045_bundle_classifications.csv",
    "784565": f"{ROOT}/Data/Tractogram_Data/VAL/784565_bundle_classifications.csv",
    "837964": f"{ROOT}/Data/Tractogram_Data/TEST/837964_bundle_classifications.csv",
    "877168": f"{ROOT}/Data/Tractogram_Data/TEST/877168_bundle_classifications.csv",
}


def midpoint_and_length(streamlines) -> tuple[np.ndarray, np.ndarray]:
    """Compute midpoint coord and arc-length for every streamline (vectorized).

    streamlines is a nibabel ArraySequence: data shape (sum_pts, 3), with
    _offsets giving each streamline's start row and _lengths its point count.
    """
    data = np.asarray(streamlines.get_data(), dtype=np.float32)
    offsets = np.asarray(streamlines._offsets, dtype=np.int64)
    lengths_pts = np.asarray(streamlines._lengths, dtype=np.int64)
    n = len(offsets)

    # Midpoint = middle indexed point per streamline
    mids = data[offsets + lengths_pts // 2]

    # Arc length: compute every consecutive-point distance once, then sum per
    # streamline. The boundary segment (last point of stream i → first point
    # of stream i+1) must be zeroed so reduceat doesn't include it.
    d = data[1:] - data[:-1]
    seg_len = np.sqrt((d * d).sum(axis=1))
    if n > 1:
        seg_len[offsets[1:] - 1] = 0.0
    lens = np.add.reduceat(seg_len, offsets).astype(np.float32)
    return mids, lens


def assign_region(mid: np.ndarray) -> np.ndarray:
    """Coordinate-based region classification of streamline midpoints.

    Uses MNI-like axes (RAS+): x = L-R, y = P-A, z = I-S.
    Boundaries are coarse — meant to bucket into the same 6 classes the paper
    uses for top-cluster centroids in Table 4 (occipital, frontal, cerebellum,
    parietal, temporal, other).
    """
    x = mid[:, 0]
    y = mid[:, 1]
    z = mid[:, 2]
    n = len(mid)
    region = np.full(n, "other", dtype=object)

    # Cerebellum / brainstem: very inferior
    is_cereb = z < -20
    # Occipital: very posterior, not inferior
    is_occip = (~is_cereb) & (y < -50)
    # Frontal: anterior
    is_front = (~is_cereb) & (y > 25)
    # Temporal: lateral, mid-z (rough)
    is_temp = (~is_cereb) & (~is_occip) & (~is_front) & (np.abs(x) > 35) & (z < 15)
    # Parietal: high z, mid y
    is_pari = (~is_cereb) & (~is_occip) & (~is_front) & (~is_temp) & (z > 25)

    region[is_cereb] = "cerebellum"
    region[is_occip] = "occipital"
    region[is_front] = "frontal"
    region[is_temp] = "temporal"
    region[is_pari] = "parietal"
    return region


def stratify_subject(subj: str) -> pd.DataFrame:
    print(f"\n=== {subj} ===")
    t0 = time.time()
    trk = nib.streamlines.load(TRK_PATHS[subj])
    streams = trk.streamlines
    n_total = len(streams)
    print(f"  loaded {n_total:,} streamlines in {time.time()-t0:.1f}s")

    t0 = time.time()
    mids, lens = midpoint_and_length(streams)
    print(f"  midpoints+lengths in {time.time()-t0:.1f}s "
          f"(mean length {lens.mean():.1f}mm)")
    del trk, streams

    il = pd.read_csv(os.path.join(INTERSECTION_DIR, f"{subj}.csv"))
    bc = pd.read_csv(BUNDLE_CLASS_PATHS[subj])
    sw = pd.read_csv(os.path.join(SUPWMA_DIR, f"{subj}_supwma_labels.csv"))
    if not (len(il) == len(bc) == len(sw) == n_total):
        raise ValueError(f"row count mismatch for {subj}")

    commit = il["commit_acc"].values
    sift = il["sift_acc"].values
    ts_cls = (bc["bundle"].values != "unclassified")
    sw_cls = (sw["rejected"].values == 0)

    # Atlas status
    atlas = np.full(n_total, "ts_no_sw_no", dtype=object)
    atlas[ts_cls & sw_cls] = "ts_yes_sw_yes"
    atlas[ts_cls & ~sw_cls] = "ts_yes_sw_no"
    atlas[~ts_cls & sw_cls] = "ts_no_sw_yes"

    # Filter status at 1.0 threshold
    commit_yes = commit >= 1.0
    sift_yes = sift >= 1.0
    fil = np.full(n_total, "neither", dtype=object)
    fil[commit_yes & sift_yes] = "foundational"
    fil[commit_yes & ~sift_yes] = "commit_only"
    fil[~commit_yes & sift_yes] = "sift_only"

    region = assign_region(mids)

    # Build per-cell summary
    rows = []
    for atl in ["ts_yes_sw_yes", "ts_yes_sw_no", "ts_no_sw_yes", "ts_no_sw_no"]:
        for fl in ["foundational", "neither", "commit_only", "sift_only"]:
            mask = (atlas == atl) & (fil == fl)
            n = int(mask.sum())
            if n == 0:
                continue
            L = lens[mask]
            M = mids[mask]
            R = region[mask]
            region_counts = pd.Series(R).value_counts().to_dict()
            row = {
                "subject": subj,
                "atlas": atl,
                "filter": fl,
                "n": n,
                "pct_of_total": 100 * n / n_total,
                "mean_length": float(L.mean()),
                "median_length": float(np.median(L)),
                "p25_length": float(np.percentile(L, 25)),
                "p75_length": float(np.percentile(L, 75)),
                "p90_length": float(np.percentile(L, 90)),
                "mid_x_mean": float(M[:, 0].mean()),
                "mid_y_mean": float(M[:, 1].mean()),
                "mid_z_mean": float(M[:, 2].mean()),
                "pct_short_under50mm": float((L < 50).mean() * 100),
                "pct_long_over120mm": float((L > 120).mean() * 100),
            }
            # Add region percentages
            for r in ["cerebellum", "occipital", "frontal", "temporal",
                      "parietal", "other"]:
                row[f"pct_{r}"] = 100 * region_counts.get(r, 0) / n
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    out_dir = f"{SUPWMA_DIR}/analysis"
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for subj in TRK_PATHS:
        if not os.path.exists(os.path.join(SUPWMA_DIR, f"{subj}_supwma_labels.csv")):
            print(f"  {subj}: SUP-WMA labels missing, skipping")
            continue
        df = stratify_subject(subj)
        all_rows.append(df)

    if not all_rows:
        print("No subjects available.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(f"{out_dir}/triple_no_breakdown.csv", index=False)
    print(f"\nPer-subject breakdown → {out_dir}/triple_no_breakdown.csv")

    # Cross-subject summary
    grp = combined.groupby(["atlas", "filter"], as_index=False).agg({
        "n": "mean", "pct_of_total": "mean",
        "mean_length": "mean", "median_length": "mean",
        "p25_length": "mean", "p75_length": "mean", "p90_length": "mean",
        "mid_x_mean": "mean", "mid_y_mean": "mean", "mid_z_mean": "mean",
        "pct_short_under50mm": "mean", "pct_long_over120mm": "mean",
        "pct_cerebellum": "mean", "pct_occipital": "mean",
        "pct_frontal": "mean", "pct_temporal": "mean",
        "pct_parietal": "mean", "pct_other": "mean",
    })
    grp = grp.rename(columns={"n": "avg_n", "pct_of_total": "avg_pct_total"})
    grp.to_csv(f"{out_dir}/triple_no_summary.csv", index=False)

    print(f"\n=== Cross-subject summary ({len(all_rows)} subjects) ===\n")
    main_cols = ["atlas", "filter", "avg_n", "avg_pct_total",
                 "mean_length", "median_length",
                 "pct_short_under50mm", "pct_long_over120mm",
                 "pct_cerebellum", "pct_occipital", "pct_frontal",
                 "pct_temporal", "pct_parietal", "pct_other"]
    print(grp[main_cols].to_string(index=False))


if __name__ == "__main__":
    main()
