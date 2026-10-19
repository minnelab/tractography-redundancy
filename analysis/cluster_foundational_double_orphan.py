"""
QuickBundles cluster the foundational streamlines that are unclassified by
BOTH atlases (TractSeg AND SUP-WMA) — the genuine "atlas-orphan" pool.

Differs from cluster_foundational_unclassified.py, which keeps every
foundational streamline that TractSeg did not classify (regardless of SWM
status). Here we also require the SUP-WMA label to be "all_others" (the
SUP-WMA reject class), so the input is the residual that no atlas labels.

Pipeline per subject:
  1. Load consensus weights (commit_acc, sift_acc), TS labels, SWM labels.
  2. Mask = (commit_acc==1.0) & (sift_acc==1.0)
           & (TS bundle == 'unclassified')
           & (SWM cluster == 'all_others')
  3. Pull those streamlines from the .trk, resample to 20 pts, QuickBundles
     with MDF (20 mm), classify cluster midpoints by anatomical region.
"""

import functools
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import length, set_number_of_points

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_triple_no import assign_region  # same region buckets used elsewhere

print = functools.partial(print, flush=True)  # noqa: A001

ROOT = os.environ.get("TRACTO_ROOT", "/home/maia-user/tracto_work")
INTERSECTION_DIR = f"{ROOT}/Data/Intersection_Labels/all_subjects"
SWM_DIR = f"{ROOT}/Data/SupWMA"
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

QB_THRESHOLD = 20.0
NUM_POINTS = 20
OUT_CSV = f"{ROOT}/Data/Intersection_Labels/foundational_double_orphan_clusters.csv"
SUBJ_SUMMARY_CSV = f"{ROOT}/Data/Intersection_Labels/foundational_double_orphan_subject_summary.csv"


def main():
    all_clusters = []
    subj_summary = []
    for subj in TRK_PATHS:
        print(f"\n{'='*60}\nSubject: {subj}\n{'='*60}")
        t0 = time.time()

        il = pd.read_csv(os.path.join(INTERSECTION_DIR, f"{subj}.csv"),
                          usecols=["commit_acc", "sift_acc"])
        bc = pd.read_csv(BUNDLE_CLASS_PATHS[subj])
        sw = pd.read_csv(os.path.join(SWM_DIR, f"{subj}_supwma_labels.csv"))

        n_total = len(il)
        if not (len(bc) == len(sw) == n_total):
            raise ValueError(
                f"{subj}: row mismatch il={n_total} bc={len(bc)} sw={len(sw)}"
            )

        foundational = (il["commit_acc"].values >= 1.0) & (il["sift_acc"].values >= 1.0)
        ts_orphan    = bc["bundle"].values == "unclassified"
        swm_orphan   = sw["supwma_cluster_name"].values == "all_others"
        mask = foundational & ts_orphan & swm_orphan
        idx  = np.where(mask)[0]

        n_F = int(foundational.sum())
        n_F_ts_orphan = int((foundational & ts_orphan).sum())
        n_double = int(mask.sum())
        print(f"  Foundational total:              {n_F:>9,}")
        print(f"  & TS-orphan:                     {n_F_ts_orphan:>9,}  "
              f"({100*n_F_ts_orphan/max(n_F,1):.1f}%)")
        print(f"  & TS-orphan & SWM-orphan:        {n_double:>9,}  "
              f"({100*n_double/max(n_F,1):.1f}% of F)")

        # Free big inputs before loading the tractogram
        del il, bc, sw, foundational, ts_orphan, swm_orphan

        print(f"  Loading tractogram: {TRK_PATHS[subj]}")
        sft = load_tractogram(TRK_PATHS[subj], "same", bbox_valid_check=False)
        sls = sft.streamlines
        target = [sls[int(i)] for i in idx]
        del sft, sls
        print(f"  Extracted {len(target):,} streamlines")

        # Geometry
        lengths = np.array(list(length(target))) if target else np.array([])
        if target:
            print(f"  Length: mean={lengths.mean():.1f}mm  median={np.median(lengths):.1f}mm")

        # Region of each streamline's midpoint
        if target:
            mids = np.stack([np.asarray(s, dtype=np.float32)[len(s) // 2]
                              for s in target])
            regions = assign_region(mids)
            reg_counts = Counter(regions)
            print(f"  Midpoint regions: "
                  + ", ".join(f"{r} {100*c/len(target):.1f}%"
                              for r, c in reg_counts.most_common()))
        else:
            reg_counts = Counter()

        # Cluster
        if len(target) < 2:
            print("  Too few streamlines to cluster; skipping QuickBundles.")
            clusters = []
        else:
            print(f"  Resampling to {NUM_POINTS} points and running QuickBundles "
                  f"(MDF, threshold {QB_THRESHOLD} mm)...")
            resampled = set_number_of_points(target, NUM_POINTS)
            qb = QuickBundles(threshold=QB_THRESHOLD,
                              metric=AveragePointwiseEuclideanMetric())
            clusters = qb.cluster(resampled)
            sizes = [len(c) for c in clusters]
            print(f"  Found {len(clusters)} clusters; max={max(sizes)}, "
                  f"median={int(np.median(sizes))}, mean={np.mean(sizes):.1f}")
            # Top 20 cluster summary
            ranked = sorted(enumerate(clusters), key=lambda x: -len(x[1]))[:20]
            for rank, (cid, cl) in enumerate(ranked):
                c_idx = np.asarray(cl.indices)
                c_lens = lengths[c_idx]
                centroid = np.asarray(cl.centroid)
                mid = centroid[len(centroid) // 2]
                region = assign_region(mid[None, :])[0]
                all_clusters.append({
                    "subject": subj,
                    "cluster_rank": rank + 1,
                    "cluster_id": cid,
                    "n_streamlines": int(len(cl)),
                    "pct_of_double_orphan": 100 * len(cl) / max(len(target), 1),
                    "mean_length_mm": float(c_lens.mean()),
                    "std_length_mm": float(c_lens.std()),
                    "region": region,
                    "centroid_start_x": float(centroid[0, 0]),
                    "centroid_start_y": float(centroid[0, 1]),
                    "centroid_start_z": float(centroid[0, 2]),
                    "centroid_end_x": float(centroid[-1, 0]),
                    "centroid_end_y": float(centroid[-1, 1]),
                    "centroid_end_z": float(centroid[-1, 2]),
                    "centroid_mid_x": float(mid[0]),
                    "centroid_mid_y": float(mid[1]),
                    "centroid_mid_z": float(mid[2]),
                })

        subj_summary.append({
            "subject": subj,
            "n_foundational": n_F,
            "n_foundational_TS_orphan": n_F_ts_orphan,
            "n_foundational_double_orphan": n_double,
            "pct_F_TS_orphan": 100 * n_F_ts_orphan / max(n_F, 1),
            "pct_F_double_orphan": 100 * n_double / max(n_F, 1),
            "mean_length_mm": float(lengths.mean()) if target else float("nan"),
            "median_length_mm": float(np.median(lengths)) if target else float("nan"),
            "n_clusters": int(len(clusters)),
            "max_cluster_size": int(max([len(c) for c in clusters], default=0)),
            "pct_cerebellum": 100 * reg_counts.get("cerebellum", 0) / max(len(target), 1),
            "pct_occipital":  100 * reg_counts.get("occipital", 0) / max(len(target), 1),
            "pct_frontal":    100 * reg_counts.get("frontal", 0) / max(len(target), 1),
            "pct_temporal":   100 * reg_counts.get("temporal", 0) / max(len(target), 1),
            "pct_parietal":   100 * reg_counts.get("parietal", 0) / max(len(target), 1),
            "pct_other":      100 * reg_counts.get("other", 0) / max(len(target), 1),
        })

        print(f"  Subject done in {time.time() - t0:.1f}s")
        del target

    pd.DataFrame(all_clusters).to_csv(OUT_CSV, index=False)
    print(f"\nSaved per-cluster summary -> {OUT_CSV}")
    pd.DataFrame(subj_summary).to_csv(SUBJ_SUMMARY_CSV, index=False)
    print(f"Saved per-subject summary -> {SUBJ_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
