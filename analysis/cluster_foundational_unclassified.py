"""
Cluster foundational-unclassified streamlines using QuickBundles.

Foundational = commit_acc >= 1.0 AND sift_acc >= 1.0
Unclassified = not assigned to any TractSeg bundle

Uses Dipy's QuickBundles with MDF (Minimum Direct Flip) distance.
"""

import os
import numpy as np
import pandas as pd
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points, length
from collections import Counter

INTERSECTION_DIR = "Data/Intersection_Labels/all_subjects"
TRK_PATHS = {
    "665254": "Data/Tractogram_Data/TEST/665254.trk",
    "734045": "Data/Tractogram_Data/TRAIN/734045.trk",
    "784565": "Data/Tractogram_Data/VAL/784565.trk",
    "837964": "Data/Tractogram_Data/TEST/837964.trk",
    "877168": "Data/Tractogram_Data/TEST/877168.trk",
}
BUNDLE_CLASS_PATHS = {
    "665254": "Data/Tractogram_Data/TEST/665254_bundle_classifications.csv",
    "734045": "Data/Tractogram_Data/TRAIN/734045_bundle_classifications.csv",
    "784565": "Data/Tractogram_Data/VAL/784565_bundle_classifications.csv",
    "837964": "Data/Tractogram_Data/TEST/837964_bundle_classifications.csv",
    "877168": "Data/Tractogram_Data/TEST/877168_bundle_classifications.csv",
}

# QuickBundles threshold (mm) — controls cluster granularity
QB_THRESHOLD = 20.0  # moderate threshold
NUM_POINTS = 20  # resample streamlines for clustering

all_cluster_summaries = []

for subj in TRK_PATHS:
    print(f"\n{'='*60}")
    print(f"Subject: {subj}")
    print(f"{'='*60}")

    # Load labels
    il = pd.read_csv(os.path.join(INTERSECTION_DIR, f"{subj}.csv"))
    bc = pd.read_csv(BUNDLE_CLASS_PATHS[subj])

    foundational = (il["commit_acc"].values >= 1.0) & (il["sift_acc"].values >= 1.0)
    unclassified = bc["bundle"].values == "unclassified"
    mask = foundational & unclassified
    indices = np.where(mask)[0]
    print(f"Foundational-unclassified streamlines: {len(indices):,}")

    # Load tractogram
    print(f"Loading tractogram: {TRK_PATHS[subj]}")
    sft = load_tractogram(TRK_PATHS[subj], "same", bbox_valid_check=False)
    all_streamlines = sft.streamlines

    # Extract target streamlines
    print("Extracting foundational-unclassified streamlines...")
    target_streamlines = [all_streamlines[i] for i in indices]
    print(f"Extracted {len(target_streamlines):,} streamlines")

    # Free memory
    del sft, all_streamlines

    # Compute streamline lengths before resampling
    lengths = np.array(list(length(target_streamlines)))
    print(f"Length stats: mean={lengths.mean():.1f}mm, median={np.median(lengths):.1f}mm, "
          f"min={lengths.min():.1f}mm, max={lengths.max():.1f}mm")

    # Resample for clustering
    print(f"Resampling to {NUM_POINTS} points...")
    resampled = set_number_of_points(target_streamlines, NUM_POINTS)

    # Run QuickBundles
    print(f"Running QuickBundles (threshold={QB_THRESHOLD}mm)...")
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=QB_THRESHOLD, metric=metric)
    clusters = qb.cluster(resampled)

    n_clusters = len(clusters)
    sizes = [len(c) for c in clusters]
    print(f"\nFound {n_clusters} clusters")
    print(f"Cluster size stats: mean={np.mean(sizes):.1f}, median={np.median(sizes):.1f}, "
          f"max={max(sizes)}, min={min(sizes)}")

    # Analyze top clusters
    print(f"\nTop 20 clusters by size:")
    sorted_clusters = sorted(enumerate(clusters), key=lambda x: -len(x[1]))

    for rank, (cid, cluster) in enumerate(sorted_clusters[:20]):
        cluster_indices = cluster.indices
        cluster_lengths = lengths[cluster_indices]
        centroid = cluster.centroid

        # Centroid spatial info (start/end points)
        start = centroid[0]
        end = centroid[-1]
        mid = centroid[len(centroid)//2]

        cluster_info = {
            "subject": subj,
            "cluster_rank": rank + 1,
            "cluster_id": cid,
            "n_streamlines": len(cluster),
            "pct_of_total": len(cluster) / len(target_streamlines) * 100,
            "mean_length_mm": cluster_lengths.mean(),
            "std_length_mm": cluster_lengths.std(),
            "centroid_start_x": start[0], "centroid_start_y": start[1], "centroid_start_z": start[2],
            "centroid_end_x": end[0], "centroid_end_y": end[1], "centroid_end_z": end[2],
            "centroid_mid_x": mid[0], "centroid_mid_y": mid[1], "centroid_mid_z": mid[2],
        }
        all_cluster_summaries.append(cluster_info)

        print(f"  #{rank+1:2d}: {len(cluster):>6,} streamlines ({len(cluster)/len(target_streamlines)*100:5.1f}%) | "
              f"length={cluster_lengths.mean():.1f}±{cluster_lengths.std():.1f}mm | "
              f"start=({start[0]:.0f},{start[1]:.0f},{start[2]:.0f}) "
              f"end=({end[0]:.0f},{end[1]:.0f},{end[2]:.0f})")

    # Size distribution summary
    size_bins = [1, 5, 10, 50, 100, 500, 1000, 5000, float("inf")]
    bin_labels = ["1-4", "5-9", "10-49", "50-99", "100-499", "500-999", "1000-4999", "5000+"]
    hist, _ = np.histogram(sizes, bins=size_bins)
    print(f"\nCluster size distribution:")
    for label, count in zip(bin_labels, hist):
        if count > 0:
            print(f"  {label:>10s}: {count:>5d} clusters")

    # What % of streamlines are in the top N clusters?
    cumulative = np.cumsum(sorted(sizes, reverse=True))
    total = sum(sizes)
    for n in [5, 10, 20, 50]:
        if n <= len(sizes):
            print(f"  Top {n:>3d} clusters contain {cumulative[n-1]/total*100:.1f}% of streamlines")

    del target_streamlines, resampled, clusters

# Save cluster summaries
summary_df = pd.DataFrame(all_cluster_summaries)
out_path = "Data/Intersection_Labels/foundational_unclassified_clusters.csv"
summary_df.to_csv(out_path, index=False)
print(f"\nCluster summaries saved to {out_path}")
print("\nDone.")
