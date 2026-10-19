"""
Plot the top QuickBundles clusters of foundational-unclassified streamlines.

Produces:
  - Per-subject: 3D renderings from axial, sagittal, coronal views (top N clusters color-coded)
  - Cross-subject comparison panel
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import set_number_of_points

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

QB_THRESHOLD = 20.0
NUM_POINTS = 20
TOP_N = 15  # number of top clusters to color-code
MAX_STREAMLINES_PER_CLUSTER = 200  # subsample for plotting speed

OUT_DIR = "Data/Intersection_Labels/cluster_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# Distinct colors for top clusters
COLORS = plt.cm.tab20(np.linspace(0, 1, 20))


def subsample_streamlines(streamlines, max_n):
    """Randomly subsample streamlines for plotting."""
    if len(streamlines) <= max_n:
        return streamlines
    idx = np.random.choice(len(streamlines), max_n, replace=False)
    return [streamlines[i] for i in idx]


def plot_streamlines_2d(ax, streamlines, color, alpha=0.15, linewidth=0.3,
                        proj_axes=(0, 1)):
    """Plot streamlines as 2D line collections on given axes."""
    x_idx, y_idx = proj_axes
    segments = []
    for sl in streamlines:
        points = np.column_stack([sl[:, x_idx], sl[:, y_idx]])
        segments.append(points)
    lc = LineCollection(segments, colors=[(*color[:3], alpha)] * len(segments),
                        linewidths=linewidth)
    ax.add_collection(lc)


def make_subject_plot(subj, target_streamlines, clusters, sorted_cluster_indices):
    """Create a 3-view plot for one subject showing top clusters."""
    views = [
        ("Axial (top)", 0, 1),       # x vs y
        ("Sagittal (left)", 1, 2),    # y vs z
        ("Coronal (front)", 0, 2),    # x vs z
    ]
    axis_labels = [
        ("L-R (x)", "A-P (y)"),
        ("A-P (y)", "I-S (z)"),
        ("L-R (x)", "I-S (z)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(f"Subject {subj} — Top {TOP_N} Foundational-Unclassified Clusters\n"
                 f"(QuickBundles, threshold={QB_THRESHOLD}mm, {len(clusters)} total clusters)",
                 fontsize=14, fontweight="bold")

    for ax_i, (view_name, xi, yi) in enumerate(views):
        ax = axes[ax_i]

        # Plot remaining clusters in gray first
        other_indices = set(range(len(clusters))) - set(sorted_cluster_indices[:TOP_N])
        for ci in other_indices:
            sl_list = [target_streamlines[i] for i in clusters[ci].indices]
            sl_sub = subsample_streamlines(sl_list, 50)
            plot_streamlines_2d(ax, sl_sub, color=(0.7, 0.7, 0.7), alpha=0.05,
                                linewidth=0.2, proj_axes=(xi, yi))

        # Plot top clusters with colors
        for rank, ci in enumerate(sorted_cluster_indices[:TOP_N]):
            sl_list = [target_streamlines[i] for i in clusters[ci].indices]
            sl_sub = subsample_streamlines(sl_list, MAX_STREAMLINES_PER_CLUSTER)
            plot_streamlines_2d(ax, sl_sub, color=COLORS[rank], alpha=0.3,
                                linewidth=0.5, proj_axes=(xi, yi))

        ax.set_title(view_name, fontsize=12)
        ax.set_xlabel(axis_labels[ax_i][0])
        ax.set_ylabel(axis_labels[ax_i][1])
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid(True, alpha=0.2)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for rank, ci in enumerate(sorted_cluster_indices[:TOP_N]):
        n = len(clusters[ci])
        legend_elements.append(
            Line2D([0], [0], color=COLORS[rank], linewidth=2,
                   label=f"#{rank+1} (n={n:,})")
        )
    legend_elements.append(
        Line2D([0], [0], color=(0.7, 0.7, 0.7), linewidth=2, label="Other clusters")
    )
    fig.legend(handles=legend_elements, loc="center right",
               bbox_to_anchor=(0.99, 0.5), fontsize=8, framealpha=0.9)
    plt.subplots_adjust(right=0.88, wspace=0.3)

    out_path = os.path.join(OUT_DIR, f"{subj}_top{TOP_N}_clusters.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


def make_individual_cluster_plots(subj, target_streamlines, clusters,
                                  sorted_cluster_indices):
    """Create individual plots for each of the top clusters."""
    fig, axes = plt.subplots(TOP_N, 3, figsize=(18, 3.5 * TOP_N))
    fig.suptitle(f"Subject {subj} — Individual Cluster Detail\n"
                 f"(Top {TOP_N} clusters by size)",
                 fontsize=14, fontweight="bold", y=1.001)

    views = [(0, 1), (1, 2), (0, 2)]
    view_names = ["Axial", "Sagittal", "Coronal"]

    for rank, ci in enumerate(sorted_cluster_indices[:TOP_N]):
        sl_list = [target_streamlines[i] for i in clusters[ci].indices]
        sl_sub = subsample_streamlines(sl_list, MAX_STREAMLINES_PER_CLUSTER)
        n = len(clusters[ci])

        for v_i, (xi, yi) in enumerate(views):
            ax = axes[rank, v_i]
            plot_streamlines_2d(ax, sl_sub, color=COLORS[rank], alpha=0.25,
                                linewidth=0.6, proj_axes=(xi, yi))

            # Plot centroid thicker
            centroid = clusters[ci].centroid
            ax.plot(centroid[:, xi], centroid[:, yi], color="black",
                    linewidth=2, alpha=0.8)

            ax.set_aspect("equal")
            ax.autoscale_view()
            ax.grid(True, alpha=0.2)

            if rank == 0:
                ax.set_title(view_names[v_i], fontsize=11)
            if v_i == 0:
                ax.set_ylabel(f"#{rank+1}\nn={n:,}", fontsize=9, fontweight="bold")

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    out_path = os.path.join(OUT_DIR, f"{subj}_individual_clusters.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ============================================================
# Main loop
# ============================================================
all_cluster_data = {}  # store for cross-subject plot

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
    print(f"  Foundational-unclassified: {len(indices):,}")

    # Load tractogram
    print(f"  Loading tractogram...")
    sft = load_tractogram(TRK_PATHS[subj], "same", bbox_valid_check=False)
    all_streamlines = sft.streamlines
    target_streamlines = [all_streamlines[i] for i in indices]
    del sft, all_streamlines

    # Resample and cluster
    print(f"  Clustering...")
    resampled = set_number_of_points(target_streamlines, NUM_POINTS)
    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=QB_THRESHOLD, metric=metric)
    clusters = qb.cluster(resampled)

    # Sort clusters by size
    sorted_indices = sorted(range(len(clusters)), key=lambda i: -len(clusters[i]))

    print(f"  {len(clusters)} clusters found, plotting...")

    # Generate plots
    make_subject_plot(subj, target_streamlines, clusters, sorted_indices)
    make_individual_cluster_plots(subj, target_streamlines, clusters, sorted_indices)

    # Store centroid data for cross-subject plot
    all_cluster_data[subj] = {
        "centroids": [clusters[i].centroid for i in sorted_indices[:TOP_N]],
        "sizes": [len(clusters[i]) for i in sorted_indices[:TOP_N]],
        "total": len(target_streamlines),
    }

    del target_streamlines, resampled, clusters

# ============================================================
# Cross-subject comparison: overlay centroids
# ============================================================
print(f"\nGenerating cross-subject centroid comparison...")
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
fig.suptitle(f"Cross-Subject Comparison — Top {TOP_N} Cluster Centroids\n"
             f"(All 5 subjects overlaid)", fontsize=14, fontweight="bold")

views = [(0, 1), (1, 2), (0, 2)]
view_names = ["Axial (top)", "Sagittal (left)", "Coronal (front)"]
axis_labels = [("L-R (x)", "A-P (y)"), ("A-P (y)", "I-S (z)"), ("L-R (x)", "I-S (z)")]

subject_colors = plt.cm.Set1(np.linspace(0, 0.7, 5))

for v_i, (xi, yi) in enumerate(views):
    ax = axes[v_i]
    for s_i, subj in enumerate(all_cluster_data):
        data = all_cluster_data[subj]
        for c_i, centroid in enumerate(data["centroids"]):
            lw = 1.0 + 2.0 * data["sizes"][c_i] / data["total"] * 20
            alpha = 0.4 + 0.4 * min(data["sizes"][c_i] / max(data["sizes"]), 1.0)
            ax.plot(centroid[:, xi], centroid[:, yi],
                    color=subject_colors[s_i], linewidth=lw, alpha=alpha)

    ax.set_title(view_names[v_i], fontsize=12)
    ax.set_xlabel(axis_labels[v_i][0])
    ax.set_ylabel(axis_labels[v_i][1])
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.grid(True, alpha=0.2)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=subject_colors[i], linewidth=2,
           label=subj)
    for i, subj in enumerate(all_cluster_data)
]
fig.legend(handles=legend_elements, loc="center right",
           bbox_to_anchor=(0.99, 0.5), fontsize=10)
plt.subplots_adjust(right=0.90, wspace=0.3)

out_path = os.path.join(OUT_DIR, "cross_subject_centroids.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out_path}")

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
