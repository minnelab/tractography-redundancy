"""
Spatial map of redundancy — for subject 837964, render where in the brain the
filter-redundant streamlines accumulate vs where foundational streamlines live.

Definitions used here (threshold t = 0.9):
  foundational := SIFT $\\geq t$ AND COMMIT $\\geq t$
  redundant    := $0 < $ SIFT $< t$ AND $0 < $ COMMIT $< t$  (partially accepted but
                  did not clear foundational)
  noise        := SIFT $== 0$ OR COMMIT $== 0$  (at least one filter rejected outright)

For each 2D bin (sagittal y-z and axial x-y projections) we compute the density
of each pool from a 500k-streamline subsample (15 equispaced points each).
The redundancy fraction per bin is shown as a heatmap.
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection

ROOT = os.environ.get("TRACTO_ROOT", "/home/maia-user/tracto_work")
SUBJ = "837964"
TRK_PATH = f"{ROOT}/Data/Tractogram_Data/TEST/{SUBJ}.trk"
BUNDLE_CSV = f"{ROOT}/Data/Tractogram_Data/TEST/{SUBJ}_bundle_classifications.csv"
INTERSECTION_CSV = f"{ROOT}/Data/Intersection_Labels/all_subjects/{SUBJ}.csv"
SWM_CSV = f"{ROOT}/Data/SupWMA/{SUBJ}_supwma_labels.csv"
OUT_DIR = f"{ROOT}/Streamlines-redundancy/figures"

EXAMPLE_BUNDLE = "FX_left"    # iconic C-shape, clean foundational core + redundant halo
BUNDLE_VIEW = (1, 2)          # sagittal view of the bundle

SAMPLE_N = 500_000
POINTS_PER_STREAM = 15
GRID = 140  # spatial bins per axis


def sample_points_for(streams, idx, n_points=POINTS_PER_STREAM):
    """For each selected streamline, return n_points equispaced 3D coords."""
    out = np.empty((len(idx), n_points, 3), dtype=np.float32)
    for i, ix in enumerate(idx):
        s = np.asarray(streams[int(ix)], dtype=np.float32)
        if s.shape[0] >= n_points:
            ts = np.linspace(0, s.shape[0] - 1, n_points).astype(int)
            out[i] = s[ts]
        else:
            ts = np.linspace(0, s.shape[0] - 1, n_points)
            for k, t in enumerate(ts):
                f = t - int(t)
                lo, hi = int(t), min(int(t) + 1, s.shape[0] - 1)
                out[i, k] = (1 - f) * s[lo] + f * s[hi]
    return out


def heatmap(pts, xi, yi, x_edges, y_edges):
    return np.histogram2d(pts[..., xi].ravel(), pts[..., yi].ravel(),
                          bins=[x_edges, y_edges])[0]


def render_streamline_lines(ax, streamlines, xi, yi, color, alpha=0.55, lw=0.45):
    if len(streamlines) == 0:
        return
    seg_list = []
    for s in streamlines:
        s = np.asarray(s)
        if s.shape[0] < 2:
            continue
        seg = np.stack([s[:-1, [xi, yi]], s[1:, [xi, yi]]], axis=1)
        seg_list.append(seg)
    if not seg_list:
        return
    segments = np.concatenate(seg_list, axis=0)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=lw, alpha=alpha))


def render_streamline_lines_grad(ax, streamlines, xi, yi, weights, cmap, vmin,
                                  vmax, alpha=0.55, lw=0.40):
    """Like render_streamline_lines but with per-streamline colors mapped from
    a scalar weight through a colormap. All segments of a streamline share
    the streamline's color. Alpha is constant across the collection."""
    if len(streamlines) == 0:
        return
    norm = Normalize(vmin=vmin, vmax=vmax)
    seg_list = []
    color_list = []
    for s, w in zip(streamlines, weights):
        s = np.asarray(s)
        if s.shape[0] < 2:
            continue
        rgba = list(cmap(norm(float(w))))
        rgba[3] = alpha
        seg = np.stack([s[:-1, [xi, yi]], s[1:, [xi, yi]]], axis=1)
        seg_list.append(seg)
        color_list.append(np.tile(rgba, (seg.shape[0], 1)))
    if not seg_list:
        return
    segments = np.concatenate(seg_list, axis=0)
    cols = np.concatenate(color_list, axis=0)
    ax.add_collection(LineCollection(segments, colors=cols, linewidths=lw))


# Gradient colormap for redundant streamlines (mean acceptance weight a in
# (0, 0.9]). Dim dark red at low a -> bright orange/yellow at high a, so
# against the black panel background near-foundational streamlines pop and
# barely-accepted ones fade into the background.
R_GRAD_CMAP = LinearSegmentedColormap.from_list(
    "r_grad",
    [(0.00, "#2a0707"), (0.25, "#7a1010"), (0.55, "#d6321d"),
     (0.80, "#ff7a2a"), (1.00, "#ffd060")],
)


def render_directional_rgb(ax, streamlines, xi, yi, alpha=0.5, lw=0.4):
    """Color each segment by |dx|,|dy|,|dz| of its local orientation (RGB)."""
    if not streamlines:
        return
    seg_list, color_list = [], []
    for s in streamlines:
        s = np.asarray(s)
        if s.shape[0] < 2:
            continue
        diffs = s[1:] - s[:-1]
        norms = np.linalg.norm(diffs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = np.abs(diffs / norms)
        seg_2d = np.stack([s[:-1, [xi, yi]], s[1:, [xi, yi]]], axis=1)
        seg_list.append(seg_2d)
        rgba = np.concatenate([unit, np.full((unit.shape[0], 1), alpha)], axis=1)
        color_list.append(rgba)
    if not seg_list:
        return
    segments = np.concatenate(seg_list, axis=0)
    colors = np.concatenate(color_list, axis=0)
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=lw))


def main():
    rng = np.random.default_rng(0)
    print("Loading intersection labels...")
    il = pd.read_csv(INTERSECTION_CSV)
    bc = pd.read_csv(BUNDLE_CSV)
    sw = pd.read_csv(SWM_CSV)
    bundle = bc["bundle"].values
    swm    = sw["supwma_cluster_name"].values
    sa = il["sift_acc"].values
    ca = il["commit_acc"].values
    n_total = len(sa)
    print(f"  {n_total:,} streamlines")

    print("Loading tractogram (~100s)...")
    trk = nib.streamlines.load(TRK_PATH)
    streams = trk.streamlines

    idx_sample = rng.choice(n_total, size=min(SAMPLE_N, n_total), replace=False)
    idx_sample.sort()

    print(f"Sampling {len(idx_sample):,} streamlines, {POINTS_PER_STREAM} points each...")
    pts = sample_points_for(streams, idx_sample)  # (N, 15, 3)
    # Per-streamline labels
    sa_s = sa[idx_sample]
    ca_s = ca[idx_sample]
    T = 0.9
    foundational = (sa_s >= T) & (ca_s >= T)
    redundant   = (sa_s > 0) & (sa_s < T) & (ca_s > 0) & (ca_s < T)
    noise       = (sa_s == 0) | (ca_s == 0)
    print(f"  pool sizes  foundational: {foundational.sum():,}   "
          f"redundant (0<both<{T}): {redundant.sum():,}   "
          f"noise (zero in either): {noise.sum():,}")

    # Compute viewports from sample
    all_pts = pts.reshape(-1, 3)
    x_min, x_max = all_pts[:, 0].min() - 2, all_pts[:, 0].max() + 2
    y_min, y_max = all_pts[:, 1].min() - 2, all_pts[:, 1].max() + 2
    z_min, z_max = all_pts[:, 2].min() - 2, all_pts[:, 2].max() + 2

    x_edges = np.linspace(x_min, x_max, GRID + 1)
    y_edges = np.linspace(y_min, y_max, GRID + 1)
    z_edges = np.linspace(z_min, z_max, GRID + 1)

    # 1x3 layout: two whole-brain density panels + one combined bundle panel
    bx, by = BUNDLE_VIEW
    fig, axes_row = plt.subplots(1, 3, figsize=(10.5, 3.5))
    axes = axes_row[np.newaxis, :]  # keep 2D indexing convention

    # ============== Row 1: whole-brain density panels with colorbars ==============
    xi, yi = 1, 2  # sagittal y-z
    xe, ye = y_edges, z_edges
    H_fnd = heatmap(pts[foundational], xi, yi, xe, ye)
    H_red = heatmap(pts[redundant],    xi, yi, xe, ye)

    im1 = axes[0, 0].imshow(H_fnd.T, origin="lower",
                             extent=[xe[0], xe[-1], ye[0], ye[-1]],
                             cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, H_fnd.max())))
    axes[0, 0].set_title("Whole brain: foundational density", fontsize=9)
    cbar1 = fig.colorbar(im1, ax=axes[0, 0], fraction=0.030, pad=0.02)
    cbar1.ax.tick_params(labelsize=7)
    cbar1.set_label("streamline points per bin", fontsize=7)

    im2 = axes[0, 1].imshow(H_red.T, origin="lower",
                             extent=[xe[0], xe[-1], ye[0], ye[-1]],
                             cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, H_red.max())))
    axes[0, 1].set_title(f"Whole brain: redundant density ($0<$ both $<{T}$)", fontsize=9)
    cbar2 = fig.colorbar(im2, ax=axes[0, 1], fraction=0.030, pad=0.02)
    cbar2.ax.tick_params(labelsize=7)
    cbar2.set_label("streamline points per bin", fontsize=7)

    # ============== Panel (0,2): bundle combined view ==============
    # All bundle streamlines colored by acceptance state: foundational (blue),
    # redundant (orange), and zero-acceptance noise (faint gray). Shows the
    # tractogram structure AND the per-streamline redundancy state in one panel.
    bundle_mask = bundle == EXAMPLE_BUNDLE
    b_all_idx = np.where(bundle_mask)[0]
    b_fnd_idx = np.where(bundle_mask & (sa >= T) & (ca >= T))[0]
    b_red_idx = np.where(bundle_mask & (sa > 0) & (sa < T) & (ca > 0) & (ca < T))[0]
    b_noise_idx = np.where(bundle_mask & ((sa == 0) | (ca == 0)))[0]
    b_all_n = int(bundle_mask.sum())
    print(f"  {EXAMPLE_BUNDLE}: n_all={b_all_n}, foundational={len(b_fnd_idx)}, "
          f"redundant={len(b_red_idx)}, noise={len(b_noise_idx)}")

    b_all_streams = [streams[int(i)] for i in b_all_idx]
    if b_all_streams:
        pts_b = np.concatenate(b_all_streams, axis=0)
        x_lim = (pts_b[:, bx].min() - 4, pts_b[:, bx].max() + 4)
        y_lim = (pts_b[:, by].min() - 4, pts_b[:, by].max() + 4)
    else:
        x_lim = y_lim = (-100, 100)

    # Render redundant (orange, background) then foundational (blue, on top).
    render_streamline_lines(axes[0, 2], [streams[int(i)] for i in b_red_idx],
                            bx, by, color="#dd8452", alpha=0.55, lw=0.60)
    render_streamline_lines(axes[0, 2], [streams[int(i)] for i in b_fnd_idx],
                            bx, by, color="#1f3a8a", alpha=0.95, lw=0.80)
    axes[0, 2].set_xlim(*x_lim); axes[0, 2].set_ylim(*y_lim)
    axes[0, 2].plot([], [], color="#1f3a8a", lw=2,
                    label=f"foundational ($n={len(b_fnd_idx)}$)")
    axes[0, 2].plot([], [], color="#dd8452", lw=2,
                    label=f"redundant ($n={len(b_red_idx)}$)")
    axes[0, 2].legend(fontsize=7, frameon=False, loc="lower right")
    axes[0, 2].set_title(f"{EXAMPLE_BUNDLE.replace('_',' ')} ($n={b_all_n}$)",
                          fontsize=9)

    for ax in axes.flat:
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bbb"); spine.set_linewidth(0.6)

    fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.02,
                        wspace=0.18, hspace=0.10)
    out_path = os.path.join(OUT_DIR, "redundancy_location_map.png")
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out_path}")

    # ============== Alternative: split bundle column (vertical) ==============
    # FX sub-panels stacked vertically (foundational on top, redundant below),
    # both rendered as directional-RGB tractograms (no overlap).
    from matplotlib.gridspec import GridSpec
    fig2 = plt.figure(figsize=(10.5, 3.6))
    gs = GridSpec(2, 3, figure=fig2,
                  width_ratios=[1.0, 1.0, 0.55],
                  hspace=0.18, wspace=0.20)
    ax_a = fig2.add_subplot(gs[:, 0])   # whole-brain foundational, full height
    ax_b = fig2.add_subplot(gs[:, 1])   # whole-brain redundant, full height
    ax_c = fig2.add_subplot(gs[0, 2])   # FX foundational
    ax_d = fig2.add_subplot(gs[1, 2])   # FX redundant

    im1b = ax_a.imshow(H_fnd.T, origin="lower",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, H_fnd.max())))
    ax_a.set_title("Foundational density", fontsize=9)
    cbar1b = fig2.colorbar(im1b, ax=ax_a, fraction=0.030, pad=0.02)
    cbar1b.ax.tick_params(labelsize=7)
    cbar1b.set_label("pts / bin", fontsize=7)

    im2b = ax_b.imshow(H_red.T, origin="lower",
                       extent=[xe[0], xe[-1], ye[0], ye[-1]],
                       cmap="magma", norm=LogNorm(vmin=1, vmax=max(2, H_red.max())))
    ax_b.set_title("Redundant density", fontsize=9)
    cbar2b = fig2.colorbar(im2b, ax=ax_b, fraction=0.030, pad=0.02)
    cbar2b.ax.tick_params(labelsize=7)
    cbar2b.set_label("pts / bin", fontsize=7)

    # FX foundational (directional RGB)
    render_directional_rgb(ax_c, [streams[int(i)] for i in b_fnd_idx],
                            bx, by, alpha=0.85, lw=0.6)
    ax_c.set_xlim(*x_lim); ax_c.set_ylim(*y_lim)
    ax_c.set_title(f"FX-L found. ({len(b_fnd_idx)})", fontsize=8)

    # FX redundant (directional RGB)
    render_directional_rgb(ax_d, [streams[int(i)] for i in b_red_idx],
                            bx, by, alpha=0.50, lw=0.4)
    ax_d.set_xlim(*x_lim); ax_d.set_ylim(*y_lim)
    ax_d.set_title(f"FX-L redund. ({len(b_red_idx)})", fontsize=8)

    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bbb"); spine.set_linewidth(0.6)

    fig2.subplots_adjust(left=0.04, right=0.97, top=0.92, bottom=0.02)
    alt_path = os.path.join(OUT_DIR, "redundancy_location_map_split.png")
    fig2.savefig(alt_path, dpi=200, facecolor="white")
    plt.close(fig2)
    print(f"saved {alt_path}")

    # ============== Axial-plane companion figure ==============
    # Same two whole-brain pools (foundational, redundant) but viewed from
    # above (x-y projection) so the abstract carries both sagittal and axial
    # context. Kept short (2 panels, no FX) so it fits under the sagittal
    # image without pushing the abstract off a single page.
    axi_xi, axi_yi = 0, 1  # axial (x-y) projection
    xe_ax, ye_ax = x_edges, y_edges
    H_fnd_ax = heatmap(pts[foundational], axi_xi, axi_yi, xe_ax, ye_ax)
    H_red_ax = heatmap(pts[redundant],    axi_xi, axi_yi, xe_ax, ye_ax)

    # The axial brain shape is taller than it is wide (y span > x span), so a
    # 10.5"-wide figure leaves huge whitespace around each panel at equal
    # aspect. Size the figure to actual content: 2 narrow panels touching +
    # a thin shared colorbar, fitting a strip ~3.6" wide.
    fig3 = plt.figure(figsize=(3.6, 1.8))
    gs3 = fig3.add_gridspec(1, 2, wspace=0.01,
                            left=0.02, right=0.80, top=0.98, bottom=0.02)
    ax_axf = fig3.add_subplot(gs3[0, 0])
    ax_axr = fig3.add_subplot(gs3[0, 1])

    # Shared log-scale norm so both panels share a single colorbar.
    vmax_shared = max(2, H_fnd_ax.max(), H_red_ax.max())
    norm_shared = LogNorm(vmin=1, vmax=vmax_shared)

    ax_axf.imshow(H_fnd_ax.T, origin="lower",
                  extent=[xe_ax[0], xe_ax[-1], ye_ax[0], ye_ax[-1]],
                  cmap="magma", norm=norm_shared)
    im_axr = ax_axr.imshow(H_red_ax.T, origin="lower",
                           extent=[xe_ax[0], xe_ax[-1], ye_ax[0], ye_ax[-1]],
                           cmap="magma", norm=norm_shared)

    cax = fig3.add_axes([0.82, 0.06, 0.025, 0.88])
    cbar = fig3.colorbar(im_axr, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("pts / bin", fontsize=7)

    for ax in (ax_axf, ax_axr):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bbb"); spine.set_linewidth(0.6)
    ax_path = os.path.join(OUT_DIR, "redundancy_location_map_axial.png")
    fig3.savefig(ax_path, dpi=200, facecolor="white")
    plt.close(fig3)
    print(f"saved {ax_path}")

    # ============== Two side-by-side panel sets (F | R) ==============
    # Per pool: axial + FX-bundle on top row, axial zoom on the bottom row.
    # The zoom window is centered on the F density peak in (x, y), and the
    # same window is reused for R for parallel comparison. A rectangle on the
    # full axial panel highlights the zoom region and a thin arrow points
    # from it down to the zoom panel.
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Rectangle
    from matplotlib.patches import ConnectionPatch

    ax_aspect = (x_max - x_min) / (y_max - y_min)        # ~0.7
    fx_aspect = (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0])

    # --- F-density peak in axial (x, y). ---
    xc_a = 0.5 * (xe_ax[1:] + xe_ax[:-1])
    yc_a = 0.5 * (ye_ax[1:] + ye_ax[:-1])
    XG, YG = np.meshgrid(xc_a, yc_a, indexing="ij")
    W = H_fnd_ax.astype(np.float64)
    if W.sum() > 0:
        xp = float((XG * W).sum() / W.sum())
        yp = float((YG * W).sum() / W.sum())
    else:
        xp, yp = 0.0, 0.0
    # Manual zoom centre — pick a cortical region with abundant U-fibres
    # rather than the F-density centroid (which lands in the limbic core).
    # Right superior parieto-occipital cortex, where short-range association
    # fibres connecting adjacent gyri are prominent.
    # Left frontal U-fibre belt — coronal slab around the precentral/SFG
    # region, where short-range cortico-cortical fibres are dense and
    # foundational density (after F=1 filter) reliably sits above background.
    # Per-unit (bundle or SWM cluster) bounding box on the axial map.
    # Uses all points of all member streamlines so the box surrounds the
    # full shape (e.g. CST's descending stem) rather than only the
    # midpoint cluster.
    def axial_bbox(name, which="bundle", pad=3.0, hemi=None):
        labels = bundle if which == "bundle" else swm
        m = labels == name
        if not m.any():
            return None
        all_idx = np.where(m)[0]
        if hemi in ("left", "right"):
            mids = np.array([np.asarray(streams[int(i)])[len(streams[int(i)]) // 2, 0]
                              for i in all_idx])
            keep = (mids < 0) if hemi == "left" else (mids > 0)
            all_idx = all_idx[keep]
        if len(all_idx) == 0:
            return None
        pts_b = np.concatenate(
            [np.asarray(streams[int(i)]) for i in all_idx], axis=0)
        return (pts_b[:, 0].min() - pad, pts_b[:, 0].max() + pad,
                pts_b[:, 1].min() - pad, pts_b[:, 1].max() + pad)

    # Both panels show TractSeg bundles — TS labels are properly lateralised.
    # F-side: FX-L (top TS essential, ~5% F rate, classic limbic bundle).
    # R-side: CC_3 (small CC sub-segment, ~3.9k streamlines, near-straight
    # horizontal sweep in axial; high R rate like the other CC bundles).
    BUNDLE_F      = "FX_left"
    BUNDLE_R      = "CC_3"
    BUNDLE_F_KIND = "bundle"
    BUNDLE_R_KIND = "bundle"
    BUNDLE_F_HEMI = None
    BUNDLE_R_HEMI = None
    LABEL_F       = "FX-L (TS)"
    LABEL_R       = "CC_3 (TS)"
    fx_lo_x,  fx_hi_x,  fx_lo_y,  fx_hi_y  = axial_bbox(
        BUNDLE_F, BUNDLE_F_KIND, hemi=BUNDLE_F_HEMI)
    cst_lo_x, cst_hi_x, cst_lo_y, cst_hi_y = axial_bbox(
        BUNDLE_R, BUNDLE_R_KIND, hemi=BUNDLE_R_HEMI)
    # Pick the row-1 aspect by union of the two bundle bboxes so both panels
    # share the same height after aspect="equal".
    bw_max = max(fx_hi_x - fx_lo_x, cst_hi_x - cst_lo_x)
    bh_max = max(fx_hi_y - fx_lo_y, cst_hi_y - cst_lo_y)
    bundle_aspect = bw_max / bh_max
    print(f"  bundle F bbox: x[{fx_lo_x:.0f},{fx_hi_x:.0f}] y[{fx_lo_y:.0f},{fx_hi_y:.0f}]")
    print(f"  bundle R bbox: x[{cst_lo_x:.0f},{cst_hi_x:.0f}] y[{cst_lo_y:.0f},{cst_hi_y:.0f}]")

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("black")
    PANEL_BG = "black"

    # Smooth the heatmaps for display. A small Gaussian (sigma ~1 bin) blurs
    # the per-streamline shot noise enough that you see anatomy, not bin
    # artefacts. Done once per panel; full-brain and zoom share the same
    # smoothed array (the zoom just sets xlim/ylim).
    # Redundant pool has ~30x more streamlines per bin than foundational,
    # so it doesn't need as much smoothing to look clean; smoothing it as
    # heavily blurs out anatomical structure that's actually well-resolved.
    try:
        from scipy.ndimage import gaussian_filter
        SIG_F = 0.3
        SIG_R = 0.0   # no blur — R has dense sampling, smoothing washes anatomy
        H_fnd_s    = gaussian_filter(H_fnd.astype(np.float64),    sigma=SIG_F)
        H_red_s    = H_red.astype(np.float64) if SIG_R == 0 else \
                     gaussian_filter(H_red.astype(np.float64),    sigma=SIG_R)
        H_fnd_ax_s = gaussian_filter(H_fnd_ax.astype(np.float64), sigma=SIG_F)
        H_red_ax_s = H_red_ax.astype(np.float64) if SIG_R == 0 else \
                     gaussian_filter(H_red_ax.astype(np.float64), sigma=SIG_R)
    except ImportError:
        H_fnd_s, H_red_s, H_fnd_ax_s, H_red_ax_s = H_fnd, H_red, H_fnd_ax, H_red_ax

    # Old triad layout (restored): top has axial + FX directional-RGB +
    # cbar per side; bottom has one wide slot per side. Wider aspect makes
    # the bottom panels less tall, which makes the whole figure more
    # landscape and uses horizontal space better when included in the PDF.
    BOTTOM_ASPECT = 2.4
    set_top_w  = ax_aspect + fx_aspect       # cell width of one bottom slot
    row0_h     = 1.0
    row1_h     = set_top_w / BOTTOM_ASPECT
    height_ratios = [row0_h, row1_h]

    cbar_w_rel = 0.08
    gap_w_rel  = 0.30
    width_ratios = [ax_aspect, fx_aspect, cbar_w_rel, gap_w_rel,
                    ax_aspect, fx_aspect, cbar_w_rel]

    fig_h_in = 4.0
    unit_in  = fig_h_in / (row0_h + row1_h + 0.04)
    fig_w_in = sum(width_ratios) * unit_in + 0.30
    fig4 = plt.figure(figsize=(fig_w_in, fig_h_in))
    # Leave room on the right for the colorbar tick labels + axis label so
    # neither "pts / bin (R)" gets clipped on save.
    gs4 = GridSpec(2, 7, figure=fig4,
                   width_ratios=width_ratios,
                   height_ratios=height_ratios,
                   hspace=0.10, wspace=0.04,
                   left=0.02, right=0.92, top=0.98, bottom=0.02)

    # F triad: axial + FX-L directional RGB + cbar + bottom bundle slot
    ax_fax_full = fig4.add_subplot(gs4[0, 0])
    ax_ffx      = fig4.add_subplot(gs4[0, 1])
    cax_f       = fig4.add_subplot(gs4[:, 2])
    ax_fbund    = fig4.add_subplot(gs4[1, 0:2])

    # R triad
    ax_rax_full = fig4.add_subplot(gs4[0, 4])
    ax_rfx      = fig4.add_subplot(gs4[0, 5])
    cax_r       = fig4.add_subplot(gs4[:, 6])
    ax_rbund    = fig4.add_subplot(gs4[1, 4:6])

    norm_f = LogNorm(vmin=1, vmax=max(2, H_fnd.max(), H_fnd_ax.max()))
    norm_r = LogNorm(vmin=1, vmax=max(2, H_red.max(), H_red_ax.max()))

    def draw_axial_with_box(ax, H, norm, box, interp="bilinear"):
        ax.imshow(H.T, origin="lower",
                  extent=[xe_ax[0], xe_ax[-1], ye_ax[0], ye_ax[-1]],
                  cmap=cmap, norm=norm, interpolation=interp)
        lo_x, hi_x, lo_y, hi_y = box
        ax.add_patch(Rectangle((lo_x, lo_y), hi_x - lo_x, hi_y - lo_y,
                                linewidth=0.8, edgecolor="#00d4aa",
                                facecolor="none"))
        return ax

    # Axial-projected streamline overlay for the zoom panel: subsample by
    # zoom-box membership, cap count, project to (x, y) and draw as flat
    # white lines so they read as streamlines on top of any colormap.
    def overlay_streamlines(ax, mask, max_n, alpha, lw, color="white"):
        sample = pts[mask]                                    # (N, 15, 3)
        if sample.size == 0:
            return
        x = sample[:, :, 0]
        y = sample[:, :, 1]
        in_box = ((x >= x_lo) & (x <= x_hi) &
                  (y >= y_lo_a) & (y <= y_hi_a)).any(axis=1)
        sample = sample[in_box]
        if sample.shape[0] > max_n:
            ix = rng.choice(sample.shape[0], size=max_n, replace=False)
            sample = sample[ix]
        if sample.shape[0] == 0:
            return
        seg_xy = np.stack([sample[:, :-1, [0, 1]],
                           sample[:, 1:,  [0, 1]]], axis=2)
        seg_xy = seg_xy.reshape(-1, 2, 2)
        ax.add_collection(LineCollection(seg_xy, colors=color,
                                         linewidths=lw, alpha=alpha))

    # Top row: full axial densities with a green box marking the bundle
    # location below. F-side box marks FX-L, R-side box marks CC_3.
    fx_box  = (fx_lo_x,  fx_hi_x,  fx_lo_y,  fx_hi_y)
    cst_box = (cst_lo_x, cst_hi_x, cst_lo_y, cst_hi_y)
    draw_axial_with_box(ax_fax_full, H_fnd_ax_s, norm_f, fx_box)
    cbar_f = fig4.colorbar(plt.cm.ScalarMappable(norm=norm_f, cmap=cmap),
                            cax=cax_f)
    cbar_f.ax.tick_params(labelsize=10, colors="black")
    cbar_f.set_label("pts / bin (F)", fontsize=10)

    draw_axial_with_box(ax_rax_full, H_red_ax_s, norm_r, cst_box,
                        interp="nearest")
    cbar_r = fig4.colorbar(plt.cm.ScalarMappable(norm=norm_r, cmap=cmap),
                            cax=cax_r)
    cbar_r.ax.tick_params(labelsize=10, colors="black")
    cbar_r.set_label("pts / bin (R)", fontsize=10)

    # Top-right slot of each triad: FX-L directional-RGB (carried over from
    # the original triad design for spatial context above the bundle slot).
    render_directional_rgb(ax_ffx, [streams[int(i)] for i in b_fnd_idx],
                            bx, by, alpha=0.9, lw=0.6)
    ax_ffx.set_xlim(*x_lim); ax_ffx.set_ylim(*y_lim)
    render_directional_rgb(ax_rfx, [streams[int(i)] for i in b_red_idx],
                            bx, by, alpha=0.55, lw=0.45)
    ax_rfx.set_xlim(*x_lim); ax_rfx.set_ylim(*y_lim)


    # ----- Bottom row: the bundle itself, F=blue / R=gradient by a --------
    def bundle_panel(ax, name, box, label, label_pos="top", kind="bundle",
                     hemi=None, show_cbar=False):
        labels = bundle if kind == "bundle" else swm
        mask = labels == name
        if not mask.any():
            return
        # Apply hemisphere filter on the streamline midpoint x-coordinate.
        if hemi in ("left", "right"):
            mid_x = np.array([
                np.asarray(streams[int(i)])[len(streams[int(i)]) // 2, 0]
                for i in np.where(mask)[0]
            ])
            keep = (mid_x < 0) if hemi == "left" else (mid_x > 0)
            kept_idx = np.where(mask)[0][keep]
            mask = np.zeros_like(mask)
            mask[kept_idx] = True
        # F threshold loosened to a >= 0.9 (was strict a = 1.0) so that
        # near-foundational streamlines also light up green in the bundle.
        # Matches the Methods definition (F: a>=t, R: 0<a<t) used everywhere
        # else in the code so the boundary at a=t is treated consistently.
        fnd = np.where(mask & (sa >= T) & (ca >= T))[0]
        red = np.where(mask & (sa > 0) & (sa < T) & (ca > 0) & (ca < T))[0]
        # Per-streamline mean acceptance weight in (0, T) for the R pool.
        # Sort low->high so high-a streamlines render on top and the bright
        # near-foundational ones are not buried under dim, barely-accepted ones.
        r_w = 0.5 * (sa[red] + ca[red])
        order = np.argsort(r_w)
        red_sorted = red[order]
        r_w_sorted = r_w[order]
        # Axial projection (x, y) so it lines up with the green box above.
        render_streamline_lines_grad(
            ax, [streams[int(i)] for i in red_sorted], 0, 1,
            r_w_sorted, R_GRAD_CMAP, 0.0, T, alpha=0.55, lw=0.40,
        )
        # Foundational color: CVD-safe greenish-teal — clearly distinct from
        # the red/orange R gradient for protan/deutan viewers.
        render_streamline_lines(ax, [streams[int(i)] for i in fnd],
                                0, 1, color="#00d4aa", alpha=0.95, lw=0.60)
        lo_x, hi_x, lo_y, hi_y = box
        ax.set_xlim(lo_x, hi_x); ax.set_ylim(lo_y, hi_y)
        ly, va = (0.96, "top") if label_pos == "top" else (0.04, "bottom")
        ax.text(0.03, ly,
                f"{label} F:{len(fnd)} R:{len(red)}",
                transform=ax.transAxes, va=va, ha="left",
                fontsize=10, color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc="black",
                          ec="none", alpha=0.55))
        if show_cbar:
            # Inset colorbar inside the bottom-right of the bundle panel so
            # the dim->bright red gradient is readable without leaning solely
            # on the caption. Drawn as an inset_axes in axes coordinates.
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            cax_in = inset_axes(ax, width="40%", height="7%",
                                loc="upper left",
                                bbox_to_anchor=(0.0, -0.02, 1, 1),
                                bbox_transform=ax.transAxes,
                                borderpad=0.6)
            cb = plt.colorbar(
                plt.cm.ScalarMappable(norm=Normalize(0.0, T),
                                       cmap=R_GRAD_CMAP),
                cax=cax_in, orientation="horizontal",
            )
            cb.set_ticks([0.0, T])
            cb.set_ticklabels(["0", f"{T:g}"])
            cb.ax.tick_params(labelsize=9, colors="white", pad=1, length=2)
            cb.outline.set_edgecolor("white")
            cb.outline.set_linewidth(0.4)
            cax_in.set_title("R streamline $a$", fontsize=9, color="white",
                              pad=2)
        for spine in ax.spines.values():
            spine.set_color("#00d4aa"); spine.set_linewidth(0.8)

    # Both bottom slots share an axial bbox sized to the bottom-slot aspect.
    # Find the union x,y extent of the two bundles, then expand y so the
    # rendered box matches BOTTOM_ASPECT exactly.
    def bbox_to_aspect(box, target_aspect):
        lo_x, hi_x, lo_y, hi_y = box
        w = hi_x - lo_x
        h = hi_y - lo_y
        desired_h = w / target_aspect
        if desired_h > h:
            extra = (desired_h - h) / 2
            lo_y -= extra; hi_y += extra
        else:
            extra = (h * target_aspect - w) / 2
            lo_x -= extra; hi_x += extra
        return (lo_x, hi_x, lo_y, hi_y)

    fx_box_panel  = bbox_to_aspect(fx_box,  BOTTOM_ASPECT)
    cst_box_panel = bbox_to_aspect(cst_box, BOTTOM_ASPECT)

    bundle_panel(ax_fbund, BUNDLE_F, fx_box_panel,
                 LABEL_F, label_pos="top",    kind=BUNDLE_F_KIND,
                 hemi=BUNDLE_F_HEMI)
    bundle_panel(ax_rbund, BUNDLE_R, cst_box_panel,
                 LABEL_R, label_pos="bottom", kind=BUNDLE_R_KIND,
                 hemi=BUNDLE_R_HEMI, show_cbar=True)

    # Arrows from each axial's green box down to the bundle panel.
    for ax_full, ax_bund, box in [(ax_fax_full, ax_fbund, fx_box),
                                   (ax_rax_full, ax_rbund, cst_box)]:
        xc = 0.5 * (box[0] + box[1])
        con = ConnectionPatch(
            xyA=(xc, box[2]), coordsA=ax_full.transData,
            xyB=(xc, box[3]), coordsB=ax_bund.transData,
            arrowstyle="-|>", mutation_scale=8,
            color="#00d4aa", lw=0.8,
        )
        fig4.add_artist(con)

    bundle_axes = (ax_fbund, ax_rbund)
    for ax in (ax_fax_full, ax_ffx, ax_fbund, ax_rax_full, ax_rfx, ax_rbund):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(PANEL_BG)
        if ax in bundle_axes:
            continue  # keep the green spines applied in bundle_panel
        for spine in ax.spines.values():
            spine.set_color("#444"); spine.set_linewidth(0.6)

    triad_path = os.path.join(OUT_DIR, "redundancy_location_map_triads.png")
    # bbox_inches="tight" guarantees that the colorbar label on the very
    # right edge ("pts / bin (R)") is not clipped by the figure crop.
    fig4.savefig(triad_path, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig4)
    print(f"saved {triad_path}")


if __name__ == "__main__":
    main()
