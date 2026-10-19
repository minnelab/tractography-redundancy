"""
Run SUP-WMA on a single subject's full tractogram.

Pipeline:
  1. Load .trk
  2. Convert to .vtp (preserves order)
  3. extract_tract_feat.py  → 15-point RAS feature matrix (H5)
  4. test_TwoStage.py       → per-streamline cluster label (H5)
  5. Save labels CSV aligned with original streamline indices

Note: streamlines are passed in T1w-acpc world coordinates. Smoke test (10k random
streamlines on subj 665254) gave 17.9% SWM coverage with anatomically sensible
short-range geometry (mean length 60.7 mm vs 97.2 mm for "all_others"), so we
proceed without applying the HCP nonlinear MNI warp.
"""

import argparse
import os
import subprocess
import sys
import time

import h5py
import nibabel as nib
import numpy as np
import vtk
import pandas as pd

SUPWMA_DIR = os.environ.get(
    "SUPWMA_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "external", "SupWMA"),
)
PY = "/opt/conda/envs/supwma/bin/python"


def trk_to_vtp(trk_path: str, vtp_path: str):
    """Convert TRK to VTP preserving streamline order. Uses vectorized vtk numpy interface."""
    from vtk.util import numpy_support as nps
    print(f"[trk_to_vtp] loading {trk_path}")
    t0 = time.time()
    trk = nib.streamlines.load(trk_path)
    streams = trk.streamlines
    n = len(streams)
    print(f"[trk_to_vtp] loaded {n:,} streamlines in {time.time()-t0:.1f}s")

    t0 = time.time()
    # Concatenate all streamline points into single array
    all_pts = np.concatenate([np.asarray(s, dtype=np.float32) for s in streams], axis=0)
    print(f"[trk_to_vtp] concatenated {all_pts.shape[0]:,} points in {time.time()-t0:.1f}s")

    t0 = time.time()
    # Build connectivity array: [n_pts_in_stream, pt_idx_0, pt_idx_1, ...] repeated
    lengths = np.array([len(s) for s in streams], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    # vtkCellArray "Lines" format: for each cell, [count, id0, id1, ...]
    # Total length = sum(lengths) + n_cells
    connectivity = np.empty(int(lengths.sum()) + n, dtype=np.int64)
    idx = 0
    for i in range(n):
        L = int(lengths[i])
        connectivity[idx] = L
        connectivity[idx + 1: idx + 1 + L] = np.arange(int(starts[i]), int(starts[i]) + L, dtype=np.int64)
        idx += L + 1
    print(f"[trk_to_vtp] built connectivity in {time.time()-t0:.1f}s")

    t0 = time.time()
    points = vtk.vtkPoints()
    points.SetData(nps.numpy_to_vtk(all_pts, deep=True, array_type=vtk.VTK_FLOAT))

    cells = vtk.vtkCellArray()
    id_arr = nps.numpy_to_vtkIdTypeArray(connectivity, deep=True)
    cells.SetCells(int(n), id_arr)

    pd_ = vtk.vtkPolyData()
    pd_.SetPoints(points)
    pd_.SetLines(cells)
    print(f"[trk_to_vtp] assembled polydata in {time.time()-t0:.1f}s "
          f"(streamlines={pd_.GetNumberOfLines()}, points={pd_.GetNumberOfPoints()})")

    t0 = time.time()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(vtp_path)
    writer.SetInputData(pd_)
    # Use appended binary for speed/size
    writer.SetDataModeToAppended()
    writer.EncodeAppendedDataOff()  # raw binary, smaller and faster
    writer.Write()
    print(f"[trk_to_vtp] wrote {vtp_path} in {time.time()-t0:.1f}s")
    return n


def extract_features(vtp_path: str, out_dir: str, prefix: str):
    print(f"[extract_features] {vtp_path}")
    t0 = time.time()
    cmd = [PY, os.path.join(SUPWMA_DIR, "extract_tract_feat.py"),
           vtp_path, out_dir,
           "-outPrefix", prefix,
           "-feature", "RAS",
           "-numPoints", "15"]
    subprocess.run(cmd, check=True, cwd=SUPWMA_DIR)
    print(f"[extract_features] done in {time.time()-t0:.1f}s")


def run_inference(feat_h5: str, out_dir: str, prefix: str):
    print(f"[run_inference] {feat_h5}")
    t0 = time.time()
    cmd = [PY, os.path.join(SUPWMA_DIR, "test_TwoStage.py"),
           "--weight_path", os.path.join(SUPWMA_DIR, "TrainedModel_TwoStage"),
           "--feat_path", feat_h5,
           "--out_path", out_dir,
           "--label_names", os.path.join(SUPWMA_DIR, "TrainedModel_TwoStage", "label_names.h5"),
           "--out_prefix", prefix]
    subprocess.run(cmd, check=True, cwd=SUPWMA_DIR)
    print(f"[run_inference] done in {time.time()-t0:.1f}s")


def save_labels_csv(pred_h5: str, label_names_h5: str, out_csv: str):
    print(f"[save_labels_csv] {pred_h5}")
    with h5py.File(pred_h5, "r") as f:
        preds = np.array(f["complete_pred_test"], dtype=np.int32)
    with h5py.File(label_names_h5, "r") as f:
        key = list(f.keys())[0]
        names = [x.decode() if isinstance(x, bytes) else x for x in f[key][()]]

    # Map class index → cluster name; index 198 == "all_others"
    cluster_names = [names[i] if i < len(names) else f"idx_{i}" for i in preds]
    rejected = (preds == 198).astype(np.int8)
    df = pd.DataFrame({
        "streamline_idx": np.arange(len(preds), dtype=np.int64),
        "supwma_cluster_idx": preds,
        "supwma_cluster_name": cluster_names,
        "rejected": rejected,
    })
    df.to_csv(out_csv, index=False)
    n_rej = rejected.sum()
    print(f"[save_labels_csv] wrote {out_csv}")
    print(f"  total: {len(preds):,}")
    print(f"  rejected (all_others): {n_rej:,} ({100*n_rej/len(preds):.1f}%)")
    print(f"  SWM-classified: {len(preds)-n_rej:,} ({100*(len(preds)-n_rej)/len(preds):.1f}%)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subject", required=True)
    p.add_argument("--trk", required=True)
    p.add_argument("--work_dir", default="/home/maia-user/tracto_work/Data/SupWMA")
    p.add_argument("--keep_vtp", action="store_true", help="keep intermediate VTP")
    p.add_argument("--keep_h5", action="store_true", help="keep intermediate feature/prediction H5s")
    args = p.parse_args()

    subj_dir = os.path.join(args.work_dir, args.subject)
    os.makedirs(subj_dir, exist_ok=True)
    prefix = args.subject

    vtp_path = os.path.join(subj_dir, f"{prefix}.vtp")
    feat_h5 = os.path.join(subj_dir, f"{prefix}_featMatrix.h5")
    pred_h5 = os.path.join(subj_dir, f"{prefix}_test_prediction_mask.h5")
    out_csv = os.path.join(args.work_dir, f"{args.subject}_supwma_labels.csv")

    t_start = time.time()
    # Sanity-check existing intermediates by file size; non-empty must exceed a floor.
    def _stale(path, min_bytes):
        if not os.path.exists(path):
            return False
        if os.path.getsize(path) < min_bytes:
            os.remove(path)
            print(f"[reset] removed stale {path}")
            return False
        return True

    if not _stale(vtp_path, 1024 * 1024):  # need >1 MB for any real tractogram
        trk_to_vtp(args.trk, vtp_path)
    else:
        print(f"[skip] VTP exists: {vtp_path}")

    if not _stale(feat_h5, 1024):
        extract_features(vtp_path, subj_dir, prefix)
    else:
        print(f"[skip] features exist: {feat_h5}")

    if not _stale(pred_h5, 1024):
        run_inference(feat_h5, subj_dir, prefix)
    else:
        print(f"[skip] predictions exist: {pred_h5}")

    label_names_h5 = os.path.join(SUPWMA_DIR, "TrainedModel_TwoStage", "label_names.h5")
    save_labels_csv(pred_h5, label_names_h5, out_csv)

    # Clean up intermediates if requested
    if not args.keep_vtp and os.path.exists(vtp_path):
        os.remove(vtp_path); print(f"[clean] removed {vtp_path}")
    if not args.keep_h5:
        for fp in (feat_h5, pred_h5):
            if os.path.exists(fp):
                os.remove(fp); print(f"[clean] removed {fp}")

    print(f"\n[done] subject {args.subject} in {time.time()-t_start:.1f}s → {out_csv}")


if __name__ == "__main__":
    main()
