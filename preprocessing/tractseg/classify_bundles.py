"""
Classify streamlines using pyAFQ with TractSeg bundle + endpoint masks.

Uses pyAFQ's recognize function with BundleDict to classify streamlines
against TractSeg bundle segmentations and endings segmentations.
Each streamline must pass through all three ROIs (main mask + both endpoints)
to be classified into a bundle.

Usage:
    python classify_bundles.py \
        --tractogram <path.trk> \
        --tractseg_dir <tractseg_output> \
        --reference <reference.nii.gz> \
        [--output <output.csv>]

Example:
    python classify_bundles.py \
        --tractogram Data/Tractogram_Data/TEST/877168.trk \
        --tractseg_dir Data/Structural_data_HCP/877168/T1w/tractseg_output \
        --reference Data/Structural_data_HCP/877168/T1w/tractseg_output/peaks.nii.gz
"""

import argparse
import os
import numpy as np
import nibabel as nib
import pandas as pd
from glob import glob
from dipy.io.streamline import load_tractogram
from AFQ.recognition.recognize import recognize
from AFQ.api.bundle_dict import BundleDict


class IdentityMapping:
    """Identity mapping for when ROIs are already in subject space."""
    def transform(self, data, **kwargs):
        return data

    def transform_inverse(self, data, **kwargs):
        return data


def build_bundle_dict(tractseg_dir):
    """Build a BundleDict from TractSeg bundle + endings segmentations."""
    bundle_seg_dir = os.path.join(tractseg_dir, "bundle_segmentations")
    endings_dir = os.path.join(tractseg_dir, "endings_segmentations")

    if not os.path.isdir(bundle_seg_dir):
        raise FileNotFoundError(f"bundle_segmentations not found at {bundle_seg_dir}")
    if not os.path.isdir(endings_dir):
        raise FileNotFoundError(f"endings_segmentations not found at {endings_dir}")

    mask_files = sorted(glob(os.path.join(bundle_seg_dir, "*.nii.gz")))

    # Skip the broad "CC" mask — use CC_1 through CC_7 instead
    skip = {"CC"}

    bundle_defs = {}
    for mf in mask_files:
        name = os.path.basename(mf).replace(".nii.gz", "")
        if name in skip:
            continue

        ending_b = os.path.join(endings_dir, f"{name}_b.nii.gz")
        ending_e = os.path.join(endings_dir, f"{name}_e.nii.gz")

        if not os.path.exists(ending_b) or not os.path.exists(ending_e):
            print(f"  Warning: endings masks missing for {name}, skipping")
            continue

        bundle_defs[name] = {
            "include": [mf, ending_b, ending_e],
            "space": "subject",
        }

    print(f"Defined {len(bundle_defs)} bundles (excluding broad CC)")
    return BundleDict(bundle_defs)


def classify_streamlines(tractogram_path, tractseg_dir, reference_path,
                         output_path=None, save_intermediates_dir=None):
    # Build bundle definitions
    bundle_dict = build_bundle_dict(tractseg_dir)

    # Load reference image
    img = nib.load(reference_path)

    # Load tractogram with reference
    print(f"Loading tractogram: {tractogram_path}")
    tg = load_tractogram(tractogram_path, reference_path, bbox_valid_check=False)
    n_streamlines = len(tg.streamlines)
    print(f"Loaded {n_streamlines} streamlines")

    # Set up intermediates directory for crash recovery
    if save_intermediates_dir is None:
        save_intermediates_dir = os.path.join(
            os.path.dirname(output_path or tractogram_path),
            "classify_intermediates"
        )
    os.makedirs(save_intermediates_dir, exist_ok=True)
    print(f"Saving intermediates to: {save_intermediates_dir}")

    # Use identity mapping since TractSeg masks are already in subject space
    mapping = IdentityMapping()

    # Run pyAFQ recognition
    print("Running pyAFQ bundle recognition (this may take a while)...")
    fiber_groups, meta = recognize(
        tg,
        img,
        mapping,
        bundle_dict,
        reg_template=img,
        return_idx=True,
        filter_by_endpoints=True,
        dist_to_atlas=4,
        save_intermediates=save_intermediates_dir,
    )

    # Build classification from indices
    bundle_col = np.full(n_streamlines, "unclassified", dtype=object)

    print("\nBundle classification results:")
    for bundle_name, fg in fiber_groups.items():
        indices = fg["idx"]
        if len(indices) == 0:
            continue
        indices = indices.astype(int)
        print(f"  {bundle_name}: {len(indices)} streamlines")
        bundle_col[indices] = bundle_name

    # Build output DataFrame
    df = pd.DataFrame({
        "streamline_idx": np.arange(n_streamlines),
        "bundle": bundle_col,
    })

    if output_path is None:
        output_path = os.path.splitext(tractogram_path)[0] + "_bundle_classifications.csv"

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\nSummary:")
    summary = df["bundle"].value_counts()
    print(summary.to_string())
    n_classified = (df["bundle"] != "unclassified").sum()
    print(f"\nTotal streamlines: {n_streamlines}")
    print(f"Classified: {n_classified} ({n_classified / n_streamlines * 100:.1f}%)")
    print(f"Unclassified: {n_streamlines - n_classified} ({(n_streamlines - n_classified) / n_streamlines * 100:.1f}%)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify tractogram streamlines using pyAFQ + TractSeg masks"
    )
    parser.add_argument(
        "--tractogram", required=True,
        help="Path to tractogram file (.trk or .tck)",
    )
    parser.add_argument(
        "--tractseg_dir", required=True,
        help="Path to TractSeg output directory",
    )
    parser.add_argument(
        "--reference", required=True,
        help="Reference NIfTI image for coordinate space (e.g., peaks.nii.gz)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output CSV path",
    )
    parser.add_argument(
        "--save_intermediates", "-s",
        help="Directory to save intermediate results for crash recovery",
    )
    args = parser.parse_args()
    classify_streamlines(args.tractogram, args.tractseg_dir, args.reference,
                         args.output, args.save_intermediates)
