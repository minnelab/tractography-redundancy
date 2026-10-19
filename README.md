# Where is the tractogram redundancy?

Code for the IST abstract **"Where is the tractogram redundancy?"**
(Sanna Persson, Kurt G. Schilling, Rodrigo Moreno).

📄 **Abstract:** [`abstract/ist_abstract.pdf`](abstract/ist_abstract.pdf)

We characterize where redundancy lives in whole-brain tractograms at the level of
anatomical bundles. Per-streamline consensus acceptance weights from randomized SIFT
(rSIFT) and randomized COMMIT (rCOMMIT) are combined with anatomical labels from
TractSeg (71 long-range bundles) and SUP-WMA (198 superficial ORG-800FC clusters)
across five HCP subjects with 10 M-streamline IFOD2 tractograms each.

## Definitions

With per-streamline acceptance weights `a ∈ [0, 1]` from both filters and threshold `t = 0.9`:

| Pool | Definition |
|---|---|
| **Foundational (F)** | `a ≥ t` in both rSIFT and rCOMMIT |
| **Redundant (R)** (partial acceptance) | `0 < a < t` in both rSIFT and rCOMMIT |
| Noise | `a = 0` in at least one filter |

Requiring both filters to agree makes the labels robust to single-filter artifacts.

## Repository layout

```
abstract/                       IST abstract (PDF) and Fig. 1
preprocessing/
  acceptance_weights/           rCOMMIT runs over random subsets + rSIFT/rCOMMIT post-processing
  tractseg/                     TractSeg segmentation + pyAFQ-based bundle recognition (71 bundles)
  supwma/                       SUP-WMA superficial WM parcellation (198 ORG-800FC clusters)
external/
  TractSeg/                     submodule: MIC-DKFZ/TractSeg @ v2.10
  SupWMA/                       submodule: SlicerDMRI/SupWMA @ 353d2a2
analysis/                       Statistics behind the abstract's results and table
figures/                        Figure generation (Fig. 1 of the abstract and supporting plots)
```

### Pipeline

| Step | Script | Produces / used for |
|---|---|---|
| 1. Acceptance weights | `preprocessing/acceptance_weights/run_commit.py`, `rc_*.py`, `process_commit_results.py`, `process_sift_results.py` | Per-streamline `commit_acc` / `sift_acc` consensus weights (`Data/Intersection_Labels/all_subjects/<subj>.csv`) |
| 2. TractSeg labels | `preprocessing/tractseg/run_tractseg_all.sh`, `classify_bundles.py` | Per-streamline TS bundle (`<subj>_bundle_classifications.csv`) |
| 3. SUP-WMA labels | `preprocessing/supwma/run_supwma_subject.py` | Per-streamline SWM cluster (`Data/SupWMA/<subj>_supwma_labels.csv`) |
| 4. Per-bundle redundancy (TS) | `analysis/analyze_bundle_filtering_detailed.py`, `analysis/analyze_bundles_graded.py` | Inter-bundle variation, most foundational / redundant TS bundles, threshold sensitivity, ANOVA |
| 5. Per-cluster redundancy (SWM) | `analysis/analyze_supwma.py` | SWM coverage, per-cluster rates, ANOVA + Kruskal–Wallis (TS and SWM) |
| 6. Unclassified foundational streamlines | `analysis/cluster_foundational_unclassified.py`, `analysis/cluster_foundational_double_orphan.py` | QuickBundles (MDF, 20 mm) clusters of foundational streamlines outside TS / outside both atlases |
| 7. Atlas × acceptance stratification | `analysis/analyze_triple_no.py` | Counts, mean length, region and top TS bundles per (TS±, SWM±) × pool cell |
| 8. Redundant pool characterization | `analysis/analyze_redundant_pool_all.py` | Pool fractions, lengths, TS-unclassified share, MDF distance R→nearest F (lower table) |
| 9. Figures | `figures/plot_redundancy_location.py` | **Fig. 1** – `redundancy_location_map_triads.png` (subject 837964) |
|  | `figures/plot_foundational_clusters.py`, `figures/plot_supwma_figures.py` | Supporting cluster renderings and SWM plots |

`cluster_foundational_double_orphan.py` imports `assign_region` from `analyze_triple_no.py`,
so keep the `analysis/` scripts together.

## Getting the code

TractSeg and SUP-WMA are included as git submodules, pinned to the versions used for the abstract:

```bash
git clone --recurse-submodules git@github.com:minnelab/tractography-redundancy.git
# or, in an existing clone:
git submodule update --init
```

- **TractSeg** (`external/TractSeg`, v2.10): install with `pip install -e external/TractSeg`
  so the `TractSeg` command used by `run_tractseg_all.sh` is on the `PATH`.
- **SUP-WMA** (`external/SupWMA`): download the trained two-stage model as described in its
  README and extract it to `external/SupWMA/TrainedModel_TwoStage/`.
  `run_supwma_subject.py` uses this submodule by default (override with `SUPWMA_DIR`).
  For headless runs we changed `matplotlib.use('TkAgg')` to `'Agg'` in
  `external/SupWMA/utils/metrics_plots.py`.

## Data

Subjects: HCP `665254`, `734045`, `784565`, `837964`, `877168` (minimally preprocessed
HCP data; tractograms are 10 M-streamline probabilistic IFOD2). Data are not included.
The analysis scripts expect this layout under a data root:

```
<ROOT>/Data/Intersection_Labels/all_subjects/<subj>.csv          # commit_acc, sift_acc
<ROOT>/Data/Tractogram_Data/{TRAIN,VAL,TEST}/<subj>.trk
<ROOT>/Data/Tractogram_Data/{TRAIN,VAL,TEST}/<subj>_bundle_classifications.csv
<ROOT>/Data/SupWMA/<subj>_supwma_labels.csv
```

Set the root with `export TRACTO_ROOT=/path/to/root` (scripts default to the original
workstation path). `analyze_bundle_filtering_detailed.py` and
`cluster_foundational_unclassified.py` use paths relative to the root, so run them from
`$TRACTO_ROOT`. Preprocessing scripts contain site-specific paths that need adjusting.

## Dependencies

Python ≥ 3.10 with `numpy`, `pandas`, `scipy`, `matplotlib`, `nibabel`, `dipy`, `h5py`, `vtk`, `tqdm`, plus:

- [TractSeg](https://github.com/MIC-DKFZ/TractSeg) (submodule) and [pyAFQ](https://github.com/yeatmanlab/pyAFQ) for bundle labels
- [SUP-WMA](https://github.com/SlicerDMRI/SupWMA) (submodule) and its dependencies (PyTorch, `whitematteranalysis`)
- [COMMIT](https://github.com/daducci/COMMIT) / [AMICO](https://github.com/daducci/AMICO) for rCOMMIT, and MRtrix3 for SIFT

## Acknowledgments

Data were provided by the Human Connectome Project (WU-Minn Consortium; 1U54MH091657).
Funded by the Swedish Research Council (2022-03389), Digital Futures, and MedTechLabs.
Compute: NAISS Alvis and Berzelius (Knut and Alice Wallenberg Foundation).
