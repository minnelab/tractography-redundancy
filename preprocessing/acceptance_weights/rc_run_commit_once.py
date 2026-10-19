from fileinput import filename
import commit
import amico
from commit import trk2dictionary
import glob
import os
import shutil


## input tck_path is the working dirctoty of subset file


def run_commit_on_subsets(raw_file_path, tck_path, runtime):
    ## Generate dwi scheme for use
    scheme = os.path.join(raw_file_path, "DWI.scheme")
    bvals = os.path.join(raw_file_path, "bvals.txt")
    bvecs = os.path.join(raw_file_path, "bvecs.txt")

    if not os.path.exists(scheme):
        amico.util.fsl2scheme(bvals, bvecs, scheme)

    ## Generate a dictionary
    subset = glob.glob(os.path.join(tck_path, "*.tck"), recursive=True)
    print(raw_file_path)
    peaks = glob.glob(os.path.join(raw_file_path, "peaks.nii.gz"), recursive=True)
    print(peaks)
    dwi_data = glob.glob(os.path.join(raw_file_path, "data.nii.gz"), recursive=True)
    scheme = glob.glob(os.path.join(raw_file_path, "*.scheme"), recursive=True)

    trk2dictionary.run(
        filename_tractogram=subset[0],
        filename_peaks=peaks[0],
        # filename_mask = 'WM.nii.gz',
        fiber_shift=0.5,
        peaks_use_affine=True,
    )

    mit = commit.Evaluation(".", ".")

    mit.load_data(dwi_data[0], scheme[0])

    mit.set_model("StickZeppelinBall")
    d_par = 1.7e-3
    d_perps_zrp = [0.51e-3]
    d_isos = [1.7e-3, 3.0e-3]
    mit.model.set(d_par, d_perps_zrp, d_isos)
    mit.generate_kernels(regenerate=True)
    mit.load_kernels()

    mit.load_dictionary(os.path.join(tck_path, "COMMIT"))
    mit.set_threads()
    mit.build_operator()
    mit.fit(tol_fun=1e-3, max_iter=1000)
    mit.save_results()
    x_ic, x_ec, x_iso = mit.get_coeffs()
