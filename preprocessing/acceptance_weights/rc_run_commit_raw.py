from fileinput import filename
import commit
import amico
from commit import trk2dictionary
import glob
import os
import shutil


def run_commit_raw(raw_file_path, repetition):
    ## Generate dwi scheme for use
    scheme = os.path.join(raw_file_path, "DWI.scheme")
    # bvals = os.path.join(raw_file_path, "bvals.txt")
    # bvecs = os.path.join(raw_file_path, "bvecs.txt")

    # if not os.path.exists(scheme):
    #     amico.util.fsl2scheme(bvals,bvecs,scheme)

    ## Generate a dictionary
    raw_file = glob.glob(os.path.join(raw_file_path, "*.tck"), recursive=True)
    peaks = os.path.join(raw_file_path, "peaks.nii.gz")
    dwi_data = os.path.join(raw_file_path, "data.nii.gz")

    for i in range(repetition):
        trk2dictionary.run(
            filename_tractogram=raw_file[0],
            filename_peaks=peaks,
            # filename_mask = 'WM.nii.gz',
            fiber_shift=0.5,
            peaks_use_affine=True,
        )

        mit = commit.Evaluation(".", ".")
        mit.load_data(dwi_data, scheme)

        mit.set_model("StickZeppelinBall")
        d_par = 1.7e-3
        d_perps_zrp = [0.51e-3]
        d_isos = [1.7e-3, 3.0e-3]
        mit.model.set(d_par, d_perps_zrp, d_isos)
        mit.generate_kernels(regenerate=True)
        mit.load_kernels()

        mit.load_dictionary(os.path.join(raw_file_path, "COMMIT"))
        mit.set_threads()
        mit.build_operator()
        mit.fit(tol_fun=1e-3, max_iter=1000)
        mit.save_results()
        x_ic, x_ec, x_iso = mit.get_coeffs()

        weight_result_path = os.path.join(
            raw_file_path, "COMMIT/Results_StickZeppelinBall/streamline_weights.txt"
        )
        rc_result_path = os.path.join(raw_file_path, "rc_results")
        if not os.path.exists(rc_result_path):
            os.makedirs(rc_result_path)
        shutil.copy2(
            weight_result_path,
            os.path.join(rc_result_path, f"streamline_weights_{i}.txt"),
        )
        shutil.rmtree(os.path.join(raw_file_path, "COMMIT"))
        return
