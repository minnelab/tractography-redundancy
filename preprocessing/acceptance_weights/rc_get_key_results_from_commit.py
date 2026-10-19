import os
import io
from runpy import run_module
from unittest import result
import shutil
import glob


def get_key_results_from_commit(
    sub_file_path, raw_file_path, subset_size, runtime, output_path=None
):
    weight_result_path = os.path.join(
        sub_file_path, "COMMIT/Results_StickZeppelinBall/streamline_weights.txt"
    )
    subset = glob.glob(
        os.path.join(sub_file_path, "*%s.json" % runtime), recursive=True
    )
    print(subset)
    subset = subset[0]

    if output_path is None:
        rc_result_path = os.path.join(raw_file_path, "rc_results")
    else:
        rc_result_path = output_path
    if not os.path.exists(rc_result_path):
        os.makedirs(rc_result_path)

    shutil.copy2(
        weight_result_path,
        os.path.join(
            rc_result_path, "streamline_weights_%s_%d.txt" % (subset_size, runtime)
        ),
    )
    shutil.copy2(
        subset, os.path.join(rc_result_path, "idx_%s_%d.txt" % (subset_size, runtime))
    )

    return
