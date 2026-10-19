import os
import glob
import shutil


def clean_commit_results(sub_file_path):
    shutil.rmtree(os.path.join(sub_file_path, "COMMIT"))
    # shutil.rmtree(os.path.join(sub_file_path,'kernels'))
    tck = glob.glob(os.path.join(sub_file_path, "*.tck"), recursive=True)
    os.remove(tck[0])
    return
