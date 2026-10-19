import sys

#sys.path.append("/mnt/disk2/xwan/randomised_commit/data")

import amico

amico.setup()
from rc_generate_run_subsets import generate_run_subsets

generate_run_subsets(
    "/home/sanna/Desktop/Data/RAW_DATA/613538",
    [250_000, 500_000, 625_000, 1000_000, 1250_000, 2500_000, 5000_000, 10000_000],
    10000000,
    out_path="./results/r-COMMIT/613538",
)
