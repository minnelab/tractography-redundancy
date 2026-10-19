import os
import argparse
import numpy as np
import pandas as pd

from rc_eval_results import eval_all_rc_results, optimized_eval_all_rc_results


def combine_rates(subject, root, sizes, trk_size=int(1e7)):
    accepted = np.zeros(trk_size)
    votes = np.zeros(trk_size)
    succeeded = []
    for rate in sizes:
        print("Processing rate: {}".format(rate))
        try:
            file = f"{root}/{subject}/eval_rate_{rate}.csv"
            df = pd.read_csv(file)
            accepted = np.nansum([accepted, df["accepted"].values], axis=0)
            votes = np.nansum([votes, df["votes"].values], axis=0)
            succeeded.append(str(rate))

        except:
            print(f"No file for {subject} and rate {rate}")
            print(f"File: {file}")

    # Ensure there is at least one vote per row
    if np.any(votes == 0):
        print("There are rows with zero votes.")
        votes = votes[votes != 0]
        accepted = accepted[votes != 0]

    acc_rate = accepted / votes

    # Ensure there are no NaNs in the resulting 'acc_rate' column
    if np.isnan(acc_rate).any():
        print("There are NaN values in the 'acc_rate' column.")
        acc_rate = acc_rate[~np.isnan(acc_rate)]

    df = pd.DataFrame({"acc_rate": acc_rate})
    return df


def process_subjects(
    subjects, sizes, root_dir, results_dir, output_dir, second_root_dir, save_csv_dir
):
    for subject in subjects:
        file_path = os.path.join(root_dir, subject)
        results_dir_path = os.path.join(file_path, results_dir)

        for size in sizes:
            if os.path.exists(os.path.join(file_path, "sub_" + size)):
                print("Processing subject: " + subject + " with size " + size)
                optimized_eval_all_rc_results(
                    file_path,
                    results_dir_path,
                    size,
                    int(1e7),
                    out_path=os.path.join(output_dir, subject),
                )
            elif os.path.exists(os.path.join(second_root_dir, subject, "sub_" + size)):
                print(
                    "Using second file path for subject: "
                    + subject
                    + " with size "
                    + size
                )
                print("Processing subject: " + subject + " with size " + size)
                results_dir_path = os.path.join(
                    second_root_dir, subject, "sub_" + size, results_dir
                )

                optimized_eval_all_rc_results(
                    os.path.join(second_root_dir, subject),
                    results_dir_path,
                    size,
                    int(1e7),
                    out_path=os.path.join(output_dir, subject),
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/mnt/disk2/xwan/data/")
    parser.add_argument("--results_dir", type=str, default="rc_results")
    parser.add_argument("--output_dir", type=str, default="commit_eval_results")
    parser.add_argument("--second_file_path", type=str, default="./results/")
    parser.add_argument("--save_csv_dir", type=str, default="./new_commit_results/")
    parser.add_argument("--no_subject_processing", action="store_true")
    args = parser.parse_args()

    subjects = [
        "599571",
        "613538",
        "665254",
 #       "734045",
        "784565",
        "837964",
        "877168",
        "904044",
    ]
#    subjects = ["734045"]
    sizes = [
        "250000",
        "500000",
        "625000",
        "1000000",
        "1250000",
        "2500000",
        "5000000",
        "10000000",
    ]
    if not args.no_subject_processing:
        process_subjects(
            subjects,
            sizes,
            args.root_dir,
            args.results_dir,
            args.output_dir,
            args.second_file_path,
            args.save_csv_dir,
        )

    for subject in subjects:
        df = combine_rates(subject, root=args.output_dir, sizes=sizes)
        df.to_csv(os.path.join(args.save_csv_dir, f"{subject}.csv"))
