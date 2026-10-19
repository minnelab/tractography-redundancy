import pandas as pd
import numpy as np
import nibabel as nib
import os
import glob
import json
import sys


def optimized_eval_all_rc_results(
    raw_file_path, results_path, size, raw_size, out_path=None
):
    if out_path is None:
        eval_path = os.path.join(raw_file_path, "eval_results")
    else:
        eval_path = out_path
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    print(results_path)
    print(glob.glob(results_path))
    idx_files = glob.glob(
        os.path.join(results_path, "idx_{}_*.txt".format(size)), recursive=True
    )
    num_idx_files = len(idx_files)
    print("Number of idx files: {}".format(num_idx_files))
    i = 0

    # Initialize the metrics
    votes = np.zeros(raw_size)
    accepted = np.zeros(raw_size)
    sums = np.zeros(raw_size)
    sums_sq = np.zeros(raw_size)

    for idx_file in idx_files:
        print("Opening file: {}".format(idx_file))
        if os.path.exists(idx_file) and os.path.getsize(idx_file) > 0:
            with open(idx_file) as f:
                json_data = json.load(f)
        else:
            print(f"File does not exist or is empty")
            continue

        weight_file = os.path.join(
            results_path,
            "streamline_weights_{}_{}.txt".format(
                json_data["size"][0], json_data["runtime"][0]
            ),
        )

        weight = np.loadtxt(weight_file, delimiter=",", unpack=False)

        # Update the metrics
        votes[json_data["idx"]] += 1
        accepted[json_data["idx"]] += weight > 0
        sums[json_data["idx"]] += weight
        sums_sq[json_data["idx"]] += weight**2

        print("{} saved among {} files.".format(i + 1, num_idx_files))
        i += 1

    # Compute the final metrics
    eps = np.finfo(
        float
    ).eps  # smallest representable positive number such that 1.0 + eps != 1.0
    acc_rate = np.divide(accepted, votes + eps)
    mean = np.divide(sums, votes + eps)
    std = np.sqrt(np.divide(sums_sq, votes + eps) - mean**2)

    # Build the final dataframe
    rate = pd.DataFrame(
        {
            "votes": votes,
            "accepted": accepted,
            "acc_rate": acc_rate,
            "mean": mean,
            "std": std,
        }
    )
    print(rate.shape)

    print("Saving summarized data into csv file...")
    rate.to_csv(os.path.join(eval_path, "eval_rate_{}.csv".format(size)))


def eval_all_rc_results(raw_file_path, results_path, size, raw_size):
    eval_path = os.path.join(raw_file_path, "eval_results")
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    eval = pd.DataFrame(index=range(raw_size))

    idx_files = glob.glob(
        os.path.join(results_path, "idx_{}_*.txt".format(size)), recursive=True
    )
    num_idx_files = len(idx_files)
    i = 0

    for idx_file in idx_files:
        with open(idx_file) as f:
            json_data = json.load(f)
        weight_file = os.path.join(
            results_path,
            "streamline_weights_{}_{}.txt".format(
                json_data["size"][0], json_data["runtime"][0]
            ),
        )
        col_name = "{}_{}".format(json_data["size"][0], json_data["runtime"][0])

        weight = np.loadtxt(weight_file, delimiter=",", unpack=False)
        eval[col_name] = np.full(raw_size, np.nan)
        eval[col_name].iloc[json_data["idx"]] = weight
        print("{} saved among {} files.".format(i + 1, num_idx_files))
        i += 1
    print("Saving data into csv file...")
    eval.to_csv(os.path.join(eval_path, "eval_{}.csv".format(size)))

    # Dropout the streamlines with all nan values/ not chosen
    no_all_nan = eval.dropna(axis=0, how="all")
    # Build df with chosen streamlines
    rate = pd.DataFrame(index=no_all_nan.index)
    rate["votes"] = np.sum(no_all_nan >= 0, axis=1)
    rate["accepted"] = np.sum(no_all_nan > 0, axis=1)
    rate["acc_rate"] = rate["accepted"] / rate["votes"]
    rate["mean"] = np.mean(no_all_nan, axis=1)
    rate["std"] = np.std(no_all_nan, axis=1)

    print("Saving summarized data into csv file...")
    rate.to_csv(os.path.join(eval_path, "eval_rate_{}.csv".format(size)))


def summarize_rc_results(results_path, size):
    """
    size consists of size names.
    e.g. size = [5000000,2000000,1250000,625000,500000,250000]
    """

    frame = []
    for i in range(len(size)):
        df = pd.read_csv(os.path.join(results_path, "eval_rate_{}.csv".format(size[i])))
        category(df)
        df["size"] = size[i]
        df.rename(columns={df.columns[0]: "idx"}, inplace=True)
        frame.append(df)
    sub = pd.concat(frame)
    sub.to_csv(os.path.join(results_path, "summarized_eval_results.csv"))


def category(eval):
    conditions = [
        (eval["acc_rate"] == 0),
        (eval["acc_rate"] > 0) & (eval["acc_rate"] <= 0.2),
        (eval["acc_rate"] > 0.2) & (eval["acc_rate"] <= 0.4),
        (eval["acc_rate"] > 0.4) & (eval["acc_rate"] <= 0.6),
        (eval["acc_rate"] > 0.6) & (eval["acc_rate"] <= 0.8),
        (eval["acc_rate"] > 0.8) & (eval["acc_rate"] < 1),
        (eval["acc_rate"] == 1),
    ]
    values = [0, 1, 2, 3, 4, 5, 6]
    eval["category"] = np.select(conditions, values)
    return eval


def extract_t_f_streamlines(results_path, summarized_results, size):
    """
    size consists of size names.
    e.g. size = ['5.00m','2.50m','1.25m','625k','500k','250k']
    """
    df = pd.read_csv(summarized_results, index_col=[0])
    for i in range(len(size)):
        df_t = df.loc[(df["size"] == size[i]) & (df["category"] == 6), ["idx"]]
        df_f = df.loc[(df["size"] == size[i]) & (df["category"] == 0), ["idx"]]

        if i != 0:
            t_streamlines = t_streamlines[t_streamlines["idx"].isin(df_t["idx"])]
            f_streamlines = f_streamlines[f_streamlines["idx"].isin(df_f["idx"])]
        else:
            t_streamlines = df_t
            f_streamlines = df_f
        print("Number of true streamlines is {}.".format(len(t_streamlines)))
        print("Number of false streamlines is {}.".format(len(f_streamlines)))

    t_streamlines.to_csv(os.path.join(results_path, "filtered_true_streamlines.csv"))
    f_streamlines.to_csv(os.path.join(results_path, "filtered_false_streamlines.csv"))
