"""
Retrieves percentage of positives votes received by streamlines across all ex-
periment instances/subset sizes. Either for all streamlines or a set of stream-
lines of interest.

Command line args:
--json_path: path to json file containing indices that are relevant for
the statistics. Will otherwise evaluate for all streamlines.

Author: Antonia Hain (s8anhain@stud.uni-saarland.de)
"""

import json
import sys
import os
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter

import evaluation_helper

OUTPUT_FOLDER_NAME = "output"
MAX_SUBSETS = 455


def build_argparser():
    default_subjects = [
        "599671-retest",
        "734045",
        "665254",
        "784565",
        "837964",
        "877168-retest",
        "904044",
    ]

    default_subjects = [
        "734045",
        "665254",
        "784565",
        "837964",
        "877168-retest",
        "904044",
    ]
    default_subjects = ["877168/877168"]

    default_subjects = [
        "599671",
        "734045",
        "665254",
        "784565",
        "837964",
        "877168/877168",
        "904044",
    ]

    p = ArgumentParser(formatter_class=RawTextHelpFormatter)
    p.add_argument(
        "--file_path",
        help="path to folder containing the output folders",
        default="/mnt/disk2/randomised_filtering/data",
    )

    p.add_argument(
        "--results_path",
        required=False,
        help="path to folder containing the results",
        default="../sift_results",
    )
    # add argument for subjects as a list
    p.add_argument(
        "--subjects",
        nargs="+",
        required=False,
        help="list of subjects to be evaluated",
        default=default_subjects,
    )

    return p


def main():
    args = vars(build_argparser().parse_args())

    root = args["file_path"]
    for subject in args["subjects"]:
        filepath = os.path.join(root, subject)
        # OUTPUT_FOLDER_NAME= os.path.join(filepath, 'output')
        # get all relevant folders from the path (name must begin with OUTPUT_FOLDER_NAME)
        print(filepath)
        folders = evaluation_helper.get_output_folders(filepath, OUTPUT_FOLDER_NAME)
        print("Found {} output folders.".format(folders))
        # add filepaths to folders
        folders = [os.path.join(filepath, folder) for folder in folders]
        # only keep folders with 10000000 streamlines
        folders = [folder for folder in folders if "10000000" in folder]
        # get vote distributions for each streamline across all subset sizes in one variable
        meta_streamline_index = evaluation_helper.get_meta_streamline_index(folders)

        # get relevant streamlines

        entries = meta_streamline_index
        name = ""

        # get acceptance rates for all streamlines of interests and write to file.
        result = np.array(
            [
                [entry[2], evaluation_helper.get_acceptance_rate(entry)]
                for entry in entries
            ]
        )
        results_path = args["results_path"]
        # check is
        np.savetxt(os.path.join(results_path, f"{subject[:6]}") + name + ".txt", result)


if __name__ == "__main__":
    main()
