#! /usr/bin/env python

import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
sys.path.insert(0, 'steps')

import libs.kaldi_io as kaldi_io
import libs.common as common_lib

from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Plot histogram of scores.")
    parser.add_argument("--plot-title", type=str,
                        action=common_lib.NullstrToNoneAction, default=None)
    parser.add_argument("--ignore-diagonal", type=str, choices=["true", "false"],
                        action=common_lib.StrToBoolAction, default=False)
    parser.add_argument("scores_file", help="Scp file of scores matrices")
    parser.add_argument("out_pdf", help="Output PDF file")

    args = parser.parse_args()
    args.ignore_diagonal = bool(args.ignore_diagonal)
    return args


def run(args):
    scores = np.array([])
    for k,v in kaldi_io.read_mat_scp(args.scores_file):
        if args.ignore_diagonal:
            v = v[np.where(~np.eye(v.shape[0],dtype=bool))]
        v = v.flatten()
        scores = np.concatenate((scores, v))

    f = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(scores, 100, facecolor='green', alpha=0.75)

    plt.xlabel('Scores')
    plt.ylabel('Count')
    if args.plot_title is None:
        plt.title('Histogram of scores')
    else:
        plt.title(args.plot_title)

    plt.grid(True)

    plt.show()

    f.savefig(args.out_pdf, bbox_inches='tight')


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
