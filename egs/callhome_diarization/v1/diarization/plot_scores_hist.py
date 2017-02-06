#! /usr/bin/env python

import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import sys
sys.path.insert(0, 'steps')

import libs.kaldi_io as kaldi_io

from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Plot histogram of scores.")
    parser.add_argument("scores_file", help="Archive of scores matrices")
    parser.add_argument("out_pdf", help="Output PDF file")

    args = parser.parse_args()
    return args


def run(args):
    scores = np.array([])
    for k,v in kaldi_io.read_mat_ark(args.scores_file):
        scores = np.concatenate((scores, v.flatten()))

    f = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(scores, 100, facecolor='green', alpha=0.75)

    plt.xlabel('Scores')
    plt.ylabel('Count')
    plt.title(r'Histogram of scores')
    plt.grid(True)

    plt.show()

    f.savefig(args.out_pdf, bbox_inches='tight')


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()
