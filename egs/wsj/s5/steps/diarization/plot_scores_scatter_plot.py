#! /usr/bin/env python

from __future__ import print_function
import argparse
import numpy as np
import sys
sys.path.insert(0, 'steps')

import libs.kaldi_io as kaldi_io
import libs.common as common_lib


def parse_args():
    parser = argparse.ArgumentParser("Plot scatter plot of scores")
    parser.add_argument("--plot-title", type=str,
                        action=common_lib.NullstrToNoneAction, default=None)
    parser.add_argument("--utt2spk-file", type=argparse.FileType('r'),
                        help="utt2spk file")
    parser.add_argument("--use-agg", choices=["true", "false"],
                        default="true", help="use agg")
    parser.add_argument("scores_file", type=str,
                        help="Scp file of scores matrices")
    parser.add_argument("reco2utt_file", type=argparse.FileType('r'),
                        help="reco2utt file of scores matrices")
    parser.add_argument("out_pdf", help="Output PDF file")

    args = parser.parse_args()
    args.use_agg = (args.use_agg == "true")
    return args


def format_coord(X, x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    numrows = len(X)
    numcols = len(X[0])
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

def reorder_mat(X, mapping):
    Y = [ [ X[i][j] for j in mapping ] for i in mapping ]
    return Y

def run(args):
    import matplotlib
    if args.use_agg:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    reco2utts = {}
    for line in args.reco2utt_file:
        parts = line.strip().split()
        reco2utts[parts[0]] = tuple(parts[1:])

    #spk_id = {}
    utt2spk = {}
    if args.utt2spk_file is not None:
        for line in args.utt2spk_file:
            parts = line.strip().split()
            #if parts[1] not in spk_id:
            #    spk_id[parts[1]] = len(spk_id)
            utt2spk[parts[0]] = parts[1] #spk_id[parts[1]]
    assert len(reco2utts)==1

    for reco, scores_mat in kaldi_io.read_mat_scp(args.scores_file):
        utts = reco2utts[reco]

        sorted_spks = []
        for i, u in enumerate(utts):
            sorted_spks.append((utt2spk[u], i))
        sorted_spks.sort(key=lambda x:x[0])
        # sorted_spks now contains (speaker, original_utt_index)

        # Create mapping from original utterance index in utts
        # to new index
        new2old = []
        for i, t in enumerate(sorted_spks):
            new2old.append(t[1])
            assert utt2spk[utts[t[1]]] == sorted_spks[i][0]

        f, axes = plt.subplots(2 if args.utt2spk_file is not None else 1,
                               sharex=True, sharey=True)

        if args.utt2spk_file is not None:
            ref_spk_mat = np.zeros((len(utts), len(utts)))
            for i, utt_i in enumerate(utts):
                for j, utt_j in enumerate(utts):
                    ref_spk_mat[i][j] = 1 if utt2spk[utt_i] == utt2spk[utt_j] else -1
            scores_mat = np.array(reorder_mat(scores_mat, new2old))
            ref_spk_mat = np.array(reorder_mat(ref_spk_mat, new2old))

            correct_mat = np.dot(ref_spk_mat, scores_mat)
            #incorrect_mat = np.dot(-ref_spk_mat, scores_mat)

            axes[0].matshow(scores_mat)
            axes[1].matshow(ref_spk_mat)
            dot_prod = np.linalg.norm(correct_mat, 2)
            sys_norm = np.linalg.norm(scores_mat, 2)
            ref_norm = np.linalg.norm(ref_spk_mat, 2)
            print("Normalized dot product for recording {0} is {1}".format(reco, dot_prod / sys_norm / ref_norm),
                  file=sys.stderr)
            axes[0].set_xticks(range(len(scores_mat)))
            axes[0].set_xticklabels([utt2spk[utts[new2old[i]]] for i in range(len(utts))], rotation=90)
            axes[0].set_yticks(range(len(scores_mat)))
            axes[0].set_yticklabels([utt2spk[utts[new2old[i]]] for i in range(len(utts))])
            axes[0].format_coord = lambda x,y: format_coord(scores_mat, x, y)
            #axes[1].set_xticks(range(len(scores_mat)))
            #axes[1].set_xticklabels([utt2spk[utt] for utt in utts], rotation=90)
            axes[1].set_yticks(range(len(scores_mat)))
            axes[1].set_yticklabels([utt2spk[utts[new2old[i]]] for i in range(len(utts))])
            axes[1].format_coord = lambda x,y: format_coord(ref_spk_mat, x, y)
        else:
            axes.matshow(scores_mat)

        f.tight_layout()
        f.subplots_adjust(hspace=0)

        if args.plot_title is None:
            f.suptitle('Scatter plot of scores')
        else:
            f.suptitle(args.plot_title)

        plt.show()

        plt.savefig(args.out_pdf + "-reco.pdf")


def main():
    args = parse_args()
    run(args)


if __name__ == '__main__':
    main()

