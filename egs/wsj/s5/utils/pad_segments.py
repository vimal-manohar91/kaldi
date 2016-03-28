#!/usr/bin/env python

# Copyright 2016 Vimal Manohar
# Apache 2.0

import argparse
import sys
import os
import subprocess
import errno
import copy
import shutil

def GetArgs():
    parser = argparse.ArgumentParser(description="""
    This script pads segments to ensure that utterance begin and end with silence """)

    parser.add_argument("--pad-length-left", type=float, default = 0,
                        help="Length to pad on the left of segments")
    parser.add_argument("--pad-length-right", type=float, default = 0,
                        help="Length to pad on the right of segments")
    parser.add_argument("--reco-lengths", type=str,
                        help="Archive of recording lengths to ensure that "
                        "the produced lengths do not go beyond the boundary of the recording")
    parser.add_argument("input_segments", metavar="<input-segments>",
                        type=str)
    parser.add_argument("output_segments", metavar="<output-segments>",
                        type=str)

    print(' '.join(sys.argv))
    args = parser.parse_args()
    return args

def CheckArgs(args):
    if args.input_segments == "-":
        args.input_handle = sys.stdin
    else:
        args.input_handle = open(args.input_segments)
    if args.output_segments == "-":
        args.output_handle = sys.stdout
    else:
        args.output_handle = open(args.output_segments, 'w')

    return args

def PadSegments(input_handle, output_handle,
                pad_left, pad_right, reco_lengths = None):
    for line in input_handle.readlines():
        splits = line.strip().split()
        assert(len(splits) in [4,5])

        utt = splits[0]
        reco = splits[1]
        beg = float(splits[2])
        end = float(splits[3])

        beg = max(0, beg - pad_left)
        end = min(end + pad_right,
                reco_lengths[reco] if reco_lengths is not None else float("inf"))

        out_line = '%s %s %.02f %.02f' % (utt,reco,beg,end)

        if (len(splits) > 4):
            out_line += ' ' + splits[4]

        output_handle.write(out_line + '\n')

def Main():
    args = GetArgs()
    args = CheckArgs(args)

    reco_lengths = {}
    if args.reco_lengths is not None:
        for line in open(args.reco_lengths):
            splits = line.strip().split()
            assert(len(splits) == 2)
            reco_lengths[splits[0]] = float(splits[1])

    PadSegments(args.input_handle, args.output_handle,
                args.pad_length_left, args.pad_length_right,
                reco_lengths if args.reco_lengths is not None else None)
    args.output_handle.close()

if __name__ == "__main__":
    Main()
