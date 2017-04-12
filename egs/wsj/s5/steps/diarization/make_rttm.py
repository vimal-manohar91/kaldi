#!/usr/bin/env python

# Copyright  2016  David Snyder
# Apache 2.0.
# TODO, script needs some work, error handling, etc

import argparse
import sys

sys.path.insert(0, 'steps')
import libs.common as common_lib

parser = argparse.ArgumentParser("Convert segments and labels to RTTM.")
parser.add_argument("segments", type=argparse.FileType('r'), help="Segments file")
parser.add_argument("labels", type=argparse.FileType('r'), help="labels file")
parser.add_argument("--reco2file-and-channel", type=str, default=None,
                    action=common_lib.NullstrToNoneAction,
                    help="reco2file_and_channel file to map recording to "
                    "file and channel")

args = parser.parse_args()

segments_fi = args.segments.readlines()
label_fi = args.labels.readlines()

# File containing speaker labels per utt
seg2label = {}
for l in label_fi:
    seg, label = l.rstrip().split()
    seg2label[seg] = label

# Segments file
utt2seg = {}
for l in segments_fi:
    seg, utt, s, e = l.rstrip().split()
    if utt in utt2seg:
        utt2seg[utt] = utt2seg[utt] + " " + s + "," + e + "," + seg2label[seg]
    else:
        utt2seg[utt] = utt + " " + s + "," + e + "," + seg2label[seg]


# TODO Cut up the segments so that they are contiguous
diarization1 = []
for utt in utt2seg:
    l = utt2seg[utt]
    t = l.rstrip().split()
    utt = t[0]
    rhs = ""
    for i in range(1, len(t)-1):
        s, e, label = t[i].split(',')
        s_next, e_next, label_next = t[i+1].split(',')
        if float(e) > float(s_next):
            avg = str((float(s_next) + float(e)) / 2.0)
            t[i+1] = ','.join([avg, e_next, label_next])
            rhs += " " + s + "," + avg + "," + label
        else:
            rhs += " " + s + "," + e + "," + label
    s, e, label = t[-1].split(',')
    rhs += " " + s + "," + e + "," + label
    diarization1.append(utt + rhs)

# TODO Merge the contiguous segments that belong to the same speaker
diarization2 = []
for l in diarization1:
    t = l.rstrip().split()
    utt = t[0]
    rhs = ""
    for i in range(1, len(t)-1):
        s, e, label = t[i].split(',')
        s_next, e_next, label_next = t[i+1].split(',')
        if float(e) == float(s_next) and label == label_next:
            t[i+1] = ','.join([s, e_next, label_next])
        else:
            rhs += " " + s + "," + e + "," + label
    s, e, label = t[-1].split(',')
    rhs += " " + s + "," + e + "," + label
    diarization2.append(utt + rhs)

reco2file_and_channel = {}
if args.reco2file_and_channel is not None:
    for line in open(args.reco2file_and_channel):
        try:
            reco, file_, channel = line.strip().split()
            reco2file_and_channel[reco] = (file_, channel)
        except Exception:
            logger.error("Unable to parse line {0} in file {1}".format(
                line, args.reco2file_and_channel))
            raise


for l in diarization2:
    t = l.rstrip().split()
    file_ = t[0]
    channel = 1
    if args.reco2file_and_channel is not None:
        file_, channel = reco2file_and_channel[file_]

    for i in range(1, len(t)):
        s, e, label = t[i].rstrip().split(',')
        print("SPEAKER {file} {channel} {st} {dur} <NA> <NA> {spk} <NA> <NA>"
              "".format(file=file_, channel=channel, st=s,
                        dur=max(0, float(e) - float(s)), spk=label))
