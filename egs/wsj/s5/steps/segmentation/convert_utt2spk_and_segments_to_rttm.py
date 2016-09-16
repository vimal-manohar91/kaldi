#! /usr/bin/env python

import sys
import argparse

parser = argparse.ArgumentParser("This script converts an RTTM with "
                                 "speaker info into kaldi utt2spk and segments")

parser.add_argument("utt2spk", type = str,
                    help = "Input utt2spk file")
parser.add_argument("segments", type = str,
                    help = "Input segments file")
parser.add_argument("reco2file_and_channel", type = str,
                    help = "Input reco2file_and_channel");
parser.add_argument("rttm_file", type = str,
                    help = "Output RTTM file")

args = parser.parse_args();

reco2file_and_channel = {}
for line in open(args.reco2file_and_channel):
    parts = line.strip().split()
    reco2file_and_channel[parts[0]] = (parts[1],parts[2])

utt2spk_reader = open(args.utt2spk, 'r')
segments_reader = open(args.segments, 'r')

rttm_writer = open(args.rttm_file, 'w')

utt2spk = {}
for line in utt2spk_reader:
    parts = line.strip().split()
    utt2spk[parts[0]] = parts[1]

for line in segments_reader:
    parts = line.strip().split()

    utt = parts[0]
    spkr = utt2spk[utt]

    reco = parts[1]

    try:
        file_id, channel = reco2file_and_channel[reco]
    except KeyError as e:
        sys.stderr.write("Could not find recording {0} in {1}\n".format(reco, args.reco2file_and_channel))
        raise e

    start_time = float(parts[2])
    duration = float(parts[3]) - start_time

    rttm_writer.write("SPEAKER {0} {1} {2:7.2f} {3:7.2f} <NA> <NA> {4} <NA>\n".format(file_id, channel, start_time, duration, spkr))

