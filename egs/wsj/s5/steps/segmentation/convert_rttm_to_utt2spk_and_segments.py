#! /usr/bin/env python

import sys
import argparse

parser = argparse.ArgumentParser("This script converts an RTTM with "
                                 "speaker info into kaldi utt2spk and segments")
parser.add_argument("rttm_file", type = str,
                    help = "Input RTTM file")
parser.add_argument("reco2file_and_channel", type = str,
                    help = "Input reco2file_and_channel");
parser.add_argument("utt2spk", type = str,
                    help = "Output utt2spk file")
parser.add_argument("segments", type = str,
                    help = "Output segments file")

args = parser.parse_args();

file_and_channel2reco = {}
for line in open(args.reco2file_and_channel):
    parts = line.strip().split()
    file_and_channel2reco[(parts[1],parts[2])] = parts[0]

utt2spk_writer = open(args.utt2spk, 'w')
segments_writer = open(args.segments, 'w')

for line in open(args.rttm_file):
    parts = line.strip().split()
    if parts[0] != "SPEAKER":
        continue

    file_id = parts[1]
    channel = parts[2]

    try:
        reco = file_and_channel2reco[(file_id, channel)]
    except KeyError as e:
        sys.stderr.write("Could not find recording with (file_id, channel) = ({0},{1}) in {2}\n".format(file_id, channel, args.reco2file_and_channel))
        raise e

    start_time = float(parts[3])
    end_time = start_time + float(parts[4])

    spkr = parts[7]

    s = int(start_time * 100)
    e = int(end_time * 100)
    utt = "{0}-{1:06d}-{2:06d}".format(spkr,s,e)

    utt2spk_writer.write("{0} {1}\n".format(utt,spkr))
    segments_writer.write("{0} {1} {2:7.2f} {3:7.2f}\n".format(utt, reco, start_time, end_time))

