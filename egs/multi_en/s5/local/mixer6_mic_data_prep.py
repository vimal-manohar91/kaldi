#!/usr/bin/env python3

# Copyright 2018  Vimal Manohar
# Apache 2.0

import argparse
import os
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("source_dir", help="Mixer6 corpus directory")
    parser.add_argument("dir", help="Output directory")

    args = parser.parse_args()
    return args


def main():

    args = get_args()

    if not os.path.exists(args.dir):
        os.makedirs(args.dir)

    reco2file_and_channel_writer = open(
        "{}/reco2file_and_channel".format(args.dir), 'w')
    segments_writer = open("{}/segments".format(args.dir), 'w')
    utt2spk_writer = open("{}/utt2spk".format(args.dir), 'w')
    mx6_components_file = "{}/docs/mx6_ivcomponents.csv".format(args.source_dir)

    wav_writer = open("{}/wav_list".format(args.dir), 'w')
    call_id2file_id = {}
    for f in glob.glob("{}/data/pcm_flac/*/*.flac".format(args.source_dir)):
        basename = os.path.basename(f)
        file_id = basename[0:-5]
        call_id = file_id.split('_')[-1]
        call_id2file_id[call_id] = file_id

        for channel in [1, 2]:
            reco_id = "{}-{}".format(file_id, channel)
            print ("{} {} {}".format(reco_id, file_id, channel),
                   file=reco2file_and_channel_writer)

        print ("{} {}".format(file_id, f),
               file=wav_writer)

    i = 0
    for line in open(mx6_components_file).readlines():
        i += 1
        if i == 1:
              continue

        parts = line.strip().split(",")

        if len(parts) != 10:
            raise TypeError("Expected 10 columns; got {0} in {1}".format(
                len(parts), line.strip()))

        session_id = parts[0]
        call_bgn = float(parts[7])
        call_end = float(parts[8])
        call_type = parts[9]

        if call_type == "no_call":
            assert call_bgn == 0.0 and call_end == 0.0

        dur = call_end - call_bgn

        for channel in [ 1, 2 ]:
            reco_id = "{}-{}".format(session_id, channel)

            print ("{} {} {}".format(reco_id, session_id, channel),
                   file=reco2file_and_channel_writer)
            print ("{utt_id} {reco_id} {st} {end}".format(
                utt_id=reco_id, reco_id=reco_id, st=call_bgn, end=call_end),
                   file=segments_writer)
            print ("{} {}".format(reco_id, reco_id),
                   file=utt2spk_writer)

    print ("Read lines corresponding to {} sessions".format(i),
           file=sys.stderr)


if __name__ == "__main__":
    main()
