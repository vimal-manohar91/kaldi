#!/usr/bin/env python3

# Copyright 2018  Vimal Manohar
# Apache 2.0

import argparse
import glob
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
    wav_writer = open("{}/wav_list".format(args.dir), 'w')

    call_id2file_id = {}
    for f in glob.glob("{}/data/ulaw_sphere/*.sph".format(args.source_dir)):
        basename = os.path.basename(f)
        file_id = basename[0:-4]
        call_id = file_id.split('_')[-1]
        call_id2file_id[call_id] = file_id

        for channel in [1, 2]:
            reco_id = "{}-{}".format(file_id, channel)
            print ("{} {} {}".format(reco_id, file_id, channel),
                   file=reco2file_and_channel_writer)

        print ("{} {}".format(file_id, f),
               file=wav_writer)

    reco2file_and_channel_writer.close()
    wav_writer.close()

    mx6_calls_file = open("{}/docs/mx6_calls.csv".format(args.source_dir))
    segments_writer = open("{}/segments".format(args.dir), 'w')
    utt2spk_writer = open("{}/utt2spk".format(args.dir), 'w')
    utt2details_writer = open(
        "{}/utt2details".format(args.dir), 'w')

    good_utts = open(
        "{}/good_utts".format(args.dir), 'w')
    acceptable_utts = open(
        "{}/acceptable_utts".format(args.dir), 'w')
    unsuitable_utts = open(
        "{}/unsuitable_utts".format(args.dir), 'w')

    num_lines = 0

    num_good = 0
    num_acceptable = 0
    num_unsuitable = 0

    column_id = {}
    for line in mx6_calls_file.readlines():
        parts = line.strip().split(",")

        if len(parts) != 21:
            raise TypeError("Expected 10 columns; got {0} in {1}".format(
                len(parts), line.strip()))

        num_lines += 1
        if num_lines == 1:
            for i,col in enumerate(parts):
                column_id[col] = i
            continue

        call_id = parts[column_id["call_id"]]
        file_id = call_id2file_id[call_id]

        for channel in [1, 2]:
            side = "a" if channel == 1 else "b"
            reco_id = "{}-{}".format(file_id, channel)

            speaker_id = parts[column_id["sid_"+side]]

            conversation_quality = parts[column_id["cnvq_"+side]]
            signal_quality = parts[column_id["sigq_"+side]]
            technical_problem = parts[column_id["tbug_"+side]]

            if conversation_quality == "":
                continue

            utt_id = "{}-{}".format(speaker_id, reco_id)

            if (conversation_quality == "G" and
                    signal_quality == "G" and
                    technical_problem == "N"):
                print (utt_id, file=good_utts)
                num_good += 1
            elif (conversation_quality == "U" or
                  signal_quality == "U" or
                  technical_problem == "Y"):
                print (utt_id, file=unsuitable_utts)
                num_unsuitable += 1
            else:
                print (utt_id, file=acceptable_utts)
                num_acceptable += 1

            print ("{} {}".format(utt_id, speaker_id),
                   file=utt2spk_writer)
            print ("{} {} 0.0 -1".format(utt_id, reco_id),
                   file=segments_writer)
            print ("{} {} {} {}".format(utt_id, conversation_quality,
                                        signal_quality, technical_problem),
                   file=utt2details_writer)

    print ("Read lines corresponding to {} calls".format(num_lines - 1),
           file=sys.stderr)

    mx6_calls_file.close()
    utt2spk_writer.close()
    segments_writer.close()
    utt2details_writer.close()
    good_utts.close()
    acceptable_utts.close()
    unsuitable_utts.close()


if __name__ == "__main__":
    main()
