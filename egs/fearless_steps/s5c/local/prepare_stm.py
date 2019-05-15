#!/usr/bin/env python3

"""
################################################################################
#
# Script to prepare a NIST .stm file for scoring ASR output.  Based on the files
# that are naturally created for Kaldi acoustic training:
#
#  -  data/segments: contains segmentID, recordingID, start-time & end-time
#
#  -  data/wav.scp: contains recordingID & waveform-name (or sph2pipe command)
#
#  -  data/utt2spk: contains segmentID % speakerID
#
#  -  data/text: contains segment ID and transcription
#
# The .stm file has lines of the form
#
#    waveform-name channel speakerID start-time end-time [<attr>] transcription
#
# Clearly, most of the information needed for creating the STM file is present
# in the four Kaldi files mentioned above, except channel --- its value will be
# obtained from the sph2pipe command if present, or will default to "1" --- and
# <attributes> from a separate demographics.tsv file. (A feature to add later?)
#
# Note: Some text filtering is done by this script, such as removing non-speech
#       tokens from the transcription, e.g. <cough>, <breath>, etc.

        $fragMarkers = ""; # If given by the user, they are stripped from words

#       But two types of tokens are retained as is, if present.
#
        $Hesitation = "<hes>"; # which captures hesitations, filled pauses, etc.
        $OOV_symbol = "<unk>"; # which our system outputs occasionally.
#
# Note: The .stm file must be sorted by filename and channel in ASCII order and
#       by the start=time in numerical order.  NIST recommends the unix command
#       "sort +0 -1 +1 -2 +3nb -4"
#
# This script will also produce an auxilliary file named reco2file_and_channel
# which is used by Kaldi scripts to produce output in .ctm format for scoring.
# So any channel ID assigned here will be consistent between ref and output.
#
# If the training text is Viterbi-aligned to the speech to obtain time marks,
# it should be straightforward to modify this script to produce a .ctm file:
#
#    waveform-file channel start-time duration word
#
# which lists the transcriptions with word-level time marks.
#
# Note: A .ctm file must be sorted via "sort +0 -1 +1 -2 +2nb -3"
#
################################################################################
"""

import argparse
import collections
import sys

from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser("""
    Prepares stm file from information in DataDir
    """)

    parser.add_argument("data_dir", help="Kaldi data directory")
    args = parser.parse_args()

    return args


class Segment(object):
    def __init__(self, reco, st, end):
        self.reco_id = reco
        self.start_time = float(st)
        self.end_time = float(end)


def read_segments(segments_file):
    segments = {}
    reco2utts = defaultdict(list)
    num_segments = 0

    for line in open(segments_file):
        tokens = line.strip().split()

        if len(tokens) != 4:
            raise Exception("Could not parse line {} in {}".format(line.strip(), segments_file))

        utt_id = tokens[0]
        if utt_id in segments:
            raise Exception("Duplicate segment ID {} in {}".format(utt_id, segments_file))

        segments[utt_id] = Segment(tokens[1], tokens[2], tokens[3])
        num_segments += 1
        reco2utts[tokens[1]].append(utt_id)

    print ("Read info about {} segment IDs from {}".format(
        num_segments, segments_file), file=sys.stderr)
    return segments, reco2utts


def read_utt2spk(utt2spk_file):
    utt2spk = {}
    for line in open(utt2spk_file):
        parts = line.strip().split()
        if len(parts) != 2:
            raise Exception("Could not parse line {} in {}".format(line.strip(), utt2spk_file))
        utt2spk[parts[0]] = parts[1]

    return utt2spk


def read_transcripts(text_file):
    """
    Reads the transcriptions from the text file
    """
    utt2text = {}

    for line in open(text_file, encoding='utf-8'):
        parts = line.strip().split()

        assert len(parts) >= 1

        if len(parts) == 1:
            utt_id = parts[0]
            text = ""
            print ("Text is empty for line in {}:\n\t{}".format(text_file, line.strip()), file=sys.stderr)
        else:
            utt_id = parts[0]
            text = " ".join(parts[1:])
        utt2text[utt_id] = text

    return utt2text


def read_reco2dur(reco2dur_file):
    reco2dur = {}

    for line in open(reco2dur_file):
        parts = line.strip().split()
        if len(parts) != 2:
            raise Exception("Could not parse line {} in {}".format(line.strip(), reco2dur_file))
        reco_id = parts[0]
        dur = float(parts[1])
        reco2dur[reco_id] = dur

    return reco2dur


def main():
    args = get_args()

    segments_file = "{}/segments".format(args.data_dir)
    utt2spk_file = "{}/utt2spk".format(args.data_dir)
    text_file = "{}/text".format(args.data_dir)
    reco2dur_file = "{}/reco2dur".format(args.data_dir)
    stm_file = "{}/stm".format(args.data_dir)

    segments, reco2utts = read_segments(segments_file)
    utt2text = read_transcripts(text_file)
    utt2spk = read_utt2spk(utt2spk_file)
    reco2dur = read_reco2dur(reco2dur_file)

    ignore_text = "IGNORE_TIME_SEGMENT_IN_SCORING"

    with open(stm_file, mode='w', encoding='utf-8') as f:
        for reco_id in sorted(list(reco2utts.keys())):
            utts = reco2utts[reco_id]
            utts.sort(key=lambda x:(segments[x].start_time, segments[x].end_time))

            reco_start_time = segments[utts[0]].start_time
            reco_end_time = segments[utts[-1]].end_time

            print ("{file_id} {channel_id} {spk_id} {start_time:.2f} {end_time:.2f} "
                   " {text}".format(file_id=reco_id, channel_id="1", spk_id="NONE",
                                    start_time=0.0, end_time=reco_start_time,
                                    text=ignore_text), file=f)

            for utt_id in utts:
                print ("{file_id} {channel_id} {spk_id} {start_time:.2f} {end_time:.2f} "
                       " {text}".format(file_id=reco_id, channel_id="1",
                                        spk_id=utt2spk[utt_id],
                                        start_time=segments[utt_id].start_time,
                                        end_time=segments[utt_id].end_time,
                                        text=utt2text[utt_id]), file=f)

            print ("{file_id} {channel_id} {spk_id} {start_time:.2f} {end_time:.2f} "
                   " {text}".format(file_id=reco_id, channel_id="1", spk_id="NONE",
                                    start_time=reco_end_time,
                                    end_time=reco2dur[reco_id],
                                    text=ignore_text), file=f)


if __name__ == "__main__":
    main()
