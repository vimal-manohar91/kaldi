#!/usr/bin/env python
# Copyright 2015   David Snyder
# Apache 2.0.
#
# Using the annotations created by refine_annotations_bn.py, this script
# creates the segments, utt2spk, and wav.scp files.
#
# This file is meant to be invoked by make_bn.sh.

import os, sys, argparse

class StrToBoolAction(argparse.Action):
    """ A custom action to convert bools from shell format i.e., true/false
        to python format i.e., True/False """
    def __call__(self, parser, namespace, values, option_string=None):
        if values == "true":
            setattr(namespace, self.dest, True)
        elif values == "false":
            setattr(namespace, self.dest, False)
        else:
            raise Exception("Unknown value {0} for --{1}".format(values, self.dest))

def GetArgs():
  parser = argparse.ArgumentParser(description = "Prepare broadcast news data directory")
  parser.add_argument("--overlapping-segments-removed", type = str,
                     action = StrToBoolAction, default = False,
                     choices = ["true", "false"],
                     help = "Specifies if the overlapping segments are removed")
  parser.add_argument("wav_dir", metavar = "<wav-dir>",
                     help = "Directory of sph files")
  parser.add_argument("out_dir", metavar = "<out-dir>",
                     help = "Output kaldi data directory")

  args = parser.parse_args()
  return args

def Main():
  args = GetArgs()

  wav_dir = args.wav_dir
  out_dir = args.out_dir

  utts = open(os.path.join(out_dir, "utt_list"), 'r').readlines()
  utts = set(x.rstrip() for x in utts)
  wav = ""
  segments = ""
  utt2spk = ""
  for subdir, dirs, files in os.walk(wav_dir):
    for file in files:
      utt = str(file).replace(".sph", "")
      if file.endswith(".sph") and utt in utts:
        wav = wav + utt + " sox " + wav_dir + "/" + utt + ".sph"  + " -c 1 -r 16000 -t wav - |\n"
  wav_fi = open(os.path.join(out_dir, "wav.scp"), 'w')
  wav_fi.write(wav)

  for utt in utts:
    music_filename = utt + "_music.key" + (".refined" if args.overlapping_segments_removed else "")
    speech_filename = utt + "_speech.key" + (".refined" if args.overlapping_segments_removed else "")
    music_fi = open(os.path.join(out_dir, music_filename), 'r').readlines()
    speech_fi = open(os.path.join(out_dir, speech_filename), 'r').readlines()
    count = 1
    for line in music_fi:
      left, right = line.rstrip().split(" ")
      segments = segments + utt + ("-music-%04d" % count) + " " + utt + " " + left + " " + right + "\n"
      utt2spk = utt2spk + utt + ("-music-%04d"%count) + " " + utt + ("-music-%04d"%count) + "\n"
      count += 1
    count = 1
    for line in speech_fi:
      left, right = line.rstrip().split(" ")
      segments = segments + utt + ("-speech-%04d" % count) + " " + utt + " " + left + " " + right + "\n"
      utt2spk = utt2spk + utt + ("-speech-%04d"%count) + " " + utt + ("-speech-%04d"%count) + "\n"
      count += 1
  utt2spk_fi = open(os.path.join(out_dir, "utt2spk"), 'w')
  utt2spk_fi.write(utt2spk)
  segments_fi = open(os.path.join(out_dir, "segments"), 'w')
  segments_fi.write(segments)

if __name__ == "__main__":
    Main()
