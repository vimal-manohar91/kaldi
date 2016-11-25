#! /usr/bin/env python

from __future__ import print_function
import sys, os, glob, argparse

def GetArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--use-vocals", type=str, default = "false",
                        choices = ["true", "false"],
                        help = "If true, also add music with vocals in the output music-set-parameters")
    parser.add_argument("root_dir", type=str,
                        help = "Root directory of MUSAN corpus")
    parser.add_argument("music_list", type=str,
                        help = "Convert music list into noise-set-paramters for steps/data/reverberate_data_dir.py")

    args = parser.parse_args()

    args.use_vocals = True if args.use_vocals == "true" else False
    return args

def ReadVocals(annotations):
    vocals = {}
    for line in open(annotations):
        parts = line.strip().split()
        if parts[2] == "Y":
            vocals[parts[0]] = True
    return vocals

def WriteMusic(utt, file_path, music_list):
    print ('{utt} {file_path}'.format(
        utt = utt,
        file_path = file_path), file = music_list)

def PrepareMusicSet(root_dir, use_vocals, music_list):
    vocals = {}
    music_dir = os.path.join(root_dir, "music")
    for root, dirs, files in os.walk(music_dir):
        if os.path.exists(os.path.join(root, "ANNOTATIONS")):
            vocals = ReadVocals(os.path.join(root, "ANNOTATIONS"))

        for f in files:
            file_path = os.path.join(root, f)
            if f.endswith(".wav"):
                utt = str(f).replace(".wav", "")
                if not use_vocals and utt in vocals:
                    continue
                WriteMusic(utt, file_path, music_list)
    music_list.close()

def Main():
    args = GetArgs()

    PrepareMusicSet(args.root_dir, args.use_vocals,
                    open(args.music_list, 'w'))

if __name__ == '__main__':
    Main()

