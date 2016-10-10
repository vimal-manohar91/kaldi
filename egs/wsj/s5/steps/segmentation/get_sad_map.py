#! /usr/bin/env python

import sys
import argparse

class StrToBoolAction(argparse.Action):
    """ A custom action to convert bools from shell format i.e., true/false
        to python format i.e., True/False """
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            if values == "true":
                setattr(namespace, self.dest, True)
            elif values == "true":
                setattr(namespace, self.dest, False)
            else:
                raise ValueError
        except ValueError:
            raise Exception("Unknown value {0} for --{1}".format(values, self.dest))

class NullstrToNoneAction(argparse.Action):
    """ A custom action to convert empty strings passed by shell
        to None in python. This is necessary as shell scripts print null strings
        when a variable is not specified. We could use the more apt None
        in python. """
    def __call__(self, parser, namespace, values, option_string=None):
            if values.strip() == "":
                setattr(namespace, self.dest, None)
            else:
                setattr(namespace, self.dest, values)

def GetArgs():
    parser = argparse.ArgumentParser(description = "Get map for SAD")

    parser.add_argument("--init-sad-map", type=str, action=NullstrToNoneAction,
                        help = "Initial SAD map that will be modified based on map-noise-to-sil and map-unk-to-speech")

    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument("--noise-phones-file", type=str, action=NullstrToNoneAction,
                             help = "Map noise phones from file to noise")
    noise_group.add_argument("--noise-phones-list", type=str, action=NullstrToNoneAction,
                             help = "Map noise phones from file to noise")

    parser.add_argument("--map-noise-to-sil", type=str,
                        action=StrToBoolAction,
                        choices=["true","false"], default = False,
                        help = "Map noise phones to silence")

    parser.add_argument("--map-unk-to-speech", type=str,
                        action=StrToBoolAction,
                        choices=["true","false"], default = False,
                        help = "Map UNK phone to speech")
    parser.add_argument("--unk", type=str, action=NullstrToNoneAction,
                        help = "UNK phone")

    parser.add_argument("lang_dir")

    args = parser.parse_args()

    return args

def Main():
    args = GetArgs()

    sad_map = {}

    for line in open('{0}/phones/nonsilence.txt'.format(args.lang_dir)):
        parts = line.strip().split()
        sad_map[parts[0]] = 1;

    for line in open('{0}/phones/silence.txt'.format(args.lang_dir)):
        parts = line.strip().split()
        sad_map[parts[0]] = 0;

    if (args.init_sad_map is not None):
        for line in open(args.init_sad_map):
            parts = line.strip().split()
            try:
                sad_map[parts[0]] = int(parts[1])
            except Exception:
                raise Exception("Invalid line " + line)

    if (args.unk is not None):
        sad_map[args.unk] = 3

    noise_phones = {}
    if (args.noise_phones_file is not None):
        for line in open(noise_phones_file):
            parts = line.strip().split()
            noise_phones[parts[0]] = 1

    if (args.noise_phones_list is not None):
        for x in args.noise_phones_list.split(":"):
            noise_phones[x] = 1

    for x,l in sad_map.iteritems():
        if l == 2 and args.map_noise_to_sil:
            l = 0
        if l == 3 and args.map_unk_to_speech:
            l = 1
        print ("{0} {1}".format(x, l))

if __name__ == "__main__":
    Main()

