#!/usr/bin/env python3

import argparse
import math
import sys


def get_args():

    parser = argparse.ArgumentParser("""
    This script converts sausages from stdin (output of lattice-mbr-decode etc.)
    into graphs and writes them to stdout as FST archives""")
    parser.add_argument("--min-prob", default=1e-8, type=float,
                        help="Minimum probability of word in the sausage "
                        "for it to be kept")
    parser.add_argument("--min-relative-prob", default=0, type=float,
                        help="Minimum probability of word relative to the "
                        "maximum in the sausage for it to be kept")
    parser.add_argument("sausage_in", type=str, nargs='?',
                        help="Sausage input. If not provided, will read from "
                        "stdin")
    parser.add_argument("--write-lattice", default="false", type=str,
                        choices=["true", "false"],
                        help="If true, then output will be an archive of "
                        "compact lattice with the FST cost treated as the "
                        "graph cost")

    args = parser.parse_args()

    args.write_lattice = True if args.write_lattice == "true" else False

    return args


def main():
    args = get_args()

    if args.sausage_in is not None and args.sausage_in != "-":
        fin = open(args.sausage_in)
    else:
        fin = sys.stdin

    for line in fin.readlines():
        parts = line.strip().split()
        utt_id = parts[0]

        print (utt_id)

        n = 0
        i = 1

        while i < len(parts):
            if parts[i] != "[":
                raise Exception("Did not get start of the sausage in line {0}".format(line.strip()))
            i += 1
            words_in_sausage = []
            tot_prob = 0
            max_prob = 0
            while parts[i] != "]":
                # words in the sausage
                w = int(parts[i])
                p = float(parts[i+1])
                max_prob = max(p, max_prob)
                tot_prob += p
                words_in_sausage.append((w,p))
                i += 2


            words_in_sausage = [ (w, p) for w, p  in words_in_sausage if
                                 p > args.min_prob and p / max_prob > args.min_relative_prob ]
            #new_tot_prob = sum([p for w, p in words_in_sausage])

            if len(words_in_sausage) == 0:
                if args.write_lattice:
                    print ("{0} {1} 0 0 0,0".format(n, n+1))
                else:
                    print ("{0} {1} 0 0 0".format(n, n+1))
            else:
                for w, p in words_in_sausage:
                    if args.write_lattice:
                        print ("{0} {1} {2} {2} {3},0".format(n, n+1, w, -math.log(p / max_prob)))
                    else:
                        print ("{0} {1} {2} {2} {3}".format(n, n+1, w, -math.log(p / max_prob)))
            i += 1
            n += 1

        num_segments = n
        print ("Got {0} words for utterance {1}".format(num_segments, utt_id), file=sys.stderr)
        if args.write_lattice:
            print ("{0}".format(num_segments))
        else:
            print ("{0}".format(num_segments))
        print ("")    # kaldi FST archive format requires separation by newline
    # Loop over utterances

if __name__ == "__main__":
    main()
