#!/usr/bin/env python3

import argparse
import re
import sys


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exclude", help="Exclude these tapes",
                        default="868 869 870 883 884 885 886")
    parser.add_argument("--prefix",
                        help="Prefix at the start of the id before the tape number",
                        default="A11_T")
    parser.add_argument("in_list", help="Input list of recordings or wav.scp")
    parser.add_argument("out_list", help="Output list of recordings or wav.scp")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    exclude_list = args.exclude.split()

    num_selected = 0
    num_lines = 0
    with open(args.out_list, 'w') as f:
        for line in open(args.in_list):
            parts = line.strip().split()
            m = re.match(r"{}([0-9]+)".format(args.prefix), parts[0])
            if m is None or len(m.group(1)) != 3:
                raise Exception("Could not parse line {}: got match {}".format(line, m))
            num_lines += 1
            if m.group(1) not in exclude_list:
                num_selected += 1
                print (line.strip(), file=f)

    print ("Selected {} lines out of {}".format(num_selected, num_lines),
           file=sys.stderr)


if __name__ == "__main__":
    main()
