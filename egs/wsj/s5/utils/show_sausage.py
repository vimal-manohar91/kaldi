#!/usr/bin/env python3
import sys

wordmap = {}
for line in open(sys.argv[1]).readlines():
    parts = line.strip().split()
    wordmap[parts[1]] = parts[0]


for line in sys.stdin.readlines():
    parts = line.strip().split()

    utt_id = parts[0]

    i = 1

    while i < len(parts):
        assert parts[i] == "["
        i += 1
        while parts[i] != "]":
            if parts[i] not in wordmap:
                raise Exception("{0} not in map".format(parts[i]))
            parts[i] = wordmap[parts[i]]
            i += 2
        i += 1

    print (" ".join([str(x) for x in parts]))
