#!/bin/env python3
# Copyright (c) 2019, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0
import sys
import re
import codecs


cmudict = dict()
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith(";;;"):
            continue
        line = re.split(r"[\t ]", line, maxsplit=1);
        word = re.sub(r"\([0-9]\)", "", word)
        word = word.strip()
        pron = line[1].strip()
        pron = re.sub(r"[0-9]", "", pron)
        prons = cmudict.get(word, list())
        prons.append(pron);
        cmudict[word] = prons

out = codecs.getwriter('utf-8')(sys.stdout.buffer);
out_err = codecs.getwriter('utf-8')(sys.stderr.buffer);
for word in codecs.getreader('utf-8')(sys.stdin.buffer):
    word = word.strip()
    if word in cmudict:
        for pron in cmudict[word]:
            print("%s %s" % (word, pron), file=out);
    elif word_lc.startswith('<'):
        print("%s %s" % (word, word), file=out);
    else:
        print(word, file=out_err)

exit(0)
