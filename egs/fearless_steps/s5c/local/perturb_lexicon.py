#!/bin/env python3
# Copyright (c) 2019, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0
import sys
import re
import codecs
import random

casing = [str.upper, str.lower]
cmudict = dict()
with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith(";;;"):
            continue
        line = re.split(r"[\t ]", line, maxsplit=1);
        word = line[0].lower()
        word = re.sub(r"\([0-9]\)", "", word)
        word = word.strip()
        word = ''.join(random.choice(casing)(letter) for letter in word)
        pron = line[1].strip()
        pron = re.sub(r"[0-9]", "", pron)
        prons = cmudict.get(word, list())
        prons.append(pron);
        cmudict[word] = prons

out = codecs.getwriter('utf-8')(sys.stdout.buffer);
for word in cmudict:
    word = word.strip()
    for pron in cmudict[word]:
        print("%s %s" % (word, pron), file=out);

exit(0)

