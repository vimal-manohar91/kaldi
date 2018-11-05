#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2018, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

import icu
import codecs
import sys
import re
import io
import unicodedata

bengali2latin = icu.Transliterator.createInstance('Bengali-Latin')

datain = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

for line in datain:
    line = line.strip()
    if not line:
        continue
    word = line
    line = unicodedata.normalize('NFKC', line)
    line = re.sub("[- ̔‘’' \"“”|\ue04d\u200d\u200c\x94\x93/?!%& ̄]", " ", line)
    line = line.strip()
    line = line.lower()

    line = bengali2latin.transliterate(line)
    line = unicodedata.normalize('NFKC', line)
    if re.search('[0-9]', line):
        continue

    line = re.sub("[.\ue04d ̄ ̐ ̔ ̥]", "", line)

    line = list(line)
    pron = list()
    
    for i in range(len(line)):
        line[i] = line[i].strip()
        if not line[i]:
            continue
    
        if (i > 0) and (line[i] in ["’", "'", ' ̥']):
            pron[-1] += line[i]
        else:
            pron.append(line[i])

    if len(pron) < 1 :
        continue
    pron = ' '.join(pron);
    entry = "{0}  {1}\n".format(word, pron).encode('utf-8')
    sys.stdout.buffer.write(entry)
