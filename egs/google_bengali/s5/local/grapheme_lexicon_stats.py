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

output_charset = set()
input_charset = set()

datain = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
for line in datain:
    line = line.strip()
    if not line:
        continue
    record = line.split(' ')
    input_charset = input_charset | set(record[0])
    output_charset = output_charset | set(record[1:-1])


print('Output charset:')
output_charset = list(output_charset)
output_charset.sort()
for elem in output_charset:
    if len(elem) == 1:
        try:
            print(' '.join((elem, "=", unicodedata.name(elem[0]),)).encode('utf-8'));
        except ValueError:
            print(' '.join(('\\x', "{0}".format(ord(elem)), "=", 'UNDEFINED NAME' ,)).encode('utf-8'));
    else:
        print(' '.join((elem, "=", '/compound/',)).encode('utf-8'));

print('')
print('Input charset:')
input_charset = list(input_charset)
input_charset.sort()
for elem in input_charset:
    try:
        print(' '.join((elem, "=", unicodedata.name(elem[0]),)).encode('utf-8'));
    except ValueError:
        print(' '.join(('\\x', "{0}".format(ord(elem)), "=", 'UNDEFINED NAME',)).encode('utf-8'));
