#!/usr/bin/env python3
from __future__ import division
from __future__ import print_function
from math import cos, sqrt
import sys


from optparse import OptionParser


def print_on_same_line(text):
    print(text, end=' ')


parser = OptionParser()
parser.add_option('--input-dim', dest='idim', help='input dimension')
parser.add_option('--output-dim', dest='odim', help='output dimension')
(options, args) = parser.parse_args()

if(options.idim is None) or (options.odim is None):
    parser.print_help()
    sys.exit(1)

idim = int(options.idim)
odim = int(options.odim)


# generate the DCT matrix
PI = 3.1415926535897932384626433832795
M_SQRT2 = 1.4142135623730950488016887

print('[')
for i in range(odim):
    for j in range(idim):
        x = cos((PI/idim) * (i + 0.5) * j)
        norm =  sqrt(2/idim)
        if (j == 0):
            norm = sqrt(1/idim) 
        print("%0.3f" % (x), end=' ')
    print()
print(']')
