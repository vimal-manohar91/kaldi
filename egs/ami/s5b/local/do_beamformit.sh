#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0.

. path.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <meetings> <channels-info> <config> <source-dir> <output-dir>"
  exit 1
fi

meetings=$1
channels_info=$2
config=$3
srcdir=$4
odir=$5

which BeamformIt || exit 1

while read line; do 
  BeamformIt \
  -c $channels_info -C $config \
  -i $srcdir \
  -o $odir -s $line
done < $meetings
