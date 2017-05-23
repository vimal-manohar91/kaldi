#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

if [ $# -ne 2 ]; then
  echo "This script downsamples a data directory to specific frequency "
  echo "using sox."
  echo "Usage: $0 <frequency> <data-dir>"
  exit 1
fi

freq=$1
dir=$2

sox=`which sox` || { echo "Could not find sox in PATH"; exit 1; }

if [ -f $dir/feats.scp ]; then
  mkdir -p $dir/.backup
  mv $dir/feats.scp $dir/.backup/
  if [ -f $dir/cmvn.scp ]; then
    mv $dir/cmvn.scp $dir/.backup/
  fi
  echo "$0: feats.scp already exists. Moving it to $dir/.backup"
fi

mv $dir/wav.scp $dir/wav.scp.tmp
cat $dir/wav.scp.tmp | python -c "import sys
for line in sys.stdin.readlines():
  splits = line.strip().split()
  if splits[-1] == '|':
    out_line = line.strip() + ' $sox -t wav - -r $freq -c 1 -b 16 -t wav - downsample |'
  else:
    out_line = 'cat {0} {1} | $sox -t wav - -r $freq -c 1 -b 16 -t wav - downsample |'.format(splits[0], ' '.join(splits[1:]))
  print (out_line)" > ${dir}/wav.scp
rm $dir/wav.scp.tmp
