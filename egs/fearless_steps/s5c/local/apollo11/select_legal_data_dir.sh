#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir>"
  exit 1
fi

indir=$1
outdir=$2

utils/copy_data_dir.sh $indir $outdir

local/apollo11/filter_recordings.py $indir/wav.scp $outdir/wav.scp || exit 1

utils/fix_data_dir.sh $outdir
