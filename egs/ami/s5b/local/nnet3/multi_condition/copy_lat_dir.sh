#!/bin/bash

# Copyright 2018 Vimal Manohar
# Apache 2.0

nj=40
cmd=queue.pl
num_data_reps=1
write_compact=false
stage=0

. utils/parse_options.sh

. ./path.sh

if [ $# -ne 3 ]; then
  cat <<EOF
Usage: $0 <data> <src-dir> <dir>
e.g. : $0 data/train exp/tri4_lats_train exp/tri4_lats_train_rvb
EOF
fi

data=$1
src_dir=$2
dir=$3

num_jobs=$(cat $src_dir/num_jobs) || exit 1

if [ $stage -le 1 ]; then
  $cmd JOB=1:$num_jobs $dir/log/copy_lattices.JOB.log \
    lattice-copy --write-compact=$write_compact \
    "ark:gunzip -c $src_dir/lat.JOB.gz |" \
    ark,scp:$dir/lat_tmp.JOB.ark,$dir/lat_tmp.JOB.scp || exit 1

  for n in $(seq $num_jobs); do
    cat $dir/lat_tmp.$n.scp
  done > $dir/lat_tmp.scp
fi

if [ $stage -le 2 ]; then
  for c in $(seq $num_data_reps); do
    cat $dir/lat_tmp.scp | awk -v suff="rev${c}_" '{print suff$0}'
  done | sort -k1,1 > $dir/lat_tmp_copy.scp
fi

utils/data/split_data.sh $data $nj

if [ $stage -le 3 ]; then
  $cmd JOB=1:$nj $dir/log/copy_rvb_lattices.JOB.log \
    lattice-copy --write-compact=$write_compact \
    "scp:utils/filter_scp.pl $data/split$nj/JOB/utt2spk $dir/lat_tmp_copy.scp |" \
    "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1
fi

echo $nj > $dir/num_jobs

for f in frame_subsampling_factor final.mdl; do
  if [ -f $srcdir/$f ]; then
    cp $srcdir/$f $dir
  fi
done
