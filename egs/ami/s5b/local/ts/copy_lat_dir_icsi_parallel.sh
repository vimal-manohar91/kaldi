#!/bin/bash

# Copyright 2018 Vimal Manohar
# Apache 2.0

nj=40
cmd=queue.pl
write_compact=false
stage=0
mics="sdm1 sdm2 sdm3 sdm4"

. utils/parse_options.sh

. ./path.sh

src_data=$1
data=$2
src_dir=$3
dir=$4

num_jobs=$(cat $src_dir/num_jobs) || exit 1

mkdir -p $dir

for mic in $mics; do 
  awk '{print $1, $1}' $src_data/utt2spk | \
    sed -e "s/_chan[0-9A-Z]_/_${mic}_/" | \
  awk '{print $2, $1}' >$dir/ihmutt2utt.$mic
done

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
  for mic in $mics; do
    cat $dir/lat_tmp.scp | utils/apply_map.pl -f 1 $dir/ihmutt2utt.$mic
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
