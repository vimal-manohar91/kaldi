#!/bin/bash

# Copyright 2018 Vimal Manohar
# Apache 2.0

cmd=queue.pl
nj=40
max_jobs_run=30

. ./path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <src-data> <data> <src-lat-dir> <lat-dir>"
  echo " e.g.: $0 "
  exit 1
fi

src_data=$1
data=$2
src_lat_dir=$3
lat_dir=$4

src_lat_nj=$(cat $src_lat_dir/num_jobs)

$train_cmd --max-jobs-run $max_jobs_run JOB=1:$src_lat_nj \
  $lat_dir/temp/log/copy_src_lats.JOB.log \
  lattice-copy "ark:gunzip -c $src_lat_dir/lat.JOB.gz |" \
  ark,scp:$lat_dir/temp/lats.JOB.ark,$lat_dir/temp/lats.JOB.scp

utils/copy_data_dir.sh $src_data $data

cat $src_data/utt2spk | perl -pe 's/^(\S+(_SDM\S+)) (\S+)$/$1 $3$2/' > \
  $data/utt_map

rm $data/spk2utt

for f in utt2spk feats.scp segments text utt2dur utt2num_frames; do
  if [ -f $src_data/$f ]; then
    cat $src_data/$f | utils/apply_map.pl -f 1 $data/utt_map > \
      $data/$f
  fi
done

utils/fix_data_dir.sh $data
utils/split_data.sh $data $nj

for n in $(seq $src_lat_nj); do
  cat $lat_dir/temp/lats.$n.scp
done | utils/apply_map.pl -f 1 $data/utt_map | sort -k1,1 > \
  $lat_dir/temp/combined_lats.scp

$train_cmd --max-jobs-run $max_jobs_run JOB=1:$nj $lat_dir/log/copy_combined_lats.JOB.log \
  lattice-copy --include=$data/split$nj/JOB/utt2spk \
  scp:$lat_dir/temp/combined_lats.scp \
  "ark:|gzip -c >$lat_dir/lat.JOB.gz" || exit 1;

echo $nj > $lat_dir/num_jobs

# copy other files from original lattice dir
for f in cmvn_opts final.mdl splice_opts tree; do
  cp $src_lat_dir/$f $lat_dir/$f
done

rm -r $lat_dir/temp
