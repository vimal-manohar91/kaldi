#!/bin/bash

nj=4
cmd=run.pl
num_data_reps=1

. utils/parse_options.sh
. ./path.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <train-data-dir> <original-lat-dir> <lat-dir-ihmdata> <out-lat-dir>"
  exit 1
fi

train_data_dir=$1  # Directory with combined real and perturbed data
original_lat_dir=$2  # For real data (no perturbation)
lat_dir_ihmdata=$3  # For perturbed data
lat_dir=$4

utils/split_data.sh $train_data_dir $nj

$cmd JOB=1:$nj $lat_dir/get_uttlist.JOB.log \
  cat $train_data_dir/split$nj/JOB/utt2spk \| \
  perl -pe "s/rev{`seq -s, 1 $num_data_reps`}_//g" '>' \
  $lat_dir/uttlist.JOB

original_lat_nj=$(cat $original_lat_dir/num_jobs)
ihm_lat_nj=$(cat $lat_dir_ihmdata/num_jobs)

# Copy real data lattices directly
$cmd --max-jobs-run 10 JOB=1:$original_lat_nj $lat_dir/temp/log/copy_original_lats.JOB.log \
  lattice-copy "ark:gunzip -c $original_lat_dir/lat.JOB.gz |" ark,scp:$lat_dir/temp/lats.JOB.ark,$lat_dir/temp/lats.JOB.scp

for n in $(seq $original_lat_nj); do
  cat $lat_dir/temp/lats.$n.scp
done > $lat_dir/temp/combined_lats.scp

# Copy perturbed data lattices with one copy for each data perturbation
$cmd --max-jobs-run 10 JOB=1:$ihm_lat_nj $lat_dir/temp2/log/copy_ihm_lats.JOB.log \
  lattice-copy "ark:gunzip -c $lat_dir_ihmdata/lat.JOB.gz |" ark,scp:$lat_dir/temp2/lats.JOB.ark,$lat_dir/temp2/lats.JOB.scp

for i in `seq 1 $num_data_reps`; do
  for n in $(seq $ihm_lat_nj); do
    cat $lat_dir/temp2/lats.$n.scp
  done | sed -e "s/^/rev${i}_/"
done >> $lat_dir/temp/combined_lats.scp

sort -u $lat_dir/temp/combined_lats.scp > $lat_dir/temp/combined_lats_sorted.scp

# Dump the final real and perturbed data lattices
$cmd --max-jobs-run 10 JOB=1:$nj $lat_dir/copy_combined_lats.JOB.log \
  lattice-copy --include=$lat_dir/uttlist.JOB \
  scp:$lat_dir/temp/combined_lats_sorted.scp \
  "ark:|gzip -c >$lat_dir/lat.JOB.gz" || exit 1;

echo $nj > $lat_dir/num_jobs

# copy other files from original lattice dir
for f in cmvn_opts final.mdl splice_opts tree; do
  cp $original_lat_dir/$f $lat_dir/$f
done
