#!/bin/bash

# Copyright  2016  David Snyder
# Apache 2.0.

# TODO This script computes the stopping threshold used in clustering.

# Begin configuration section.
cmd="run.pl"
stage=0
cleanup=true
num_points=0
gmm_calibration_opts=
threshold_stddev=2
calibration_method=kMeans
per_reco=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 <scores-dir> <output-dir>"
  echo " e.g.: $0 exp/ivectors_callhome_heldout exp/ivectors_callhome_test exp/ivectors_callhome_test"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --stage <stage|0>                                # To control partial reruns"
  exit 1;
fi

scores_dir=$1
dir=$2

nj=$(cat $scores_dir/num_jobs) || exit 1

if $per_reco; then
  threshold_wxfilename=ark,t:$dir/thresholds.JOB.ark.txt
else
  threshold_wxfilename=$dir/threshold.JOB.txt
fi

if [ $stage -le 0 ]; then
  echo "$0: Computing calibration thresholds"

  if [ $calibration_method == "kMeans" ]; then
    $cmd JOB=1:$nj $dir/log/compute_calibration.JOB.log \
      compute-calibration --num-points=$num_points ark:$scores_dir/scores.JOB.ark \
      $threshold_wxfilename || exit 1
  elif [ $calibration_method == "SingleGaussian" ]; then
    $cmd JOB=1:$nj $dir/log/compute_calibration.JOB.log \
      compute-calibration-gaussian --threshold-stddev=$threshold_stddev \
      ark:$scores_dir/scores.JOB.ark $threshold_wxfilename || exit 1
  elif [ $calibration_method == "GMM" ]; then
    $cmd JOB=1:$nj $dir/log/compute_calibration_gmm.JOB.log \
      compute-calibration-gmm --num-points=$num_points $gmm_calibration_opts \
      ark:$scores_dir/scores.JOB.ark $threshold_wxfilename || exit 1
  else 
    echo "$0: Unknown calibration-method $calibration_method"
    exit 1
  fi
fi

if [ $stage -le 1 ]; then
  echo "$0: combining calibration thresholds across jobs"
  if $per_reco; then
    for j in $(seq $nj); do cat $dir/thresholds.$j.ark.txt; done >$dir/thresholds_per_reco.ark.txt || exit 1;
    awk '{ sum += $2; n++ } END { if (n > 0) print sum / n; }' $dir/thresholds_per_reco.ark.txt > $dir/threshold.txt
  else
    for j in $(seq $nj); do echo `cat $dir/threshold.$j.txt`; done >$dir/thresholds.txt || exit 1;
    awk '{ sum += $1; n++ } END { if (n > 0) print sum / n; }' $dir/thresholds.txt > $dir/threshold.txt
  fi
fi

if $cleanup; then
  rm -rf $dir/tmp
  for j in $(seq $nj); do rm $dir/threshold.$j.txt 2>/dev/null ; done || true;
  for j in $(seq $nj); do rm $dir/thresholds.$j.ark.txt 2>/dev/null ; done || true
fi
