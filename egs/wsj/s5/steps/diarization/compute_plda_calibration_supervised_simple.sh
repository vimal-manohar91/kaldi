#! /bin/bash

# Copyright 2016  David Snyder
#           2017  Vimal Manohar
# Apache 2.0.

cmd=run.pl
nj=4
target_energy=0.5 
stage=0
sampling_opts=

. path.sh

set -e 
set -o pipefail

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <plda-dir> <ivectors-dir> <dir>"
  echo " e.g.: $0 exp/extractor_train_bn96_spkrid_c1024_i128 exp/ivectors_spkrid_train_bn96 exp/supervised_calibration_train_bn96_spkrid"
  exit 1
fi

pldadir=$1
ivectors_dir=$2
dir=$3

if [ $stage -le 2 ]; then
  steps/diarization/score_plda.sh --cmd "$cmd --mem 4G" \
    --nj $nj --target-energy $target_energy --per-spk false $pldadir \
    $ivectors_dir $ivectors_dir/plda_scores
fi

if [ $stage -le 3 ]; then
  mkdir -p $dir/tmp/
  
  splits=
  for n in `seq $nj`; do 
    mkdir -p $dir/tmp/
    splits="$splits $dir/tmp/reco2utt.$n.$nj"
  done
  utils/split_scp.pl $ivectors_dir/plda_scores/reco2utt $splits

  $cmd JOB=1:$nj $dir/log/calibration_trials.JOB.log \
    sample-scores-into-trials $sampling_opts \
    scp:$ivectors_dir/plda_scores/scores.scp \
    ark,t:$dir/tmp/reco2utt.JOB.$nj \
    ark,t:$ivectors_dir/utt2spk \
    $dir/trials.JOB.$nj
fi

threshold=`for n in $(seq 20); do cat $dir/log/calibration_trials.$n.log | perl -ne 'if (m/optimum-threshold=(\S+)/) { print $1;}'; done | awk '{i+=$1; j++;} END{print i/j}'`
echo $threshold > $dir/threshold.txt
