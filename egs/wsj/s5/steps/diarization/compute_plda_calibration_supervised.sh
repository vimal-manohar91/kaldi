#! /bin/bash

# Copyright 2016  David Snyder
#           2017  Vimal Manohar
# Apache 2.0.

cmd=run.pl
nj=4
target_energy=0.5 
stage=0
ivector_opts=

. path.sh

set -e 
set -o pipefail

. utils/parse_options.sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 <data> <extractor> <plda-dir> <dir>"
  echo " e.g.: $0 data/train_bn96_spkrid exp/extractor_train_bn96_spkrid_c1024_i128 exp/ivectors_spkrid_train_bn96 exp/supervised_calibration_train_bn96_spkrid"
  exit 1
fi

data=$1
extractor=$2
pldadir=$3
dir=$4

name=`basename $data`

if [ $stage -le 0 ]; then
  rm -r $dir/data/${name}_reco || true
  mkdir -p $dir/data/${name}_reco

  $cmd $dir/get_subsegments.log \
    segmentation-init-from-segments --frame-overlap=0.0 \
    $data/segments ark:- \| \
    segmentation-split-segments --max-segment-length=250 --overlap-length=100 \
    ark:- ark:- \| \
    segmentation-to-segments --frame-overlap=0.0 \
    ark:- ark,t:/dev/null $dir/data/${name}_reco/sub_segments

  utils/data/subsegment_data_dir.sh $data $dir/data/${name}_reco/sub_segments \
    $dir/data/${name}_reco
  awk '{print $1" "$2}' $dir/data/${name}_reco/segments > \
    $dir/data/${name}_reco/utt2spk
  utils/utt2spk_to_spk2utt.pl \
    $dir/data/${name}_reco/utt2spk > $dir/data/${name}_reco/spk2utt

  utils/data/get_utt2num_frames.sh $data

  cat $dir/data/${name}_reco/sub_segments | cut -d ' ' -f 1,2 | \
    utils/apply_map.pl -f 2 $data/utt2num_frames > \
    $dir/data/${name}_reco/utt2max_frames
  
  utils/data/get_subsegmented_feats.sh \
    $data/feats.scp 0.01 0.0 $dir/data/${name}_reco/sub_segments | \
    utils/data/fix_subsegmented_feats.pl \
    $dir/data/${name}_reco/utt2max_frames > \
    $dir/data/${name}_reco/feats.scp

  steps/compute_cmvn_stats.sh $dir/data/${name}_reco
fi

if [ $stage -le 1 ]; then
  steps/diarization/extract_ivectors_nondense.sh --cmd "$cmd --mem 20G" \
    --nj $nj --use-vad false $ivector_opts \
    $extractor \
    $dir/data/${name}_reco $dir/ivectors_dense_spkrid_${name}_reco
fi

if [ $stage -le 2 ]; then
  steps/diarization/score_plda.sh --cmd "$cmd --mem 4G" \
    --nj $nj --target-energy $target_energy $pldadir \
    $dir/ivectors_dense_spkrid_${name}_reco \
    $dir/ivectors_dense_spkrid_${name}_reco/plda_scores
fi

if [ $stage -le 3 ]; then
  utils/split_data.sh $dir/ivectors_dense_spkrid_${name}_reco $nj

  $cmd JOB=1:$nj $dir/log/calibration_trials.JOB.log \
    sample-scores-into-trials \
    scp:$dir/ivectors_dense_spkrid_${name}_reco/plda_scores/scores.scp \
    ark,t:$dir/ivectors_dense_spkrid_${name}_reco/split$nj/JOB/spk2utt \
    "ark,t:cat $dir/data/${name}_reco/sub_segments | cut -d ' ' -f 1,2 | utils/apply_map.pl -f 2 $data/utt2spk |" \
    $dir/trials.JOB
  
  threshold=`for n in $(seq 20); do cat $dir/log/calibration_trials.$n.log | perl -ne 'if (m/optimum-threshold=(\S+)/) { print $1;}'; done | awk '{i+=$1; j++;} END{print i/j}'`

  echo $threshold > $dir/threshold
fi
  threshold=`for n in $(seq 20); do cat $dir/log/calibration_trials.$n.log | perl -ne 'if (m/optimum-threshold=(\S+)/) { print $1;}'; done | awk '{i+=$1; j++;} END{print i/j}'`

  echo $threshold > $dir/threshold


# $cmd $dir/ivectors_dense_spkrid_${name}_reco/calibration_eer.log \
#   compute-eer $dir/trials
# threshold=`cat exp/ivectors_dense_spkrid_${name}_reco/calibration_eer.log | \
#   grep "threshold" | perl -ne 'm/at threshold (\S+)$/; print $1'`
