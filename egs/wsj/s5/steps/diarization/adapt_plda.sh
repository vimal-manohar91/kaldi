#! /bin/bash

# Copyright 2017 Vimal Manohar
# Apache 2.0.

cmd=run.pl
adaptation_opts=
labels_ark=
method=UnsupervisedKaldi
smoothing=0.1

. path.sh

set -e -o pipefail

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <plda-in-dir> <ivector-dir> <plda-out-dir>"
  echo " e.g.: $0 exp/ivectors_s0.3_spkrid_i400_sre_8k exp/diarization/diarization_o_cp_s0.3_retry_i400_s0.3_mc0_eval97.seg_lstm_sad_music_1e/ivectors_perspk_spkrid_eval97.seg_lstm_sad_music_1e_lp_cp_s0.3 exp/diarization/diarization_o_cp_s0.3_retry_i400_s0.3_mc0_eval97.seg_lstm_sad_music_1e/ivectors_perspk_spkrid_eval97.seg_lstm_sad_music_1e_lp_cp_s0.3/plda_adapt_sre"
  exit 1
fi

plda_in_dir=$1
ivecdir=$2
plda_out_dir=$3

for f in $plda_in_dir/plda $plda_in_dir/transform.mat $ivecdir/mean.vec; do
  [ ! -f $f ] && echo "$0: Could not find file $f" && exit 1
done

mkdir -p $plda_out_dir

per_spk=false
if [ -f $ivecdir/ivector_spk.scp ]; then
  per_spk=true
else
  [ ! -f $ivecdir/ivector.scp ] && echo "$0: Could not find file $ivecdir/ivector.scp" && exit 1
fi

if [ $method == "UnsupervisedKaldi" ]; then
  if $per_spk; then
    ivectors="ark:copy-vector scp:$ivecdir/ivector_spk.scp ark:- |"
  else
    ivectors="ark:copy-vector scp:$ivecdir/ivector.scp ark:- |"
  fi

  ivectors="$ivectors ivector-subtract-global-mean $ivecdir/mean.vec ark:- ark:- | transform-vec $plda_in_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"

  $cmd $plda_out_dir/log/adapt_plda.log \
    ivector-adapt-plda $adaptation_opts \
    $plda_in_dir/plda "$ivectors" $plda_out_dir/plda

  cp $plda_in_dir/transform.mat $plda_out_dir
  cp $ivecdir/mean.vec $plda_out_dir
elif [ $method == "SupervisedEMInit" ]; then
  if [ -z "$labels_ark" ]; then
    echo "--labels-ark is required for --method $method"
    exit 1
  fi

  for f in $labels_ark; do
    [ ! -f $f ] && echo "$0: Could not find file $f" && exit 1
  done

  if $per_spk; then
    ivectors="ark:utils/filter_scp.pl $labels_ark $ivecdir/ivector_spk.scp | copy-vector scp:- ark:- |"
  else
    ivectors="ark:utils/filter_scp.pl $labels_ark $ivecdir/ivector.scp | copy-vector scp:- ark:- |"
  fi

  ivectors="$ivectors ivector-subtract-global-mean $ivecdir/mean.vec ark:- ark:- | transform-vec $plda_in_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
  
  $cmd $plda_out_dir/log/adapt_plda.log \
    ivector-compute-plda $adaptation_opts \
    --plda-in=$plda_in_dir/plda "ark,t:utils/utt2spk_to_spk2utt.pl $labels_ark |" \
    "$ivectors" - \| \
    ivector-copy-plda --smoothing=$smoothing - $plda_out_dir/plda

  cp $plda_in_dir/transform.mat $plda_out_dir
  cp $ivecdir/mean.vec $plda_out_dir
elif [ $method == "SupervisedInterpolateParams" ]; then
  if [ -z "$labels_ark" ]; then
    echo "--labels-ark is required for --method $method"
    exit 1
  fi

  for f in $labels_ark; do
    [ ! -f $f ] && echo "$0: Could not find file $f" && exit 1
  done

  if $per_spk; then
    ivectors="ark:utils/filter_scp.pl $labels_ark $ivecdir/ivector_spk.scp | copy-vector scp:- ark:- |"
  else
    ivectors="ark:utils/filter_scp.pl $labels_ark $ivecdir/ivector.scp | copy-vector scp:- ark:- |"
  fi

  ivectors="$ivectors ivector-subtract-global-mean $ivecdir/mean.vec ark:- ark:- | transform-vec $plda_in_dir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"
  
  $cmd $plda_out_dir/log/estimate_plda.log \
    ivector-compute-plda \
    "ark,t:utils/utt2spk_to_spk2utt.pl $labels_ark |" \
    "$ivectors" - \| \
    ivector-copy-plda --smoothing=$smoothing - $plda_out_dir/plda_self

  $cmd $plda_out_dir/log/interpolate_plda.log \
    ivector-sum-plda $adaptation_opts \
    $plda_in_dir/plda $plda_out_dir/plda_self $plda_out_dir/plda

  cp $plda_in_dir/transform.mat $plda_out_dir
  cp $ivecdir/mean.vec $plda_out_dir
else
  echo "$0: Unknown method $method" 
  exit 1
fi
