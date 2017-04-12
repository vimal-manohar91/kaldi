#!/bin/bash

# Copyright 2016  David Snyder
#           2017  Vimal Manohar
# Apache 2.0.
#
# TODO details on what this does.
# See README for more info on the required data.

. cmd.sh
. path.sh

set -e

stage=-10
calibration_stage=-1    # supervised calibration
extractor_stage=-10

exp=exp
dataset=train_bn96
ivector_train_dataset=
diagubm_subset_utts=8000  # subset for training diag UBM
fullubm_subset_utts=  # subset for training full UBM

reco_nj=40
nj=40
mem=35G

suffix=
ivector_suffix=

# Feats options
mfccdir=`pwd`/mfcc_spkrid_16k
vaddir=`pwd`/mfcc_spkrid_16k
mfcc_config=conf/mfcc_spkrid_16k.conf 

use_whole_dir_and_segment=false
cmvn_affix=sliding_cmvn
cmvn_opts="--norm-means=false --norm-vars=false"
use_sliding_cmvn=true
use_perutt_cmvn=false

use_vad=false 
delta_order=1
posterior_scale=1.0
max_count=0

# iVector options
num_components=1024
ivector_dim=128

# All the durations are in seconds
intersegment_duration=5
min_chunk_duration=1.5
max_chunk_duration=10

# PLDA options
target_energy=0.5
train_plda_opts=

. utils/parse_options.sh

mfcc_id=`basename $mfcc_config .conf`

make_mfcc() {
  if [ $# -lt 3 ]; then
    echo "$0: make_mfcc <dataset> <logdir> <mfccdir>"
    exit 1
  fi

  mfcc_config=conf/mfcc.conf
  nj=4
  cmd=run.pl

  for n in `seq 3`; do 
    if [ $1 == "--mfcc-config" ]; then
      mfcc_config=$2
      shift; shift;
    fi
    if [ $1 == "--nj" ]; then
      nj=$2
      shift; shift;
    fi
    if [ $1 == "--cmd" ]; then
      cmd=$2
      shift; shift;
    fi
  done

  dataset=$1
  logdir=$2
  mfccdir=$3
    
  steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $nj \
    --cmd "$cmd" \
    data/${dataset}_spkrid $logdir $mfccdir
  steps/compute_cmvn_stats.sh \
    data/${dataset}_spkrid $logdir $mfccdir
  steps/diarization/compute_cmvn_stats_perutt.sh \
    data/${dataset}_spkrid $logdir $mfccdir
  steps/sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
    data/${dataset}_spkrid $logdir $mfccdir
  utils/fix_data_dir.sh data/${dataset}_spkrid
}

make_sliding_cmvn_mfcc() {
  if [ $# -lt 4 ]; then
    echo "$0: make_mfcc <dataset> <logdir> <mfccdir> <cmvn-affix>"
    exit 1
  fi

  mfcc_config=conf/mfcc.conf
  nj=4
  cmd=run.pl
  cmvn_opts=

  for n in `seq 4`; do 
    if [ $1 == "--mfcc-config" ]; then
      mfcc_config=$2
      shift; shift;
    fi
    if [ $1 == "--nj" ]; then
      nj=$2
      shift; shift;
    fi
    if [ $1 == "--cmd" ]; then
      cmd=$2
      shift; shift;
    fi
    if [ $1 == "--cmvn-opts" ]; then
      cmvn_opts=$2
      shift; shift;
    fi
  done

  dataset=$1
  logdir=$2
  mfccdir=$3
  cmvn_affix=$4

  data_whole=data/${dataset}_whole_spkrid
  utils/data/get_reco2utt.sh data/${dataset}
  utils/split_data.sh --per-reco data/${dataset} $nj
  sdata=data/${dataset}/split${nj}reco

  utils/copy_data_dir.sh data/${dataset} data/${dataset}_${cmvn_affix}
  $cmd JOB=1:$nj $logdir/apply_cmvn_sliding.JOB.log \
    apply-cmvn-sliding ${cmvn_opts} "scp:utils/filter_scp.pl $sdata/JOB/reco2utt $data_whole/feats.scp |" ark:- \| \
    extract-feature-segments ark:- $sdata/JOB/segments ark:- \| copy-feats ark:- \
    ark,scp:$mfccdir/${cmvn_affix}_mfcc_${dataset}.JOB.ark,$mfccdir/${cmvn_affix}_mfcc_${dataset}.JOB.scp
    
  $cmd JOB=1:$nj $logdir/compute_vad.JOB.log \
    extract-feature-segments "scp:utils/filter_scp.pl $sdata/JOB/reco2utt $data_whole/feats.scp |" $sdata/JOB/segments ark:- \| \
    compute-vad --config=conf/vad.conf ark:- \
    ark,scp:$vaddir/vad_${dataset}_${cmvn_affix}.JOB.ark,$vaddir/vad_${dataset}_${cmvn_affix}.JOB.scp
    
  for n in `seq $nj`; do
    cat $mfccdir/${cmvn_affix}_mfcc_${dataset}.$n.scp
  done | \
    sort -k1,1 > data/${dataset}_${cmvn_affix}/feats.scp

  for n in `seq $nj`; do
    cat $vaddir/vad_${dataset}_${cmvn_affix}.$n.scp
  done | \
    sort -k1,1 > data/${dataset}_${cmvn_affix}/vad.scp
  
  steps/compute_cmvn_stats.sh --fake data/${dataset}_${cmvn_affix}
  utils/fix_data_dir.sh data/${dataset}_${cmvn_affix}
}

[ -z "$ivector_train_dataset" ] && ivector_train_dataset=$dataset

if $use_whole_dir_and_segment && $use_sliding_cmvn; then
  if [ $stage -le -1 ]; then
    utils/data/convert_data_dir_to_whole.sh data/${dataset} data/${dataset}_whole_spkrid
    make_mfcc --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
      ${dataset}_whole_spkrid $exp/$mfcc_id/${dataset}_whole $mfccdir

    if [ $ivector_train_dataset != $dataset ]; then
      utils/data/convert_data_dir_to_whole.sh data/${ivector_train_dataset} data/${ivector_train_dataset}_whole_spkrid
      make_mfcc --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
        ${dataset}_whole_spkrid $exp/$mfcc_id/${dataset}_whole $mfccdir
    fi
  fi
  
  if [ $stage -le 0 ]; then
    make_sliding_cmvn_mfcc --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
      ${dataset} $exp/$mfcc_id/${dataset}_${cmvn_affix} $mfccdir $cmvn_affix
    
    if [ $ivector_train_dataset != $dataset ]; then
      make_sliding_cmvn_mfcc --mfcc-config $mfcc_config --nj $reco_nj --cmd "$train_cmd" \
        ${ivector_train_dataset} $exp/$mfcc_id/${ivector_train_dataset}_${cmvn_affix} $mfccdir $cmvn_affix
    fi
  fi
  
  dataset=${dataset}_${cmvn_affix}
  ivector_train_dataset=${ivector_train_dataset}_${cmvn_affix}
  cmvn_opts="--norm-means=false --norm-vars=false"
else
  if [ $stage -le 0 ]; then
    utils/copy_data_dir.sh data/$dataset data/${dataset}_spkrid
    make_mfcc --cmd "$train_cmd" --nj $nj --mfcc-config $mfcc_config \
      $dataset $exp/$mfcc_id/${dataset} $mfccdir
    if [ $ivector_train_dataset != $dataset ]; then
      utils/copy_data_dir.sh data/$ivector_train_dataset \
        data/${ivector_train_dataset}_spkrid
      make_mfcc --cmd "$train_cmd" --nj $nj --mfcc-config $mfcc_config \
        $ivector_train_dataset $exp/$mfcc_id/${ivector_train_dataset} $mfccdir
    fi
  fi
  dataset=${dataset}_spkrid
  ivector_train_dataset=${ivector_train_dataset}_spkrid
fi

name=`basename ${dataset}`

diagubm_subset_suffix=
if [ ! -z "$diagubm_subset_utts" ]; then
  diagubm_subset_suffix="_`echo $diagubm_subset_utts | perl -pe 's/000//g'`k"
fi
fullubm_subset_suffix=
if [ ! -z "$fullubm_subset_utts" ]; then
  fullubm_subset_suffix="_`echo $fullubm_subset_utts | perl -pe 's/000//g'`k"
fi

if [ $stage -le 1 ]; then
  # Reduce the amount of training data for the UBM.
  if [ ! -z "$diagubm_subset_utts" ]; then
    utils/subset_data_dir.sh data/${ivector_train_dataset} $diagubm_subset_utts \
      data/${ivector_train_dataset}${diagubm_subset_suffix}
  fi

  if [ ! -z "$fullubm_subset_utts" ]; then
    utils/subset_data_dir.sh data/${ivector_train_dataset} $fullubm_subset_utts \
      data/${ivector_train_dataset}${fullubm_subset_suffix}
  fi
fi

if [ $stage -le 2 ]; then
  # Train UBM and i-vector extractor.
  steps/diarization/train_diag_ubm.sh --cmd "$train_cmd --mem 20G" \
    --nj 20 --num-threads 8 --delta-order $delta_order --cmvn-opts "$cmvn_opts" \
    --use-sliding-cmvn $use_sliding_cmvn --use-perutt-cmvn $use_perutt_cmvn \
    data/${dataset}${diagubm_subset_suffix} ${num_components} \
    $exp/diag_ubm${suffix}_${name}_${num_components}
fi

if [ $stage -le 3 ]; then
  steps/diarization/train_full_ubm.sh --nj $nj --cmd "$train_cmd --mem 25G" \
    --remove-low-count-gaussians false \
    data/${ivector_train_dataset}${fullubm_subset_suffix} \
    $exp/diag_ubm${suffix}_${name}_${num_components} $exp/full_ubm${suffix}_${name}_${num_components}
fi

if [ $stage -le 4 ]; then
  steps/diarization/train_ivector_extractor.sh \
    --cmd "$train_cmd --mem $mem" --stage $extractor_stage \
    --ivector-dim $ivector_dim --num-iters 5 --posterior-scale $posterior_scale \
    $exp/full_ubm${suffix}_${name}_${num_components}/final.ubm data/${ivector_train_dataset} \
    $exp/extractor${suffix}_${name}_c${num_components}_i${ivector_dim}
fi

ivectors_dir=$exp/ivectors${suffix}_spkrid_i${ivector_dim}_${name}${ivector_suffix}
if $use_vad; then
  ivectors_dir=${ivectors_dir}_vad
fi

if [ $stage -le 5 ]; then
  utils/fix_data_dir.sh data/${dataset}
  rm -r data/$dataset/split* || true

  mkdir -p $ivectors_dir/${name}_random_chunks

  steps/segmentation/get_random_chunks.py --intersegment-duration=$intersegment_duration \
    --min-chunk-duration=$min_chunk_duration --max-chunk-duration=$max_chunk_duration \
    data/${dataset}/segments $ivectors_dir/${name}_random_chunks/random_subsegments
  utils/data/subsegment_data_dir.sh data/${dataset} \
    $ivectors_dir/${name}_random_chunks/random_subsegments \
    $ivectors_dir/${name}_random_chunks
  steps/diarization/compute_cmvn_stats_perutt.sh --nj $nj --cmd "$train_cmd" \
    $ivectors_dir/${name}_random_chunks
fi

if [ $stage -le 6 ]; then
  steps/diarization/extract_ivectors_nondense.sh --cmd "$train_cmd --mem 10G" \
    --nj $nj --use-vad $use_vad --posterior-scale $posterior_scale --max-count $max_count \
    $exp/extractor${suffix}_${name}_c${num_components}_i${ivector_dim} \
    $ivectors_dir/${name}_random_chunks $ivectors_dir
fi

if [ $stage -le 7 ]; then
  steps/diarization/train_plda.sh --cmd "$train_cmd" $train_plda_opts \
    ${ivectors_dir}/${name}_random_chunks $ivectors_dir $ivectors_dir
fi

if [ $stage -le 8 ]; then
  steps/diarization/compute_plda_calibration_supervised_simple.sh \
    --target-energy $target_energy --cmd "$train_cmd" --nj 20 \
    --stage $calibration_stage $ivectors_dir \
    $ivectors_dir \
    $exp/supervised_calibration${suffix}_i${ivector_dim}_${name}
fi

exit 0
