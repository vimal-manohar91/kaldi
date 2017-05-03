#!/bin/bash

# Copyright 2016  David Snyder
#           2017  Vimal Manohar
# Apache 2.0.
#
# See README for more info on the required data.

. cmd.sh
. path.sh

set -e

mfccdir=`pwd`/mfcc_spkrid_16k
vaddir=`pwd`/mfcc_spkrid_16k
num_components=2048
ivector_dim=128

suffix=
ivector_suffix=

mic=mdm8
stage=1

. utils/parse_options.sh

base_mic=$(echo $mic | sed 's/[0-9]//g') # sdm, ihm or mdm
nmics=$(echo $mic | sed 's/[a-z]//g') # e.g. 8 for mdm8.

set -euo pipefail

if [ $stage -le -1 ]; then
  local/ami_text_prep.sh data/local/downloads
fi

# Path where AMI gets downloaded (or where locally available):
AMI_DIR=$PWD/wav_db # Default,
case $(hostname -d) in
  fit.vutbr.cz) AMI_DIR=/mnt/scratch05/iveselyk/KALDI_AMI_WAV ;; # BUT,
  clsp.jhu.edu) AMI_DIR=/export/corpora4/ami/amicorpus ;; # JHU,
  cstr.ed.ac.uk) AMI_DIR= ;; # Edinburgh,
esac

# Download AMI corpus, You need around 130GB of free space to get whole data ihm+mdm,
if [ $stage -le 0 ]; then
  if [ -d $AMI_DIR ] && ! touch $AMI_DIR/.foo 2>/dev/null; then
    echo "$0: directory $AMI_DIR seems to exist and not be owned by you."
    echo " ... Assuming the data does not need to be downloaded.  Please use --stage 1 or more."
    exit 1
  fi
  if [ -e data/local/downloads/wget_$mic.sh ]; then
    echo "data/local/downloads/wget_$mic.sh already exists, better quit than re-download... (use --stage N)"
    exit 1
  fi
  local/ami_download.sh $mic $AMI_DIR
fi

if [ "$base_mic" == "mdm" ]; then
  PROCESSED_AMI_DIR=$AMI_DIR/beamformed
  if [ $stage -le 1 ]; then
    # for MDM data, do beamforming
    ! hash BeamformIt && echo "Missing BeamformIt, run 'cd ../../../tools/; make beamformit;'" && exit 1
    local/ami_beamform.sh --cmd "$train_cmd" --nj 20 $nmics $AMI_DIR $PROCESSED_AMI_DIR
  fi
else
  PROCESSED_AMI_DIR=$AMI_DIR
fi

# Prepare original data directories data/ihm/train_orig, etc.
if [ $stage -le 2 ]; then
  local/ami_${base_mic}_data_prep.sh $PROCESSED_AMI_DIR $mic
  local/ami_${base_mic}_scoring_data_prep.sh $PROCESSED_AMI_DIR $mic dev
  local/ami_${base_mic}_scoring_data_prep.sh $PROCESSED_AMI_DIR $mic eval
fi
 
if [ $stage -le 3 ]; then
  local/ami_ihm_data_prep.sh $AMI_DIR ihm
  local/ami_ihm_scoring_data_prep.sh $AMI_DIR ihm dev
  local/ami_ihm_scoring_data_prep.sh $AMI_DIR ihm eval
fi

if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
  utils/create_split_dir.pl \
   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$mfccdir/storage $mfccdir/storage
fi
if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $vaddir/storage ]; then
  utils/create_split_dir.pl \
   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5b/$vaddir/storage $vaddir/storage
fi

if [ $stage -le 4 ]; then
  if [ "$mic" != "ihm" ]; then
    local/prepare_oraclespk_datadir.sh --dataset train --mic $mic
    local/prepare_oraclespk_datadir.sh --dataset dev --mic $mic
    local/prepare_oraclespk_datadir.sh --dataset eval --mic $mic
  fi
fi

if [ $stage -le 5 ]; then
  steps/diarization/prepare_diarization.sh --dataset mdm8/train_oraclespk --exp exp/$mic \
    --suffix $suffix --ivector-suffix $ivector_suffix \
    --mfccdir $mfccdir --vaddir $mfccdir --use-vad true --delta-order 1 \
    --num-components $num_components --ivector-dim $ivector_dim \
    --posterior-scale 1.0 --max-count 0 \
    --min-chunk-duration 1.5 --max-chunk-duration 3 \
    --intersegment-duration 17 --train-plda-opts "--utts-per-spk-min 2"
fi

if [ $stage -le 6 ]; then
  local/prepare_oraclesad_datadir.sh --dataset dev --mic $mic
fi

if [ $stage -le 7 ]; then
  steps/diarization/do_diarization_datadir.sh \
    --cmd "$train_cmd" --per-spk true \
    --use-src-mean true --use-src-transform false \
    --get-uniform-subsegments true --do-change-point-detection false \
    --stage -10 \
    --mfcc-config conf/mfcc_spkrid_16k.conf \
    --mfcc-config-cp conf/mfcc.conf \
    --mfccdir $mfccdir \
    --change-point-merge-opts "--use-full-covar --distance-metric=bic --threshold=2.0 --statistics-scale=1.0" \
    --cp-suffix _cp_th2.0 \
    --distance-threshold "" \
    --get-whole-data-and-segment false \
    --sliding-cmvn-opts "--norm-means --norm-vars=false --center" \
    --cluster-method plda-avg-scores --calibration-method kMeans \
    --target-energy 0.5 \
    --ivector-opts "--posterior-scale 0.3 --max-count 0" \
    --reco-nj 18 --nj 80 \
    --plda-suffix "" \
    --use-vad true \
    data/$mic/dev_oraclesad \
    exp/$mic/extractor${suffix}_train_oraclespk_spkrid_c${num_components}_i${ivector_dim} \
    exp/$mic/ivectors${suffix}_spkrid_i${ivector_dim}_train_oraclespk_spkrid${ivector_suffix}_vad \
    exp/$mic/diarization/diarization${suffix}_spkrid_train_oraclespk_i${ivector_dim}_dev_oraclesad{,/dev_oraclesad.diarized}
fi

exit 0
