#!/bin/bash

## Run local/snr/run_unsad_common.sh first!

###############################################################################
## Train SAD network
###############################################################################

train_data_dir=data/train_azteec_unsad_music_whole_sp_multi_lessreverb_hires
unsad_dir=exp/unsad_whole_data_prep_train_100k_sp

## Modify the network configs by modifying the parameters in the file
local/snr/run_train_sad_jesus.sh \
  --train-data-dir $train_data_dir \
  --vad-scp $unsad_dir/reco_vad/vad_azteec_multi.scp \
  --deriv-weights-scp $unsad_dir/final_vad/deriv_weights_azteec_multi.scp \
  --minibatch-size 512 

###############################################################################
## Create segments using the SAD network
###############################################################################

nnet_dir=exp/nnet3_sad_snr/nnet_raw_a
orig_test_data_dir=data/dev
testid=dev

local/snr/run_test_indomain.sh \
  --reco-nj 10 \
  --segmentation-config conf/segmentation.conf \
  --weights-segmentation-config conf/weights_segmentation.conf \
  --use-gpu true --do-downsampling true \
  --mfcc-config conf/mfcc_hires_bp.conf \
  --sad-nnet-iter final \
  $orig_test_data_dir data/${testid} $nnet_dir
