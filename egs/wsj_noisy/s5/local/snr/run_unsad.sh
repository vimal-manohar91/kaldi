#!/bin/bash

## Run local/snr/run_unsad_common.sh first!

###############################################################################
## Train large nnet for predicting subband features
###############################################################################

unsad_dir=exp/unsad_whole_data_prep_train_100k_sp
train_data_dir=data/train_azteec_unsad_whole_sp_multi_lessreverb_hires
nnet_dir=exp/nnet3_irm_predictor/nnet_cnn_tdnn_d_bp_vh_train_azteec_sp_unsad_whole_multi_lessreverb

local/snr/run_train_snr_predictor_cnn.sh \
  --targets-scp $train_data_dir/irm_targets.scp \
  --train-data-dir $train_data_dir \
  --dir $nnet_dir \
  --initial-effective-lrate 0.00001 --final-effective-lrate 0.0000001 \
  --relu-dims "1024 1024 512 512 512" \
  --splice-indexes "`seq -s , -11 6` 0 -6,-3,0,1,3 0 -7,0,2" \
  --cnn-layer "--filt-x-dim=6 --filt-y-dim=24 --filt-x-step=2 --filt-y-step=8 --num-filters=256 --pool-x-size=2 --pool-y-size=5 --pool-z-size=1 --pool-x-step=1 --pool-y-step=3 --pool-z-step=1" \
  --cnn-reduced-dim 512 \
  --num-epochs 1 \
  --deriv-weights-scp $unsad_dir/final_vad/deriv_weights_azteec_multi_for_corrupted.scp 

###############################################################################
## Predict subband features for a subset of data
###############################################################################

dataid=train_azteec_unsad_whole_sp_multi_lessreverb
subset_data_dir=data/${dataid}_1k_hires

nnet_dir=exp/nnet3_irm_predictor/nnet_cnn_tdnn_d_bp_vh_train_azteec_sp_unsad_whole_multi_lessreverb
pred_dir=exp/frame_snrs_irm_cnn_tdnn_d_bp_vh_train_azteec_sp_unsad_whole_multi_lessreverb_1k

local/snr/compute_frame_snr.sh --cmd "$train_cmd" --nj 40 --compute-snr false \
  $nnet_dir $subset_data_dir /dev/null $pred_dir

cat $pred_dir/nnet_pred.scp | \
  awk '{print $1" copy-matrix --apply-exp=true "$2" - |"}' > $pred_dir/nnet_pred_exp.scp

###############################################################################
## Train compressed nnet using subset of data
###############################################################################

targets=$pred_dir/nnet_pred_exp.scp
train_data_dir=data/train_azteec_unsad_whole_sp_multi_lessreverb_1k_hires

local/snr/run_train_snr_predictor_cnn.sh \
  --targets-scp $train_data_dir//nnet_pred_exp.scp \
  --train-data-dir $train_data_dir \
  --dir exp/nnet3_irm_predictor/nnet_tdnn_d_bp_vh_train_azteec_sp_unsad_whole_multi_lessreverb \
  --initial-effective-lrate 0.00001 --final-effective-lrate 0.0000001 \
  --relu-dims "1024 512 512 512" \
  --splice-indexes "`seq -s , -3 3` -4,-1,2 -6,-3,3 -7,0,2" \
  --cnn-layer "" --cnn-reduced-dim "" \
  --num-epochs 4 --target-type IrmExp \
  --deriv-weights-scp $unsad_dir/final_vad/deriv_weights_azteec_multi_for_corrupted.scp 

###############################################################################
## Train SAD network
###############################################################################

dataid=data/train_azteec_unsad_whole_sp_multi_lessreverb_1k
pred_dir=exp/frame_snrs_irm_cnn_tdnn_d_bp_vh_train_azteec_sp_unsad_whole_multi_lessreverb_1k

local/snr/create_snr_data_dir.sh \
  --append-to-orig-feats false --add-pov-feature false \
  --add-raw-pov false --type Irm \
  --dataid $dataid data/${dataid}_fbank \
  $pred_dir exp/make_snr_data_dir snr_feats \
  data/${dataid}_snr

train_data_dir=data/${dataid}_snr
unsad_dir=exp/unsad_whole_data_prep_train_100k_sp
nnet_dir=exp/nnet3_sad_snr/lstm_irm_d_bp_ch_train_azteec_sp_unsad_whole_multi_lessreverb_1k_splice5_2

local/snr/run_train_sad_lstm.sh \
  --train-data-dir $train_data_dir --snr-scp $train_data_dir/feats.scp \
  --nj 64 \
  --vad-scp $unsad_dir/reco_vad/vad_azteec_multi.scp \
  --deriv-weights-scp $unsad_dir/final_vad/deriv_weights_azteec_multi_for_corrupted.scp \
  --dir $nnet_dir \
  --max-param-change 0.1 \
  --num-epochs 8 \
  --feat-type raw \
  --splice-indexes "-5,-4,-3,-2,-1,0,1,2"


###############################################################################
## Create segments using the SAD network
###############################################################################

snr_nnet_dir=exp/nnet3_irm_predictor/nnet_cnn_tdnn_d_bp_vh_train_azteec_sp_unsad_whole_multi_lessreverb
sad_nnet_dir=exp/nnet3_sad_snr/lstm_irm_d_bp_ch_train_azteec_sp_unsad_whole_multi_lessreverb_1k_splice5_2
orig_test_data_dir=data/dev
testid=dev

local/snr/run_test.sh \
  --reco-nj 10 \
  --segmentation-config conf/segmentation.conf \
  --weights-segmentation-config conf/weights_segmentation.conf \
  --use-gpu true --do-downsampling true \
  --mfcc-config conf/mfcc_hires_bp.conf \
  --affix a --extra-left-context 40 --feature-type Irm \
  $orig_test_data_dir data/${testid} \
  $snr_nnet_dir $sad_nnet_dir


