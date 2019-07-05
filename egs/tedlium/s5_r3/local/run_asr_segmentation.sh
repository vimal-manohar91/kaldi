#!/bin/bash

# Copyright  2017  Nagendra Kumar Goel
#            2017  Vimal Manohar
# Apache 2.0

# We assume the run.sh has been executed (because we are using model
# directories like exp/tri4a)

# This script demonstrates nnet3-based speech activity detection for
# segmentation.
# This script:
# 1) Prepares targets (per-frame labels) for a subset of training data 
#    using GMM models
# 2) Augments the training data with reverberation and additive noise
# 3) Trains TDNN+Stats or TDNN+LSTM neural network using the targets 
#    and augmented data
# 4) Demonstrates using the SAD system to get segments of dev data and decode

lang=data/lang   # Must match the one used to train the models

data_dir=data/train_cleaned
# Model directory used to align the $data_dir to get target labels for training
# SAD. This should typically be a speaker-adapted system.
model_dir=exp/tri3_cleaned
graph_dir=exp/tri3_cleaned/graph

merge_weights=1.0,0.25,0.25

prepare_targets_stage=-10
nstage=-10
train_stage=-10
test_stage=-10
get_egs_stage=6
num_data_reps=3
affix=_1a   # For segmentation
test_affix=1a
stage=-1
nj=80
reco_nj=40

# test options
test_nj=30
test_stage=1

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi

dir=exp/segmentation${affix}
mkdir -p $dir

# See $lang/phones.txt and decide which should be garbage
garbage_phones="NSN"
silence_phones="SIL"

for p in $garbage_phones; do 
  for a in "" "_B" "_E" "_I" "_S"; do
    echo "$p$a"
  done
done > $dir/garbage_phones.txt

for p in $silence_phones; do 
  for a in "" "_B" "_E" "_I" "_S"; do
    echo "$p$a"
  done
done > $dir/silence_phones.txt

if ! cat $dir/garbage_phones.txt $dir/silence_phones.txt | \
  steps/segmentation/internal/verify_phones_list.py $lang/phones.txt; then
  echo "$0: Invalid $dir/{silence,garbage}_phones.txt"
  exit 1
fi

data_id=$(basename $data_dir)

whole_data_dir=${data_dir}_whole
whole_data_id=$(basename $whole_data_dir)

targets_dir=$dir/$(basename $model_dir)_${whole_data_id}_targets_sub3

rvb_data_dir=${whole_data_dir}_rvb_hires
rvb_targets_dir=${targets_dir}_rvb

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi

###############################################################################
# Extract features for the whole data directory
###############################################################################
if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --nj $reco_nj --cmd "$train_cmd"  --write-utt2num-frames true \
    $whole_data_dir exp/make_mfcc/${whole_data_id}
  steps/compute_cmvn_stats.sh $whole_data_dir exp/make_mfcc/${whole_data_id}
  utils/fix_data_dir.sh $whole_data_dir
fi

if [ $stage -le 3 ]; then
  steps/segmentation/prepare_targets_gmm.sh --stage $prepare_targets_stage \
    --train-cmd "$train_cmd" --decode-cmd "$decode_cmd" --num-threads-decode 4 \
    --nj $nj --reco-nj $reco_nj \
    --garbage-phones-list $dir/garbage_phones.txt \
    --silence-phones-list $dir/silence_phones.txt \
    --merge-weights "$merge_weights" \
    --graph-dir "$graph_dir" \
    $lang $data_dir $whole_data_dir $model_dir $dir
fi

if [ $stage -le 4 ]; then
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  if [ ! -d "RIRS_NOISES" ]; then
    wget -O rirs_noises.zip --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
    rm rirs_noises.zip
  fi

  local/make_mx6.sh /export/corpora/LDC/LDC2013S03/mx6_speech data
  
  local/make_musan.sh /export/corpora/JHU/musan data

  for name in noise music; do
    utils/data/get_reco2dur.sh data/musan_${name}
  done
fi

if [ $stage -le 5 ]; then
  rvb_opts=()
  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  steps/data/augment_data_dir_for_asr.py --utt-prefix "noise" --fg-interval 1 \
    --fg-snrs "20:15:10:5:0" --fg-noise-dir "data/musan_noise" \
    ${whole_data_dir} ${whole_data_dir}_noise || exit 1

  steps/data/augment_data_dir_for_asr.py --utt-prefix "music" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" \
    --bg-noise-dir "data/musan_music" \
    ${whole_data_dir} ${whole_data_dir}_music || exit 1

  steps/data/augment_data_dir_for_asr.py --utt-prefix "babble" \
    --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
    --bg-noise-dir "data/mx6_mic" \
    ${whole_data_dir} ${whole_data_dir}_babble || exit 1

  # corrupt the data to generate multi-condition data
  # for data_dir in train dev test; do
  seed=0
  for name in noise music babble; do 
    python steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --speech-rvb-probability 0.8 \
      --pointsource-noise-addition-probability 0 \
      --isotropic-noise-addition-probability 0 \
      --num-replications 1 \
      --source-sampling-rate 16000 \
      --random-seed $seed \
      ${whole_data_dir}_${name} ${whole_data_dir}_${name}_reverb || exit 1
    seed=$[seed+1]
  done

  utils/combine_data.sh \
    ${rvb_data_dir} \
    ${whole_data_dir}_noise_reverb \
    ${whole_data_dir}_music_reverb \
    ${whole_data_dir}_babble_reverb || exit 1
fi

if [ $stage -le 6 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf \
    --nj $reco_nj ${rvb_data_dir}
  steps/compute_cmvn_stats.sh ${rvb_data_dir}
  utils/fix_data_dir.sh $rvb_data_dir
fi

if [ $stage -le 7 ]; then
    rvb_targets_dirs=()
    for name in noise music babble; do
      steps/segmentation/copy_targets_dir.sh --utt-prefix "rev1_${name}_" \
        $targets_dir ${targets_dir}_temp_${name} || exit 1
      rvb_targets_dirs+=(${targets_dir}_temp_${name})
    done

    steps/segmentation/combine_targets_dirs.sh \
      $rvb_data_dir ${rvb_targets_dir} \
      ${rvb_targets_dirs[@]} || exit 1;

    rm -r ${rvb_targets_dirs[@]}
fi

sad_nnet_dir=exp/segmentation${affix}/tdnn_stats_asr_sad_1a
#sad_nnet_dir=exp/segmentation${affix}/tdnn_lstm_asr_sad_1a
#sad_opts="--extra-left-context 70 --extra-right-context 0 --frames-per-chunk 150 --extra-left-context-initial 0 --extra-right-context-final 0 --acwt 0.3"

if [ $stage -le 8 ]; then
  # Train a STATS-pooling network for SAD
  local/segmentation/tuning/train_stats_asr_sad_1a.sh \
    --stage $nstage --train-stage $train_stage --get-egs-stage $get_egs_stage \
    --targets-dir ${rvb_targets_dir} \
    --data-dir ${rvb_data_dir} --affix "1a" \
    --dir $sad_nnet_dir || exit 1

  # # Train a TDNN+LSTM network for SAD
  # local/segmentation/tuning/train_lstm_asr_sad_1a.sh \
  #   --stage $nstage --train-stage $train_stage \
  #   --targets-dir ${rvb_targets_dir} \
  #   --data-dir ${rvb_data_dir} --affix "1a" || exit 1
fi

if [ $stage -le 9 ]; then
  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
    --reco2file-and-channel=data/how2_dev_hires/reco2file_and_channel \
    data/how2_dev_hires/{utt2spk,segments,ref.rttm}
fi

if [ $stage -le 10 ]; then

fi
