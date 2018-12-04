#!/bin/bash

stage=0
nj=80
sad_stage=-5

. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

set -e -o pipefail -u

if [ $stage -le 0 ]; then
  local/mixer6/make_mx6_calls.pl /export/common/data/corpora/LDC/LDC2013S03 data/local/mx6

  for mic in 02 04 05 06 07 08 09 10 11 12 13; do
    local/mixer6/make_mx6_mic.pl /export/common/data/corpora/LDC/LDC2013S03 $mic data/local/mx6
  done

  utils/combine_data.sh data/local/mx6/mx6_mic_04_to_13 \
    data/local/mx6/mx6_mic_{04,05,06,07,08,09,10,11,12,13}
fi

if [ $stage -le 1 ]; then
  utils/data/get_reco2dur.sh \
    --cmd "$train_cmd" --nj 32 \
    data/local/mx6/mx6_mic_04_to_13
fi

if [ $stage -le 2 ]; then
  utils/fix_data_dir.sh data/local/mx6/mx6_mic_04_to_13

  utils/copy_data_dir.sh data/local/mx6/mx6_mic_04_to_13 \
    data/local/mx6/mx6_mic_04_to_13_filtered || true

  utils/filter_scp.pl data/local/mx6/mx6_mic_04_to_13/reco2dur \
    data/local/mx6/mx6_mic_04_to_13/wav.scp > data/local/mx6/mx6_mic_04_to_13_filtered/wav.scp

  utils/fix_data_dir.sh \
    data/local/mx6/mx6_mic_04_to_13_filtered

  utils/subset_data_dir.sh \
    data/local/mx6/mx6_mic_04_to_13_filtered 2000 \
    data/local/mx6/mx6_mic_04_to_13_2k
fi

if [ $stage -le 3 ]; then
  utils/combine_data.sh data/mx6_mic \
    data/local/mx6/mx6_mic_02 data/local/mx6/mx6_mic_04_to_13_2k

  utils/copy_data_dir.sh data/local/mx6/mx6_mic_04_to_13_2k data/mx6_mic_04_to_13_2k

  utils/copy_data_dir.sh data/local/mx6/mx6_calls data/mx6_calls

  utils/copy_data_dir.sh data/local/mx6/mx6_mic_02 data/mx6_mic_02
fi

if [ $stage -le 4 ]; then
  if [ ! -d sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a/ ]; then
    mkdir -p sad_model
    (
    cd sad_model
    wget http://kaldi-asr.org/models/0004_tdnn_stats_asr_sad_1a.tar.gz
    tar -xzvf 0004_tdnn_stats_asr_sad_1a.tar.gz
    )
  fi
fi

if false && [ $stage -le 4 ]; then
 steps/segmentation/detect_speech_activity.sh --stage $sad_stage \
   --cmd "$train_cmd" --nj $nj --convert-data-dir-to-whole true \
   --extra-left-context 79 --extra-right-context 21 \
   --extra-left-context-initial 0 --extra-right-context-final 0 \
   --frames-per-chunk 150 --mfcc-config sad_model/conf/mfcc_hires.conf \
   data/mx6_mic sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   sad_model/mfcc_hires sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   data/mx6_mic_1a

 for f in data/mx6_mic_1a_seg/{utt2spk,feats.scp}; do
   if [ ! -f $f ]; then
     echo "$0: Could not find $f"
     exit 1
   fi
 done
fi

if [ $stage -le 5 ]; then
  utils/copy_data_dir.sh data/local/mx6/mx6_mic_02 data/mx6_mic_02

  steps/segmentation/detect_speech_activity.sh --stage $sad_stage \
   --cmd "$train_cmd" --nj $nj --convert-data-dir-to-whole true \
   --extra-left-context 79 --extra-right-context 21 \
   --extra-left-context-initial 0 --extra-right-context-final 0 \
   --frames-per-chunk 150 --mfcc-config sad_model/conf/mfcc_hires.conf \
   data/mx6_mic_02 sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   sad_model/mfcc_hires sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   data/mx6_mic_02_1a
fi
