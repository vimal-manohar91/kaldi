#!/bin/bash

stage=0
nj=80
sad_stage=-5

. utils/parse_options.sh

. ./path.sh
. ./cmd.sh

if [ $stage -le 0 ]; then
  # Prepare SWBD corpora.
  local/make_swbd2_phase1.pl /export/common/data/corpora/LDC/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/common/data/corpora/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/common/data/corpora/LDC/LDC2002S06 \
    data/swbd2_phase3_train
  local/make_swbd_cellular1.pl /export/common/data/corpora/LDC/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/common/data/corpora/LDC/LDC2004S07 \
    data/swbd_cellular2_train
fi

if [ $stage -le 1 ]; then
  mkdir -p data/local/sre
  wget -P data/local/sre http://www.openslr.org/resources/15/speaker_list.tgz
  tar -C data/local/sre -xvf data/local/sre/speaker_list.tgz
  sre_ref=data/local/sre/speaker_list

  local/make_sre.pl /export/common/data/corpora/LDC/LDC2006S44 04 \
    data/local/sre/speaker_list data/sre2004
fi

if [ $stage -le 2 ]; then
  local/make_mx6_calls.pl /export/common/data/corpora/LDC/LDC2013S03 data/local/mx6

  for mic in 02 04 05 06 07 08 09 10 11 12 13; do
    local/make_mx6_mic.pl /export/common/data/corpora/LDC/LDC2013S03 $mic data/local/mx6
  done

  utils/combine_data.sh data/local/mx6/mx6_mic_04_to_13 \
    data/local/mx6/mx6_mic_{04,05,06,07,08,09,10,11,12,13}
fi

if [ $stage -le 3 ]; then
  utils/data/get_reco2dur.sh \
     --read-entire-file true --cmd "$train_cmd" --nj 32 --permissive true \
     data/local/mx6/mx6_mic_04_to_13

  utils/copy_data_dir.sh data/local/mx6/mx6_mic_04_to_13 \
    data/local/mx6/mx6_mic_04_to_13_filtered

  utils/filter_scp.pl data/local/mx6/mx6_mic_04_to_13/reco2dur \
    data/local/mx6/mx6_mic_04_to_13/wav.scp > data/local/mx6_mic_04_to_13_filtered/wav.scp

  utils/fix_data_dir.sh \
    data/local/mx6/mx6_mic_04_to_13_filtered

  utils/subset_data_dir.sh \
    data/local/mx6/mx6_mic_04_to_13_filtered 2000 \
    data/local/mx6/mx6_mic_04_to_13_2k
fi

if [ $stage -le 4 ]; then
  utils/combine_data.sh data/mx6_mic \
    data/local/mx6/mx6_mic_02 data/local/mx6/mx6_mic_04_to_13_2k

  utils/copy_data_dir.sh data/local/mx6/mx6_calls data/mx6_calls
fi

if [ $stage -le 5 ]; then
  utils/combine_data.sh data/train_semisup \
    data/swbd2_phase1_train \
    data/swbd2_phase2_train \
    data/swbd2_phase3_train \
    data/swbd_cellular1_train \
    data/swbd_cellular2_train \
    data/sre2004 data/mx6_calls data/mx6_mic
fi

mkdir -p sad_model
if [ $stage -le 6 ]; then
  (
  cd sad_model
  wget http://kaldi-asr.org/models/0004_tdnn_stats_asr_sad_1a.tar.gz
  tar -xzvf 0004_tdnn_stats_asr_sad_1a.tar.gz
  )
fi

if [ $stage -le 7 ]; then
 steps/segmentation/detect_speech_activity.sh --stage $sad_stage \
   --cmd "$train_cmd" --nj $nj --convert-data-dir-to-whole true \
   --extra-left-context 79 --extra-right-context 21 \
   --extra-left-context-initial 0 --extra-right-context-final 0 \
   --frames-per-chunk 150 --mfcc-config sad_model/conf/mfcc_hires.conf \
   data/train_semisup sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   sad_model/mfcc_hires sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   data/train_semisup_1a

 for f in data/train_semisup_1a_seg/{utt2spk,feats.scp}; do
   if [ ! -f $f ]; then
     echo "$0: Could not find $f"
     exit 1
   fi
 done
fi

if [ $stage -le 8 ]; then
 steps/segmentation/detect_speech_activity.sh --stage $sad_stage \
   --cmd "$train_cmd" --nj $nj --convert-data-dir-to-whole true \
   --extra-left-context 79 --extra-right-context 21 \
   --extra-left-context-initial 0 --extra-right-context-final 0 \
   --frames-per-chunk 150 --mfcc-config sad_model/conf/mfcc_hires.conf \
   data/train_mixer6_mic_02 sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   sad_model/mfcc_hires sad_model/exp/segmentation_1a/tdnn_stats_asr_sad_1a \
   data/train_mixer6_mic_02_1a
fi
