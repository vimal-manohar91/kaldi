#!/bin/bash

# Copyright Johns Hopkins University (Author: Daniel Povey, Vijayaditya Peddinti) 2016.  Apache 2.0.
# This script generates the ctm files for dev_aspire, test_aspire and eval_aspire
# for scoring with ASpIRE scoring server.
# It also provides the WER for dev_aspire data.

set -e

# general opts
iter=final
stage=0
decode_num_jobs=30
affix=

# segmentation opts
sad_affix=
sad_opts="--extra-left-context 79 --extra-right-context 21 --frames-per-chunk 150 --extra-left-context-initial 0 --extra-right-context-final 0 --acwt 0.3"
sad_graph_opts=
sad_priors_opts=
sad_stage=0
segment_only=false

# ivector+decode opts
# tuned based on the ASpIRE nnet2 online system
max_count=75  # parameter for extract_ivectors.sh
max_state_duration=40
silence_weight=0.00001

# decode opts
decode_opts="--min-active 1000"
lattice_beam=8
extra_left_context=0 # change for (B)LSTM
extra_right_context=0 # change for BLSTM
frames_per_chunk=50 # change for (B)LSTM
acwt=0.1 # important to change this when using chain models
post_decode_acwt=1.0 # important to change this when using chain models
extra_left_context_initial=-1

score_opts="--min-lmwt 1 --max-lmwt 20"

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh || exit 1;

if [ $# -ne 6 ]; then
  echo "Usage: $0 [options] <data-dir> <sad-nnet-dir> <work-dir> <lang-dir> <graph-dir> <model-dir>"
  echo " Options:"
  echo "    --stage (0|1|2)   # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 dev_aspire data/lang exp/tri5a/graph_pp exp/nnet3/tdnn"
  exit 1;
fi

data_set=$1 #select from {dev_aspire, test_aspire, eval_aspire}*
sad_nnet_dir=$2
sad_work_dir=$3
lang=$4 # data/lang
graph=$5 #exp/tri5a/graph_pp
dir=$6 # exp/nnet3/tdnn

model_affix=`basename $dir`
ivector_root_dir=exp/nnet3
affix=${affix:+_${affix}}${iter:+_iter${iter}}

if [ "$data_set" == "test_aspire" ]; then
  out_file=single_dev_test${affix}_$model_affix.ctm
elif [ "$data_set" == "eval_aspire" ]; then
  out_file=single_eval${affix}_$model_affix.ctm
elif [ "$data_set" ==  "dev_aspire" ]; then
  # we will just decode the directory without oracle segments file
  # as we would like to operate in the actual evaluation condition
  out_file=single_dev${affix}_${model_affix}.ctm
fi

if [ $stage -le 2 ]; then
  steps/segmentation/detect_speech_activity.sh \
    --nj $decode_num_jobs --stage $sad_stage \
    --affix "$sad_affix" --graph-opts "$sad_graph_opts" \
    --transform-probs-opts "$sad_priors_opts" $sad_opts \
    data/$data_set $sad_nnet_dir mfcc_hires $sad_work_dir \
    $sad_work_dir/${data_set}${sad_affix:+_$sad_affix} || exit 1
fi

segmented_data_set=${data_set}${sad_affix:+_$sad_affix}

if [ $stage -le 3 ]; then
  if [ ! -f $sad_work_dir/${segmented_data_set}_seg/reco2file_and_channel ]; then
    awk '{print $2" "1}' $sad_work_dir/${segmented_data_set}_seg/segments | \
      sort -u > $sad_work_dir/${segmented_data_set}_seg/reco2file_and_channel
  fi

  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
    --reco2file-and-channel=${sad_work_dir}/${segmented_data_set}_seg/reco2file_and_channel \
    ${sad_work_dir}/${segmented_data_set}_seg/{utt2spk,segments,sys.rttm} || exit 1
fi

if $segment_only; then
  echo "$0: --segment-only is true. Exiting."
  exit 0
fi

if [ $stage -le 4 ]; then
  steps/online/nnet3/prepare_online_decoding.sh \
    --mfcc-config conf/mfcc_hires.conf \
    --max-count $max_count \
      $lang exp/nnet3/extractor "$dir" ${dir}_online
fi

decode_dir=${dir}_online/decode_${segmented_data_set}${affix}_pp
if [ $stage -le 6 ]; then
  echo "Generating lattices, with --acwt $acwt and --post-decode-acwt $post_decode_acwt "
      # the following options have not yet been implemented
      # --frames-per-chunk "$frames_per_chunk"
      #--extra-left-context $extra_left_context  \
      #--extra-right-context $extra_right_context  \
  steps/online/nnet3/decode.sh --nj $decode_num_jobs --cmd "$decode_cmd" \
      --config conf/decode.config $decode_opts \
      --acwt $acwt --post-decode-acwt $post_decode_acwt \
      --extra-left-context-initial $extra_left_context_initial \
      --silence-weight $silence_weight \
      --per-utt true \
      --skip-scoring true --iter $iter --lattice-beam $lattice_beam \
     $graph data/${segmented_data_set}_hires ${decode_dir}_tg || \
     { echo "$0: Error decoding" && exit 1; }
fi

if [ $stage -le 7 ]; then
  echo "Rescoring lattices"
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    --skip-scoring true \
    ${lang}_pp_test{,_fg} data/${segmented_data_set}_hires \
    ${decode_dir}_{tg,fg};
fi

decode_dir=${decode_dir}_fg
if [ $stage -le 8 ]; then
  local/score_aspire.sh --cmd "$decode_cmd" \
    $score_opts \
    --word-ins-penalties "0.0,0.25,0.5,0.75,1.0" \
    --ctm-beam 6 \
    --iter $iter \
    --decode-mbr true \
    --tune-hyper true \
    $lang $decode_dir $data_set $segmented_data_set $out_file
fi
