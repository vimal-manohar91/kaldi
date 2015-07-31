#!/bin/bash

# This script adapt an online-nnet2 model to new data directory by 
# training only the last affine component.

stage=-1
set -e 
set -o pipefail
set -u

train_stage=-100
use_gpu=true
srcdir=exp/nnet2_online/nnet_ms_a_multilang/0_online
ivector_src=exp/nnet2_online/ASM/extractor
lang=../kaldi/egs/SBS-mul-exp/data/lang/
alidir=../kaldi/egs/SBS-mul-exp/exp/tri3b
data_dir=../kaldi/egs/SBS-mul-exp/data/train
trainfeats=exp/nnet2_online/nnet_ms_a_multilang_SBS-mul-noSW/train_activations
graph_dir=../kaldi/egs/SBS-mul-exp/exp/tri3b/graph
test_dir=../kaldi/egs/SBS-mul-exp/data/eval
dir=exp/nnet2_online/nnet_ms_a_multilang_SBS-mul-noSW
nj=20
decode_nj=5

iter=140

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads" 
fi

if [ $stage -le 0 ]; then
  echo "$0: dumping activations from multilingual model"
  steps/online/nnet2/dump_nnet_activations.sh \
    --cmd "$train_cmd" --nj $nj \
    $data_dir $srcdir $trainfeats
fi

if [ $stage -le 1 ]; then
  echo "$0: training 0-hidden-layer model on top of activations of multilingual model"
  steps/nnet2/retrain_fast.sh --stage $train_stage \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --cmd "$decode_cmd" \
    --num-jobs-nnet 4 \
    --mix-up 0 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     $trainfeats/data $lang $alidir $dir 
fi

if [ $stage -le 2 ]; then
  echo "$0: formatting combined model for online decoding."
  steps/online/nnet2/prepare_online_decoding_retrain.sh $srcdir $lang $dir ${dir}_online
fi

if [ $stage -le 3 ]; then
  # do online decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --skip-scoring true --cmd "$decode_cmd" --nj $decode_nj \
    $graph_dir $test_dir ${dir}_online/decode_`basename $test_dir`
fi

if [ $stage -le 4 ]; then
  # do online per-utterance decoding with the combined model.
  steps/online/nnet2/decode.sh --config conf/decode.config --skip-scoring true --cmd "$decode_cmd" --nj $decode_nj \
     --per-utt true \
    $graph_dir $test_dir ${dir}_online/decode_utt_`basename $test_dir`
fi

if [ $stage -le 5 ]; then
  steps/nnet2/create_appended_model.sh $srcdir $dir ${dir}_combined_init

  initial_learning_rate=0.01
  nnet-am-copy --learning-rate=$initial_learning_rate ${dir}_combined_init/final.mdl ${dir}_combined_init/final.mdl
fi

if [ $stage -le 6 ]; then
  # This version of the get_egs.sh script does the feature extraction and iVector
  # extraction in a single binary, reading the config, as part of the script.
  steps/online/nnet2/get_egs.sh --cmd "$train_cmd" --num-jobs-nnet 4 \
    $data_dir $alidir ${dir}_online ${dir}_combined
fi

if [ $stage -le 7 ]; then
  steps/nnet2/train_more.sh --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
     ${dir}_combined_init/final.mdl ${dir}_combined/egs ${dir}_combined 
fi

if [ $stage -le 8 ]; then
  # Create an online-decoding dir corresponding to what we just trained above.
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    $lang $ivector_src \
    ${dir}_combined ${dir}_combined_online || exit 1;
fi

if [ $stage -le 9 ]; then
  # do the online decoding on top of the retrained _combined_online model, and
  # also the per-utterance version of the online decoding.
  steps/online/nnet2/decode.sh --config conf/decode.config --skip-scoring true --cmd "$decode_cmd" --nj $decode_nj \
    $graph_dir $test_dir ${dir}_combined_online/decode_`basename $test_dir`
  steps/online/nnet2/decode.sh --config conf/decode.config --skip-scoring true --cmd "$decode_cmd" --nj $decode_nj \
    --per-utt true \
    $graph_dir $test_dir ${dir}_combined_online/decode_utt_`basename $test_dir`
fi

exit 0

