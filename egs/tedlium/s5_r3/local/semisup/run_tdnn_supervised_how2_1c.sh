#!/bin/bash

# run_tdnn_1b.sh is the script which results are presented in the corpus release paper.
# It uses 2 to 6 jobs and add proportional-shrink 10.

# WARNING
# This script is flawed and misses key elements to optimize the tdnnf setup.
# You can run it as is to reproduce results from the corpus release paper,
# but a more up-to-date version should be looked at in other egs until another
# setup is added here.

# local/chain/compare_wer_general.sh exp/chain_cleaned/tdnn_1a exp/chain_cleaned/tdnn_1b
# System                      tdnn_1a   tdnn_1b   tdnn_1b
# Scoring script	            sclite    sclite   score_basic
# WER on dev(orig)              8.2       7.9         7.9
# WER on dev(rescored ngram)    7.6       7.4         7.5
# WER on dev(rescored rnnlm)    6.3       6.2         6.2
# WER on test(orig)             8.1       8.0         8.2
# WER on test(rescored ngram)   7.7       7.7         7.9
# WER on test(rescored rnnlm)   6.7       6.7         6.8
# Final train prob            -0.0802   -0.0899
# Final valid prob            -0.0980   -0.0974
# Final train prob (xent)     -1.1450   -0.9449
# Final valid prob (xent)     -1.2498   -1.0002
# Num-params                  26651840  25782720

# local/chain/compare_wer_general.sh exp/chain_cleaned/tdnn_1c_sp_bi
# System                tdnn_1c1_sp_bi
# WER on dev(orig)           8.18
# WER on dev(rescored)       7.59
# WER on test(orig)          8.39
# WER on test(rescored)      7.83
# Final train prob        -0.0625
# Final valid prob        -0.0740
# Final train prob (xent)   -0.9813
# Final valid prob (xent)   -0.9876
# Num-params                 9468080


## how you run this (note: this assumes that the run_tdnn.sh soft link points here;
## otherwise call it directly in its location).
# by default, with cleanup:
# local/chain/run_tdnn.sh

# without cleanup:
# local/chain/run_tdnn.sh  --train-set train --gmm tri3 --nnet3-affix "" &

# note, if you have already run the corresponding non-chain nnet3 system
# (local/nnet3/run_tdnn.sh), you may want to run with --stage 14.


set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
train_stage=-10
nj=1600
decode_nj=30
max_jobs_run=30

tdnn_affix=_sup_how2_1c

nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
chain_affix=_cleaned

supervised_set=train_cleaned
adaptation_set=how2_train

sup_chain_dir=exp/chain_cleaned/tdnn_1c2_sp_bi  # supervised chain system
sup_tree_dir=exp/chain_cleaned/tree # tree directory for supervised chain system
src_ivector_root_dir=exp/nnet3_cleaned  # i-vector extractor root directory

get_egs_stage=-10

lang_test=data/lang
test_graph_affix=

# Neural network opts
hidden_dim=1536
bottleneck_dim=160
small_dim=256

common_egs_dir=  # Supply this to skip unsupervised egs creation

# training options
num_epochs=2
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

use_babble=true
frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
tolerance=1   # frame-tolerance for chain training

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

diff $sup_tree_dir/tree $sup_chain_dir/tree || { echo "$0: $sup_tree_dir/tree and $sup_chain_dir/tree differ"; exit 1; }

dir=exp/chain${chain_affix}/tdnn${tdnn_affix}

adaptation_set_perturbed=${adaptation_set}

ivector_dir=$src_ivector_root_dir/ivectors_${adaptation_set}
if [ $stage -le 1 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    data/${adaptation_set}_hires $src_ivector_root_dir/extractor \
    ${src_ivector_root_dir}/ivectors_${adaptation_set} || exit 1
fi

graph_dir=$sup_chain_dir/graph${test_graph_affix}
adaptation_lat_dir=${sup_chain_dir}_lats_${adaptation_set}

if [ $stage -le 3 ]; then
  steps/nnet3/align_lats.sh \
    --nj $nj --cmd "$decode_cmd --mem 4G" \
    --acoustic-scale 1.0 --scale-opts "--transition-scale=1.0 --self-loop-scale=1.0" \
    --frames-per-chunk 150 \
    --online-ivector-dir $ivector_dir \
    --generate-ali-from-lats true \
    data/${adaptation_set}_hires $lang_test ${sup_chain_dir} \
    ${adaptation_lat_dir} || exit 1
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

adaptation_data_dir=data/${adaptation_set}
aug_data_dir=${adaptation_data_dir}_aug_hires

maybe_babble=
if $use_babble; then
  maybe_babble=babble
fi

if [ $stage -le 5 ]; then
  rvb_opts=()
  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  steps/data/augment_data_dir_for_asr.py --utt-prefix "noise" --fg-interval 1 \
    --fg-snrs "20:15:10:5:0" --fg-noise-dir "data/musan_noise" \
    ${adaptation_data_dir} ${adaptation_data_dir}_noise || exit 1

  steps/data/augment_data_dir_for_asr.py --utt-prefix "music" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" \
    --bg-noise-dir "data/musan_music" \
    ${adaptation_data_dir} ${adaptation_data_dir}_music || exit 1

  if $use_babble; then
    steps/data/augment_data_dir_for_asr.py --utt-prefix "babble" \
      --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
      --bg-noise-dir "data/mx6_mic" \
      ${adaptation_data_dir} ${adaptation_data_dir}_babble || exit 1
  fi
  # corrupt the data to generate multi-condition data
  # for data_dir in train dev test; do
  seed=0
  reverb_dirs=
  for name in noise music $maybe_babble; do 
    steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --speech-rvb-probability 0.8 \
      --pointsource-noise-addition-probability 0 \
      --isotropic-noise-addition-probability 0 \
      --num-replications 1 \
      --source-sampling-rate 16000 \
      --random-seed $seed \
      ${adaptation_data_dir}_${name} ${adaptation_data_dir}_${name}_reverb || exit 1
    seed=$[seed+1]
    reverb_dirs="$reverb_dirs ${adaptation_data_dir}_${name}"
  done

  utils/combine_data.sh \
    ${aug_data_dir} ${adaptation_data_dir} ${reverb_dirs} || exit 1
fi

if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $aug_data_dir/data/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$aug_data_dir/data/storage $aug_data_dir/data/storage
  fi

  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf \
    --nj $nj ${aug_data_dir}
  steps/compute_cmvn_stats.sh ${aug_data_dir}
  utils/fix_data_dir.sh $aug_data_dir
fi

aug_ivector_dir=$src_ivector_root_dir/ivectors_${adaptation_set}_aug
if [ $stage -le 7 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    $aug_data_dir $src_ivector_root_dir/extractor \
    ${src_ivector_root_dir}/ivectors_${adaptation_set}_aug || exit 1
fi

aug_lat_dir=${sup_chain_dir}_lats_${adaptation_set}_aug
if [ $stage -le 8 ]; then
  utt_prefixes=
  for name in noise music $maybe_babble; do 
    utt_prefixes="$utt_prefixes rev1-${name}_"
  done

  steps/copy_lat_dir.sh --cmd "$decode_cmd" --nj $nj --write-compact false \
    --utt-prefixes "$utt_prefixes" --include-original true \
    $aug_data_dir $adaptation_lat_dir $aug_lat_dir || exit 1
fi

frame_subsampling_factor=1
if [ -f $sup_chain_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $sup_chain_dir/frame_subsampling_factor)
fi

cmvn_opts=$(cat $sup_chain_dir/cmvn_opts) || exit 1

if [ $stage -le 11 ]; then
  mkdir -p $dir

  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $sup_tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=$hidden_dim
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  linear-component name=prefinal-l dim=$small_dim $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=$hidden_dim small-dim=$small_dim
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=$hidden_dim small-dim=$small_dim
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 12 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh \
    --num-repeats "0,1" \
    $sup_tree_dir $adaptation_lat_dir \
    $dir
fi

if [ $train_stage -le -4 ]; then
  train_stage=-4
fi

if [ $stage -le 18 ]; then
 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir $aug_ivector_dir \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width $frames_per_eg \
    --egs.stage $get_egs_stage \
    --chain.left-tolerance=1 --chain.right-tolerance=1 \
    --chain.alignment-subsampling-factor 1 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir $aug_data_dir \
    --tree-dir $sup_tree_dir \
    --lat-dir $aug_lat_dir \
    --dir $dir
fi

if [ $stage -le 19 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_test $dir $dir/graph${test_graph_affix}
fi

if [ $stage -le 20 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in dev test; do
      (
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $dir/graph data/${dset}_hires $dir/decode_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
        data/${dset}_hires ${dir}/decode_${dset} ${dir}/decode_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
