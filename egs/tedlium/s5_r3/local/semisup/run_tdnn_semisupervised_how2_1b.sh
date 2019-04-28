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

tdnn_affix=_semisup_how2_1b

nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
chain_affix=_cleaned

supervised_set=train_cleaned
unsupervised_set=how2_unsup_1a_seg

sup_chain_dir=exp/chain_cleaned/tdnn_1c2_sp_bi  # supervised chain system
sup_lat_dir=exp/chain_cleaned/tri3_cleaned_train_cleaned_sp_lats  # supervised set lattices
sup_tree_dir=exp/chain_cleaned/tree # tree directory for supervised chain system
src_ivector_root_dir=exp/nnet3_cleaned  # i-vector extractor root directory
sup_ivector_dir=exp/nnet3_cleaned/ivectors_train_cleaned_sp_hires

lang_test=data/lang
test_graph_affix=

# Semi-supervised options
supervision_weights=1.0,1.0   # Weights for supervised, unsupervised data egs.
                              # Can be used to scale down the effect of unsupervised data
                              # by using a smaller scale for it e.g. 1.0,0.3
lm_weights=3,1  # Weights on phone counts from supervised, unsupervised data for denominator FST creation
num_copies=3,1

# Neural network opts
hidden_dim=1536
bottleneck_dim=160
small_dim=256

sup_egs_dir=   # Supply this to skip supervised egs creation
unsup_egs_dir=  # Supply this to skip unsupervised egs creation
unsup_egs_opts=  # Extra options to pass to unsupervised egs creation
use_smart_splitting=true

extra_supervision_opts="--only-scale-graph --normalize"

# training options
num_epochs=2
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

sup_frames_per_eg=150,110,100
unsup_frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices when
                      # creating numerator supervision
lattice_prune_beam=4.0  # beam for pruning the lattices prior to getting egs
                        # for unsupervised data
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

supervised_set_perturbed=${supervised_set}_sp
unsupervised_set_perturbed=${unsupervised_set}

unsup_ivector_dir=$src_ivector_root_dir/ivectors_${unsupervised_set}
if [ $stage -le 1 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    data/${unsupervised_set}_hires $src_ivector_root_dir/extractor \
    ${src_ivector_root_dir}/ivectors_${unsupervised_set} || exit 1
fi

graph_dir=$sup_chain_dir/graph${test_graph_affix}
unsup_lat_dir=${sup_chain_dir}/decode${test_graph_affix}_${unsupervised_set}
best_path_dir=${sup_chain_dir}/best_path${test_graph_affix}_${unsupervised_set}

if [ $stage -le 2 ]; then
  if [ ! -f $graph_dir/HCLG.fst ]; then
    utils/mkgraph.sh --self-loop-scale 1.0 \
      $lang_test $sup_chain_dir $graph_dir
  fi
fi

if [ $stage -le 3 ]; then
  steps/nnet3/decode_semisup.sh \
    --nj $nj --cmd "$decode_cmd --mem 4G" --num-threads 4 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --write-compact false \
    --frames-per-chunk 150 \
    --online-ivector-dir ${src_ivector_root_dir}/ivectors_${unsupervised_set} \
    $graph_dir data/${unsupervised_set}_hires $unsup_lat_dir || exit 1
fi

if [ $stage -le 4 ]; then
  steps/best_path_weights.sh --cmd "$decode_cmd" \
    --acwt 0.1 \
    $unsup_lat_dir $best_path_dir || exit 1
fi

frame_subsampling_factor=1
if [ -f $sup_chain_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $sup_chain_dir/frame_subsampling_factor)
fi

cmvn_opts=$(cat $sup_chain_dir/cmvn_opts) || exit 1

if [ $stage -le 10 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh \
    --cmd "$train_cmd" \
    ${sup_tree_dir} $dir/sup_den_fst
fi

if [ $stage -le 11 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh \
    --cmd "$train_cmd" --num-repeats "0,1"\
    ${sup_tree_dir} $best_path_dir $dir/unsup_den_fst
fi

if [ $stage -le 12 ]; then
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

  output name=output-0 input=output.affine
  output name=output-1 input=output.affine

  output name=output-0-xent input=output-xent.log-softmax
  output name=output-1-xent input=output-xent.log-softmax
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

. $dir/configs/vars

left_context=$model_left_context
right_context=$model_right_context

egs_left_context=$(perl -e "print int($left_context + $frame_subsampling_factor / 2)")
egs_right_context=$(perl -e "print int($right_context + $frame_subsampling_factor / 2)")

supervised_set_perturbed=${supervised_set}_sp

if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set_perturbed}

  if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
    mkdir -p $sup_egs_dir/
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the supervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
               --left-context $egs_left_context --right-context $egs_right_context \
               --frame-subsampling-factor $frame_subsampling_factor \
               --alignment-subsampling-factor $frame_subsampling_factor \
               --frames-per-eg $sup_frames_per_eg \
               --frames-per-iter 5000000 --constrained false \
               --cmvn-opts "$cmvn_opts" \
               --online-ivector-dir $sup_ivector_dir \
               --generate-egs-scp true \
               data/${supervised_set_perturbed}_hires $dir/sup_den_fst \
               $sup_lat_dir $sup_egs_dir
  fi
fi

if $use_smart_splitting; then
  get_egs_script=steps/nnet3/chain/get_egs_split.sh
else
  get_egs_script=steps/nnet3/chain/get_egs.sh
fi

if [ -z "$unsup_egs_dir" ]; then
  unsup_egs_dir=$dir/egs_${unsupervised_set_perturbed}

  if [ $stage -le 13 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi
    mkdir -p $unsup_egs_dir
    touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the unsupervised data"
    $get_egs_script \
      --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --left-tolerance $tolerance --right-tolerance $tolerance \
      --left-context $egs_left_context --right-context $egs_right_context \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 5000000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
      --lattice-prune-beam "$lattice_prune_beam" \
      --extra-supervision-opts "$extra_supervision_opts" \
      --deriv-weights-scp $best_path_dir/weights.scp \
      --online-ivector-dir $unsup_ivector_dir \
      --generate-egs-scp true $unsup_egs_opts \
      data/${unsupervised_set_perturbed}_hires $dir/unsup_den_fst \
      $unsup_lat_dir $unsup_egs_dir
  fi
fi

comb_egs_dir=$dir/comb_egs
if [ $stage -le 14 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 64 \
    --lang2weight $supervision_weights \
    --lang2num-copies $num_copies 2 \
    $sup_egs_dir $unsup_egs_dir $comb_egs_dir
  touch $comb_egs_dir/.nodelete # keep egs around when that run dies.
fi

if [ $train_stage -le -4 ]; then
  # This is to skip stages of den-fst creation, which was already done.
  train_stage=-4
fi

if [ $stage -le 18 ]; then
 steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.online-ivector-dir $sup_ivector_dir \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.lda-output-name "output-0" \
    --egs.dir "$comb_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $sup_frames_per_eg \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 5000000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs false \
    --feat-dir data/${supervised_set_perturbed}_hires \
    --tree-dir $sup_tree_dir \
    --lat-dir $sup_lat_dir \
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
