#!/bin/bash

set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
train_stage=-10
nj=1600
decode_nj=30
max_jobs_run=30

tlstm_affix=_sup_how2_1a

nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
chain_affix=_cleaned

supervised_set=train_cleaned
adaptation_set=how2_train

# Seed model options
sup_chain_dir=exp/chain_cleaned/tdnn_1c2_sp_bi  # supervised chain system
sup_tree_dir=exp/chain_cleaned/tree # tree directory for supervised chain system
src_ivector_root_dir=exp/nnet3_cleaned  # i-vector extractor root directory

get_egs_stage=-10

lang_test=data/lang_how2
test_graph_affix=_how2

# Neural network opts
hidden_dim=1536
cell_dim=1536
projection_dim=384

common_egs_dir=  # Supply this to skip unsupervised egs creation

# training options
num_epochs=4
chunk_left_context=40
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.025
label_delay=5

use_babble=true
frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
test_sets="how2_dev5"

# decode options
extra_left_context=50
extra_right_context=0
frames_per_chunk_decoding=160

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

dir=exp/chain${chain_affix}/tdnn_lstm${tlstm_affix}

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
noisy_data_dir=${adaptation_data_dir}_noisy_hires

maybe_babble=
if $use_babble; then
  maybe_babble=babble
fi

if [ $stage -le 5 ]; then
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
  noisy_dirs=
  for name in noise music $maybe_babble; do 
    noisy_dirs="$noisy_dirs ${adaptation_data_dir}_${name}"
  done

  utils/combine_data.sh \
    ${noisy_data_dir} ${adaptation_data_dir} ${noisy_dirs} || exit 1
fi

if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $noisy_data_dir/data/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/ami-$(date +'%m_%d_%H_%M')/s5/$noisy_data_dir/data/storage $noisy_data_dir/data/storage
  fi

  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run $max_jobs_run" --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf \
    --nj $nj ${noisy_data_dir}
  steps/compute_cmvn_stats.sh ${noisy_data_dir}
  utils/fix_data_dir.sh $noisy_data_dir
fi

noisy_ivector_dir=$src_ivector_root_dir/ivectors_${adaptation_set}_noisy
if [ $stage -le 7 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    $noisy_data_dir $src_ivector_root_dir/extractor \
    ${src_ivector_root_dir}/ivectors_${adaptation_set}_noisy || exit 1
fi

noisy_lat_dir=${sup_chain_dir}_lats_${adaptation_set}_noisy
if [ $stage -le 8 ]; then
  utt_prefixes=
  for name in noise music $maybe_babble; do 
    utt_prefixes="$utt_prefixes ${name}_"
  done

  steps/copy_lat_dir.sh --cmd "$decode_cmd" --nj $nj --write-compact false \
    --utt-prefixes "$utt_prefixes" --include-original true \
    $noisy_data_dir $adaptation_lat_dir $noisy_lat_dir || exit 1
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
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  tdnn_opts="l2-regularize=0.002"
  lstm_opts="l2-regularize=0.0005 decay-time=20"
  output_opts="l2-regularize=0.001" 

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$hidden_dim $tdnn_opts

  fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm4 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm4 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm4 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
  # We use separate outputs for supervised and unsupervised data
  # so we can properly track the train and valid objectives.
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
    --feat.online-ivector-dir $noisy_ivector_dir \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width $frames_per_eg \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.stage $get_egs_stage \
    --chain.left-tolerance=1 --chain.right-tolerance=1 \
    --chain.alignment-subsampling-factor 1 \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.max-param-change 2.0 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs false \
    --feat-dir $noisy_data_dir \
    --tree-dir $sup_tree_dir \
    --lat-dir $noisy_lat_dir \
    --dir $dir
fi

test_graph_dir=$sup_tree_dir/graph${test_graph_affix}
if [ $stage -le 19 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  if [ ! -s $test_graph_dir/HCLG.fst ]; then
    utils/mkgraph.sh --self-loop-scale 1.0 $lang_test $sup_tree_dir $test_graph_dir
  fi
fi

if [ $stage -le 20 ]; then
  rm $dir/.error 2>/dev/null || true
  for dset in $test_sets; do
      (
      steps/nnet3/decode.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset}_hires \
          --frames-per-chunk $frames_per_chunk_decoding \
          --extra-left-context $extra_left_context \
          --extra-right-context $extra_right_context \
          --extra-left-context-initial 0 --extra-right-context-final 0 \
          --scoring-opts "--min-lmwt 5 " \
         $test_graph_dir data/${dset}_hires $dir/decode${test_graph_affix}_${dset} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" ${lang_test} ${lang_test}_rescore \
        data/${dset}_hires ${dir}/decode${test_graph_affix}_${dset} ${dir}/decode${test_graph_affix}_${dset}_rescore || exit 1
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
