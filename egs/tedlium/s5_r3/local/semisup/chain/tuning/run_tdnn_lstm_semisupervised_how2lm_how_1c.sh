#!/bin/bash

# This is similar to run_tdnn_semisupervised_how2_1b.sh, but uses 
# a better LM for decoding which as trained by including some HOW2 transcripts.

set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
train_stage=-10
nj=1600
decode_nj=30

tlstm_affix=_semisup_how2_1c

nnet3_affix=_cleaned  # cleanup affix for nnet3 and chain dirs, e.g. _cleaned
chain_affix=_cleaned

supervised_set=train_cleaned
unsupervised_set=how2_unsup_1a_seg

# Seed model options
sup_chain_dir=exp/chain_cleaned/tdnn_1c2_sp_bi  # supervised chain system
sup_lat_dir=exp/chain_cleaned/tri3_cleaned_train_cleaned_sp_lats  # supervised set lattices
sup_tree_dir=exp/chain_cleaned/tree # tree directory for supervised chain system
src_ivector_root_dir=exp/nnet3_cleaned  # i-vector extractor root directory
sup_ivector_dir=exp/nnet3_cleaned/ivectors_train_cleaned_sp_hires

lang_test=data/lang_how2
test_graph_affix=_how2

# Semi-supervised options
supervision_weights=1.0,1.0   # Weights for supervised, unsupervised data egs.
                              # Can be used to scale down the effect of unsupervised data
                              # by using a smaller scale for it e.g. 1.0,0.3
num_copies=1,1

# Neural network opts
hidden_dim=1536
cell_dim=1536
projection_dim=384

sup_egs_dir=   # Supply this to skip supervised egs creation
unsup_egs_dir=  # Supply this to skip unsupervised egs creation
unsup_egs_opts=  # Extra options to pass to unsupervised egs creation
use_smart_splitting=true

extra_supervision_opts="--only-scale-graph --normalize"

# training options
num_epochs=2
chunk_left_context=40
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.025
label_delay=5

use_babble=true
sup_frames_per_eg=150,110,100
unsup_frames_per_eg=150  # Using a frames-per-eg of 150 for unsupervised data
                         # was found to be better than allowing smaller chunks
                         # (160,140,110,80) like for supervised system
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices when
                      # creating numerator supervision
lattice_prune_beam=4.0  # beam for pruning the lattices prior to getting egs
                        # for unsupervised data
tolerance=1   # frame-tolerance for chain training

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

supervised_set_perturbed=${supervised_set}_sp

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

exit 1

if [ $stage -le 5 ]; then
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

unsupervised_data_dir=data/${unsupervised_set}
unsupervised_data_dir_noisy=${unsupervised_data_dir}_noisy_hires
unsupervised_set_perturbed=${unsupervised_set}_noisy

maybe_babble=
if $use_babble; then
  maybe_babble=babble
fi

if [ $stage -le 6 ]; then
  steps/data/augment_data_dir_for_asr.py --utt-prefix "noise" --fg-interval 1 \
    --fg-snrs "20:15:10:5:0" --fg-noise-dir "data/musan_noise" \
    ${unsupervised_data_dir} ${unsupervised_data_dir}_noise || exit 1

  steps/data/augment_data_dir_for_asr.py --utt-prefix "music" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" \
    --bg-noise-dir "data/musan_music" \
    ${unsupervised_data_dir} ${unsupervised_data_dir}_music || exit 1

  if $use_babble; then
    steps/data/augment_data_dir_for_asr.py --utt-prefix "babble" \
      --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
      --bg-noise-dir "data/mx6_mic" \
      ${unsupervised_data_dir} ${unsupervised_data_dir}_babble || exit 1
  fi
  noisy_dirs=
  for name in noise music $maybe_babble; do 
    noisy_dirs="$noisy_dirs ${unsupervised_data_dir}_${name}"
  done

  utils/combine_data.sh \
    ${unsupervised_data_dir_noisy} ${noisy_dirs} || exit 1
fi

if [ $stage -le 7 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsupervised_data_dir_noisy/data/storage ]; then
    utils/create_split_dir.pl \
     /export/b{05,06,11,12}/$USER/kaldi-data/egs/how2-$(date +'%m_%d_%H_%M')/s5/$unsupervised_data_dir_noisy/data/storage $unsupervised_data_dir_noisy/data/storage
  fi

  steps/make_mfcc.sh --cmd "$train_cmd --max-jobs-run 30" --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf \
    --nj $nj ${unsupervised_data_dir_noisy}
  steps/compute_cmvn_stats.sh ${unsupervised_data_dir_noisy}
  utils/fix_data_dir.sh $unsupervised_data_dir_noisy
fi

unsup_ivector_dir_noisy=$src_ivector_root_dir/ivectors_${unsupervised_set}_noisy
if [ $stage -le 8 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    $unsupervised_data_dir_noisy $src_ivector_root_dir/extractor \
    ${unsup_ivector_dir_noisy} || exit 1
fi

unsup_lat_dir_noisy=${sup_chain_dir}/decode${test_graph_affix}_${unsupervised_set}_noisy
if [ $stage -le 9 ]; then
  utt_prefixes=
  for name in noise music $maybe_babble; do 
    utt_prefixes="$utt_prefixes ${name}_"
  done

  steps/copy_lat_dir.sh --cmd "$decode_cmd" --nj $nj --write-compact false \
    --utt-prefixes "$utt_prefixes" \
    $unsup_data_dir_noisy $unsup_lat_dir $unsup_lat_dir_noisy || exit 1
    
  for name in noise music $maybe_babble; do 
    cat $best_path_dir/weights.scp | awk -v name=$name '{print name"_"$0}'
  done > $unsup_lat_dir_noisy/weights.scp
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
  relu-batchnorm-layer name=tdnn10 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn11 input=Append(-3,0,3) dim=$hidden_dim $tdnn_opts
  fast-lstmp-layer name=lstm5 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm5 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm5 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
  # We use separate outputs for supervised and unsupervised data
  # so we can properly track the train and valid objectives.
  output name=output-0 input=output.affine@$label_delay
  output name=output-1 input=output.affine@$label_delay

  output name=output-0-xent input=output-xent.log-softmax@$label_delay
  output name=output-1-xent input=output-xent.log-softmax@$label_delay
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

. $dir/configs/vars

left_context=$[model_left_context + chunk_left_context]
right_context=$[model_right_context + chunk_right_context]
left_context_initial=$model_left_context
right_context_final=$model_right_context

egs_left_context=$(perl -e "print int($left_context + $frame_subsampling_factor / 2)")
egs_right_context=$(perl -e "print int($right_context + $frame_subsampling_factor / 2)")
egs_left_context_initial=$(perl -e "print int($left_context_initial + $frame_subsampling_factor / 2)")
egs_right_context_final=$(perl -e "print int($right_context_final + $frame_subsampling_factor / 2)")

if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set_perturbed}

  if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b{05,06,11,12}/$USER/kaldi-data/egs/tedlium-$(date +'%m_%d_%H_%M')/s5_r2/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
    mkdir -p $sup_egs_dir/
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the supervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
               --left-context $egs_left_context --right-context $egs_right_context \
               --left-context-initial $egs_left_context_initial \
               --right-context-final $egs_right_context_final \
               --frame-subsampling-factor $frame_subsampling_factor \
               --alignment-subsampling-factor $frame_subsampling_factor \
               --frames-per-eg $sup_frames_per_eg \
               --frames-per-iter 1500000 --constrained false \
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
       /export/b{05,06,11,12}/$USER/kaldi-data/egs/how2-$(date +'%m_%d_%H_%M')/s5_r3/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi
    mkdir -p $unsup_egs_dir
    touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the unsupervised data"
    $get_egs_script \
      --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --left-tolerance $tolerance --right-tolerance $tolerance \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial \
      --right-context-final $egs_right_context_final \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
      --lattice-prune-beam "$lattice_prune_beam" \
      --extra-supervision-opts "$extra_supervision_opts" \
      --deriv-weights-scp $unsup_lat_dir_noisy/weights.scp \
      --online-ivector-dir $unsup_ivector_dir_noisy \
      --generate-egs-scp true $unsup_egs_opts \
      ${unsupervised_data_dir_noisy} $dir/unsup_den_fst \
      $unsup_lat_dir_noisy $unsup_egs_dir
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
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --trainer.num-chunk-per-minibatch 32,16 \
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
    --feat-dir data/${supervised_set_perturbed}_hires \
    --tree-dir $sup_tree_dir \
    --lat-dir $sup_lat_dir \
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
