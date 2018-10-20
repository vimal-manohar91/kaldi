#!/bin/bash

# Unsupervised set: train_unsup100k_500k
# unsup_frames_per_eg=150
# Deriv weights: Lattice posterior of best path pdf
# Unsupervised weight: 1.0
# Weights for phone LM (supervised, unsupervises): 3,2
# LM for decoding unsupervised data: 4gram

set -u -e -o pipefail

stage=-2
train_stage=-100
nj=40
decode_nj=80

supervised_set=train_cleaned
unsupervised_set=train_unt.asr_seg_1a

srcdir=exp/chain_cleaned/tdnn_lstm_bab9_2_nepochs10_h512_sp
treedir=exp/chain_cleaned/tree
src_extractor=exp/nnet3_cleaned/extractor
sup_lat_dir=exp/chain_cleaned/tri5_cleaned_train_cleaned_sp_lats

nnet3_affix=_cleaned_semisup
chain_affix=_cleaned_semisup

frames_per_eg=150,120,90,75

# Unsupervised options
unsup_frames_per_eg=150  # if empty will be equal to the supervised model's config -- you will need to change minibatch_size for comb training accordingly
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices
lattice_prune_beam=4.0  # If supplied will prune the lattices prior to getting egs for unsupervised data
tolerance=1
phone_insertion_penalty=

smbr_leaky_hmm_coefficient=0.00001
mmi_factor_schedule="output-0=1.0,1.0 output-1=1.0,1.0"
smbr_factor_schedule="output-0=0.0,0.0 output-1=0.0,0.0"

# Semi-supervised options
affix=_semisup_1a  # affix for new chain-model directory trained on the combined supervised+unsupervised subsets
supervision_weights=1.0,1.0
chain_smbr_extra_opts="--one-silence-class"
lm_weights=3,1
num_copies=2,1
sup_egs_dir=
unsup_egs_dir=
unsup_egs_opts=

remove_egs=false
common_egs_dir=

hidden_dim=1024
cell_dim=1024
projection_dim=256

apply_deriv_weights=true
use_smart_splitting=true

# training options
num_epochs=2
minibatch_size=64,32
chunk_left_context=40
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=5

num_threads_ubm=12
# decode options
extra_left_context=50
extra_right_context=0

decode_iter=

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

lang=data/lang_chain
unsup_decode_lang=data/langp_test
unsup_decode_graph_affix=

test_lang=data/langp_test
test_graph_affix=
extractor=exp/nnet3${nnet3_affix}/extractor
graphdir=$srcdir/graph${unsup_decode_graph_affix}
semisup_set=train_semisup

utils/combine_data.sh data/${semisup_set} \
  data/${supervised_set} data/${unsupervised_set}

local/chain/run_ivector_common.sh --stage $stage \
                                  --nj $nj \
                                  --train-set $semisup_set \
                                  --gmm tri5_cleaned \
                                  --num-threads-ubm $num_threads_ubm \
                                  --nnet3-affix "$nnet3_affix" \
                                  --generate-alignments false

if [ ! -f $graphdir/HCLG.fst ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 $unsup_decode_lang $srcdir $graphdir
fi

if [ $stage -le 7 ]; then
  if [ -f data/${supervised_set}_sp_hires_nopitch/feats.scp ]; then
    echo "$0: data/${supervised_set}_sp_hires_nopitch/feats.scp exists. Remove it or re-run from next stage"
    exit 1
  fi

  steps/make_mfcc_pitch_online.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" data/${supervised_set}_sp_hires
  steps/compute_cmvn_stats.sh data/${supervised_set}_sp_hires
  utils/fix_data_dir.sh data/${supervised_set}_sp_hires

  utils/data/limit_feature_dim.sh 0:39 \
    data/${supervised_set}_sp_hires data/${supervised_set}_sp_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh \
    data/${supervised_set}_sp_hires_nopitch || exit 1;
  utils/fix_data_dir.sh data/${supervised_set}_sp_hires_nopitch
fi

if [ $stage -le 8 ]; then
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/${supervised_set}_sp_hires_nopitch data/${supervised_set}_sp_max2_hires_nopitch

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $decode_nj \
    data/${supervised_set}_sp_max2_hires_nopitch exp/nnet3${nnet3_affix}/extractor \
    exp/nnet3${nnet3_affix}/ivectors_${supervised_set}_sp_hires || exit 1
fi

if [ $stage -le 9 ]; then
  if [ -f data/${unsupervised_set}_sp_hires_nopitch/feats.scp ]; then
    echo "$0: data/${unsupervised_set}_sp_hires_nopitch/feats.scp exists. Remove it or re-run from next stage"
    exit 1
  fi

  utils/data/perturb_data_dir_speed_3way.sh data/$unsupervised_set data/${unsupervised_set}_sp_hires
  utils/data/perturb_data_dir_volume.sh data/${unsupervised_set}_sp_hires

  steps/make_mfcc_pitch_online.sh --nj $decode_nj --cmd "$train_cmd" \
    --mfcc-config conf/mfcc_hires.conf data/${unsupervised_set}_sp_hires || exit 1
  steps/compute_cmvn_stats.sh data/${unsupervised_set}_sp_hires
  utils/fix_data_dir.sh data/${unsupervised_set}_sp_hires

  utils/data/limit_feature_dim.sh 0:39 \
    data/${unsupervised_set}_sp_hires data/${unsupervised_set}_sp_hires_nopitch || exit 1;
  steps/compute_cmvn_stats.sh \
    data/${unsupervised_set}_sp_hires_nopitch || exit 1;
  utils/fix_data_dir.sh data/${unsupervised_set}_sp_hires_nopitch
fi

if [ $stage -le 10 ]; then
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    data/${unsupervised_set}_sp_hires_nopitch data/${unsupervised_set}_sp_max2_hires_nopitch

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $decode_nj \
    data/${unsupervised_set}_sp_max2_hires_nopitch exp/nnet3${nnet3_affix}/extractor \
    exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_sp_hires || exit 1
fi

if [ $stage -le 11 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $decode_nj \
    data/${unsupervised_set}_sp_max2_hires_nopitch $src_extractor \
    $(dirname $src_extractor)/ivectors_${unsupervised_set}_sp_hires || exit 1
fi

if [ $stage -le 12 ]; then
  echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $srcdir"
  steps/nnet3/decode_semisup.sh --num-threads 4 --nj $decode_nj --cmd "$decode_cmd" \
            --acwt 1.0 --post-decode-acwt 10.0 --write-compact false --skip-scoring true \
            --online-ivector-dir $(dirname $src_extractor)/ivectors_${unsupervised_set}_sp_hires \
            --frames-per-chunk 160 \
            --extra-left-context $extra_left_context \
            --extra-right-context $extra_right_context \
            --extra-left-context-initial 0 --extra-right-context-final 0 \
            --scoring-opts "--min-lmwt 10 --max-lmwt 10" --word-determinize false \
            $graphdir data/${unsupervised_set}_sp_hires $srcdir/decode_${unsupervised_set}_sp
fi
ln -sf ../final.mdl $srcdir/decode_${unsupervised_set}_sp/ || true

frame_subsampling_factor=1
if [ -f $srcdir/frame_subsampling_factor ]; then
  frame_subsampling_factor=`cat $srcdir/frame_subsampling_factor`
fi

if [ $stage -le 13 ]; then
  steps/best_path_weights.sh --cmd "${train_cmd}" --acwt 0.1 \
    data/${unsupervised_set}_sp_hires \
    $srcdir/decode_${unsupervised_set}_sp \
    $srcdir/best_path_${unsupervised_set}_sp
  echo $frame_subsampling_factor > $srcdir/best_path_${unsupervised_set}_sp/frame_subsampling_factor
fi

cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1

sup_ali_dir=exp/tri5_cleaned

diff $treedir/tree $srcdir/tree || { echo "$0: $treedir/tree and $srcdir/tree differ"; exit 1; }

dir=exp/chain${chain_affix}/tdnn_lstm${affix}_sp

if [ $stage -le 14 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${treedir} ${srcdir}/best_path_${unsupervised_set}_sp \
    $dir
fi

if [ $stage -le 15 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=40"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=43 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$hidden_dim

  fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
  fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$hidden_dim
  fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=$hidden_dim
  fast-lstmp-layer name=lstm4 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts

  output-layer name=output input=lstm4 output-delay=$label_delay dim=$num_targets include-log-softmax=false max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm4 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

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

egs_left_context=`perl -e "print int($left_context + $frame_subsampling_factor / 2)"`
egs_right_context=`perl -e "print int($right_context + $frame_subsampling_factor / 2)"`
egs_left_context_initial=`perl -e "print int($left_context_initial + $frame_subsampling_factor / 2)"`
egs_right_context_final=`perl -e "print int($right_context_final + $frame_subsampling_factor / 2)"`

supervised_set=${supervised_set}_sp
if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set}

  if [ $stage -le 16 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
    mkdir -p $sup_egs_dir/
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the supervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
               --left-context $egs_left_context --right-context $egs_right_context \
               --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
               --frame-subsampling-factor $frame_subsampling_factor \
               --alignment-subsampling-factor 3 \
               --frames-per-eg $frames_per_eg \
               --frames-per-iter 1500000 \
               --cmvn-opts "$cmvn_opts" \
               --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${supervised_set}_hires \
               --generate-egs-scp true \
               data/${supervised_set}_hires $dir \
               $sup_lat_dir $sup_egs_dir
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

unsupervised_set=${unsupervised_set}_sp
unsup_lat_dir=${srcdir}/decode_${unsupervised_set}

if [ -z "$unsup_egs_dir" ]; then
  [ -z $unsup_frames_per_eg ] && [ ! -z "$frames_per_eg" ] && unsup_frames_per_eg=$frames_per_eg
  unsup_egs_dir=$dir/egs_${unsupervised_set}

  if [ $stage -le 17 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi
    mkdir -p $unsup_egs_dir
    touch $unsup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the unsupervised data"
    if $use_smart_splitting; then
      get_egs_script=steps/nnet3/chain/get_egs_split.sh
    else
      get_egs_script=steps/nnet3/chain/get_egs.sh
    fi

    $get_egs_script --cmd "$decode_cmd --h-rt 100:00:00" --alignment-subsampling-factor 1 \
               --left-tolerance $tolerance --right-tolerance $tolerance \
               --left-context $egs_left_context --right-context $egs_right_context \
               --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
               --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
               --frame-subsampling-factor $frame_subsampling_factor \
               --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
               --lattice-prune-beam "$lattice_prune_beam" \
               --deriv-weights-scp $srcdir/best_path_${unsupervised_set}/weights.scp \
               --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${unsupervised_set}_hires \
               --generate-egs-scp true $unsup_egs_opts \
               data/${unsupervised_set}_hires $dir \
               $unsup_lat_dir $unsup_egs_dir
  fi
fi

comb_egs_dir=$dir/comb_egs

if [ $stage -le 18 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 128 \
    --lang2weight $supervision_weights --lang2num-copies "$num_copies" \
    2 $sup_egs_dir $unsup_egs_dir $comb_egs_dir
  touch $comb_egs_dir/.nodelete # keep egs around when that run dies.
fi

if [ $train_stage -le -4 ]; then
  train_stage=-4
fi

if [ $stage -le 19 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --egs.dir "$comb_egs_dir" \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${supervised_set}_hires \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights $apply_deriv_weights \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --chain.smbr-leaky-hmm-coefficient $smbr_leaky_hmm_coefficient \
    --chain.mmi-factor-schedule="$mmi_factor_schedule" \
    --chain.smbr-factor-schedule="$smbr_factor_schedule" \
    --chain.smbr-extra-opts="$chain_smbr_extra_opts" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch "$minibatch_size" \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --cleanup.remove-egs false \
    --feat-dir data/${supervised_set}_hires \
    --tree-dir $treedir \
    --lat-dir $sup_lat_dir \
    --dir $dir --lang data/lang_chain || exit 1;
fi

graph_dir=$dir/graph${test_graph_affix}
if [ $stage -le 20 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 $test_lang $dir $graph_dir
fi

wait;
exit 0;
