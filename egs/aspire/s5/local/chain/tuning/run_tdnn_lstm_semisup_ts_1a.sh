#!/bin/bash

set -e

# configs for 'chain'
affix=v8

stage=7   # skip ivector extractor training as it is already done for baseline system
train_stage=-10
get_egs_stage=-10
nj=70

# seed model params
src_dir=exp/chain_norvb/tdnn_lstm_1a
treedir=exp/chain_norvb/tree_bi_a
src_ivector_dir=exp/nnet3_norvb/ivectors_train_sp

tdnn_affix=1a
chain_affix=_semisup_ts

hidden_dim=1024
cell_dim=1024
projection_dim=256

# training options
num_epochs=2
minibatch_size=64,32
chunk_left_context=40
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.025
label_delay=5

# decode options
extra_left_context=50
extra_right_context=0
decode_iter=

# training options
num_epochs=4
remove_egs=false
common_egs_dir=

num_data_reps=3
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

decode_graph_affix=_pp
decode_lang=data/lang_pp_test
rescore_lang=data/lang_pp_test_fg

train_set=train_rvb

lat_dir=$src_dir/decode_${train_set}_lats  # training lattices directory
norvb_lat_dir=$src_dir/decode_${rorvb_train_set}_lats  # training lattices directory
dir=exp/chain${chain_affix}/tdnn_lstm${tdnn_affix}
train_data_dir=data/${train_set}_hires
train_ivector_dir=exp/nnet3/ivectors_${train_set}_hires
lang=data/lang_chain

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage --num-data-reps 3 || exit 1

get_egs_script=steps/nnet3/chain/get_egs.sh
egs_opts=

if $use_transcripts; then
  if [ $stage -le 7 ]; then
    steps/nnet3/align_lats.sh --nj $nj --cmd "$decode_cmd" \
      --acoustic-scale 1.0 --scale-opts "--transition-scale=1.0 --self-loop-scale=1.0" \
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk 160 \
      --online-ivector-dir $src_ivector_dir \
      --skip-scoring true \
      $norvb_train_data_dir data/lang_chain $src_dir $norvb_lat_dir || exit 1
  fi
else
  decode_graph_dir=$src_dir/graph_${decode_graph_affix}

  if [ $stage -le 7 ]; then
    # Note: it might appear that this $lang directory is mismatched, and it is as
    # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
    # the lang directory.
    utils/mkgraph.sh --self-loop-scale 1.0 ${decode_lang} $dir $graph_dir
  fi

  if [ $stage -le 8 ]; then
    steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --write-compact false --determinize-opts "--word-determinize=false" \
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk 160 \
      --online-ivector-dir $src_ivector_dir \
      --skip-scoring true \
      $decode_graph_dir $norvb_train_data_dir $norvb_lat_dir || exit 1

    steps/lmrescore_const_arpa_undeterminized.sh --cmd "$decode_cmd" \
       --write-compact false --acwt 0.1 --beam 8.0 --skip-scoring true \
       ${decode_lang} ${decode_lang}_fg $norvb_train_data_dir \
       ${norvb_lat_dir}_fg || exit 1
  fi
  norvb_lat_dir=${norvb_lat_dir}_fg
fi

if [ $stage -le 8 ]; then
  mkdir -p $lat_dir

  utils/split_data.sh data/${train_set} $nj

  for n in `seq $nj`; do
    awk '{print $1}' data/${train_set}/split$nj/$n/utt2spk | \
      perl -ane 's/rev[1-3]_//g' > $lat_dir/uttlist.$n.$nj
  done

  rm -f $lat_dir/lat_tmp.*.{ark,scp} 2>/dev/null

  norvb_nj=$(cat $norvb_lat_dir/num_jobs)
  $train_cmd JOB=1:$norvb_nj $lat_dir/JOB/copy_lattices.JOB.log \
    lattice-copy "ark:gunzip -c $norvb_lat_dir/lat.JOB.gz |" \
    ark,scp:$lat_dir/lat_tmp.JOB.ark,$lat_dir/lat_tmp.JOB.scp || exit 1

  for n in `seq 3`; do
    cat $lat_dir/lat_tmp.*.scp | awk '{print "rev"n"_"$1" "$2}'
  done > $lat_dir/lat_rvb.scp

  $train_cmd JOB=1:$nj $lat_dir/JOB/copy_rvb_lattices.JOB.log \
    "scp:utils/filter_scp.pl data/${train_set}/split$nj/JOB/utt2spk $lat_dir/lat_rvb.scp |" \
    "ark:gzip -c > $lat_dir/lat.JOB.gz |" || exit 1

  rm $lat_dir/lat_tmp.* $lat_dir/lat_rvb.scp

  echo $nj > $lat_dir/num_jobs

  for f in cmvn_opts final.mdl splice_opts tree frame_subsampling_factor; do
    if [ -f $norvb_lat_dir/$f ]; then cp $norvb_lat_dir/$f $lat_dir/$f; fi 
  done

  get_egs_script=steps/nnet3/chain/get_egs_split.sh
  egs_opts="--lattice-lm-scale 0.5 --lattice-beam 4.0 --tolerance 1"
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=40"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

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

  ## adding the layers for chain branch
  output-layer name=output input=lstm4 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

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

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  mkdir -p $dir/egs
  touch $dir/egs/.nodelete # keep egs around when that run dies.

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch 64,32 \
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
    --egs.stage $get_egs_stage --egs.get-egs-script=$get_egs_script \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true $egs_opts" \
    --egs.chunk-width 160,140,110,80 \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph_pp
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_pp_test $dir $graph_dir
fi

if [ $stage -le 14 ]; then
#%WER 27.8 | 2120 27217 | 78.2 13.6 8.2 6.0 27.8 75.9 | -0.613 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iterfinal_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
  local/nnet3/prep_test_aspire.sh --stage 1 --decode-num-jobs 30 --affix "$affix" \
   --acwt 1.0 --post-decode-acwt 10.0 \
   --window 10 --overlap 5 \
   --sub-speaker-frames 6000 --max-count 75 --ivector-scale 0.75 \
   --pass2-decode-opts "--min-active 1000" \
   --extra-left-context $extra_left_context \
   --extra-right-context $extra_right_context \
   --extra-left-context-initial 0 \
   --extra-right-context-final 0 \
   dev_aspire data/lang $dir/graph_pp $dir
fi

#if [ $stage -le 15 ]; then
#  #Online decoding example
# %WER 31.5 | 2120 27224 | 74.0 13.0 13.0 5.5 31.5 77.1 | -0.558 | exp/chain/tdnn_7b_online/decode_dev_aspire_whole_uniformsegmented_win10_over5_v9_online_iterfinal_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys

#  local/nnet3/prep_test_aspire_online.sh --stage 2 --decode-num-jobs 30 --affix "v7" \
#   --acwt 1.0 --post-decode-acwt 10.0 \
#   --window 10 --overlap 5 \
#   --max-count 75 \
#   --pass2-decode-opts "--min-active 1000" \
#   dev_aspire data/lang $dir/graph_pp exp/chain/tdnn_7b
#fi




exit 0;

# %WER 32.7 | 2120 27222 | 73.6 15.3 11.2 6.3 32.7 78.5 | -0.530 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter100_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 30.4 | 2120 27211 | 74.8 12.7 12.5 5.1 30.4 77.0 | -0.458 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter200_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys
# %WER 29.1 | 2120 27216 | 76.6 13.8 9.6 5.7 29.1 76.8 | -0.527 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter300_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 28.8 | 2120 27211 | 77.0 13.8 9.2 5.8 28.8 76.3 | -0.587 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter400_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 28.7 | 2120 27218 | 77.1 13.8 9.1 5.8 28.7 77.0 | -0.566 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter500_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 28.5 | 2120 27210 | 77.5 13.9 8.7 6.0 28.5 76.1 | -0.596 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter600_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 28.2 | 2120 27217 | 77.0 12.4 10.6 5.2 28.2 75.8 | -0.540 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter700_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys
# %WER 28.4 | 2120 27218 | 77.6 13.6 8.8 6.0 28.4 76.3 | -0.607 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter800_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 28.2 | 2120 27208 | 77.4 12.6 10.0 5.6 28.2 76.6 | -0.555 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter900_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys
# %WER 27.8 | 2120 27214 | 78.0 13.5 8.5 5.9 27.8 75.9 | -0.631 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter1000_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 27.9 | 2120 27216 | 77.6 13.0 9.4 5.5 27.9 76.1 | -0.544 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter1200_pp_fg/score_10/penalty_0.0/ctm.filt.filt.sys
# %WER 27.8 | 2120 27216 | 77.4 13.1 9.5 5.3 27.8 75.7 | -0.615 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter1300_pp_fg/score_9/penalty_0.25/ctm.filt.filt.sys
# %WER 27.7 | 2120 27220 | 78.1 13.6 8.3 5.8 27.7 75.1 | -0.569 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter1400_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
# %WER 27.7 | 2120 27217 | 78.1 13.6 8.3 5.9 27.7 75.1 | -0.605 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iter1500_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
