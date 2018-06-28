#!/bin/bash

set -e

# configs for 'chain'
affix=v8

stage=7   # skip ivector extractor training as it is already done for baseline system
train_stage=-10
get_egs_stage=-10
nj=70
max_jobs_run=30
test_stage=0

exp=exp/semisup300k

# seed model params
src_dir=exp/semisup300k/chain_norvb/tdnn_lstm_1a_sp
treedir=exp/semisup300k/chain_norvb/tree_bi_a
src_ivector_extractor=exp/nnet3_norvb/extractor

extractor=exp/nnet3/extractor

sup_lat_dir=exp/semisup300k/chain/tri5b_train_300k_rvb_sp_lats
supervised_set=train_300k_rvb_sp

norvb_unsupervised_set=train
unsupervised_set=train_rvb

tdnn_affix=_1b
chain_affix=_semisup_wt

hidden_dim=1024
cell_dim=1024
projection_dim=256

# training options
num_epochs=1
minibatch_size=64,32
chunk_left_context=40
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
xent_regularize=0.025
label_delay=5

lm_weights=1,3    # 3 - To compensate for using alignments before reverberation.
supervision_weights=1.0,1.0
num_copies=2,1

# decode options
extra_left_context=50
extra_right_context=0
decode_iter=

# training options
remove_egs=false
sup_egs_dir=
unsup_egs_dir=
unsup_egs_opts=

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

decode_graph_affix=_300k_pp
decode_lang=data/lang_300k_pp_test

unsup_lat_dir=$src_dir/decode_${unsupervised_set}  # training lattices directory

norvb_unsup_lat_dir=$src_dir/decode_${norvb_unsupervised_set}  # training lattices directory

dir=$exp/chain${chain_affix}/tdnn_lstm${tdnn_affix}
train_data_dir=data/${unsupervised_set}_hires
lang=data/lang_chain

# The iVector-extraction and feature-dumping parts are the same as the standard
# oracle chain setup, and you can skip them by setting "--stage 7" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage --num-data-reps 3 || exit 1

# i-vector extraction for the clean data is same as in the norvb system.
# You can skip this if it's already done.
unsup_src_ivector_dir=$(dirname $src_ivector_extractor)/ivectors_${norvb_unsupervised_set}_hires
if [ $stage -le 7 ]; then
  # Get features for "clean" train set
  utils/copy_data_dir.sh data/${norvb_unsupervised_set} data/${norvb_unsupervised_set}_hires
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $nj --cmd "$train_cmd" \
    data/${norvb_unsupervised_set}_hires
  steps/compute_cmvn_stats.sh data/${norvb_unsupervised_set}_hires
  utils/fix_data_dir.sh data/${norvb_unsupervised_set}_hires

  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${norvb_unsupervised_set}_hires data/${norvb_unsupervised_set}_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    data/${norvb_unsupervised_set}_hires_max2 $src_ivector_extractor \
    $unsup_src_ivector_dir
fi

decode_graph_dir=$src_dir/graph${decode_graph_affix}

if [ $stage -le 8 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${decode_lang} $src_dir $decode_graph_dir
fi

if [ $stage -le 9 ]; then
  steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --write-compact false --determinize-opts "--word-determinize=false" \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial 0 \
    --extra-right-context-final 0 \
    --frames-per-chunk 160 \
    --online-ivector-dir $unsup_src_ivector_dir \
    --skip-scoring true \
    $decode_graph_dir data/${norvb_unsupervised_set}_hires $norvb_unsup_lat_dir || exit 1
fi

if [ $stage -le 11 ]; then
  mkdir -p $unsup_lat_dir

  utils/split_data.sh data/${unsupervised_set}_hires $nj

  #for n in `seq $nj`; do
  #  awk '{print $1}' data/${unsupervised_set}/split$nj/$n/utt2spk | \
  #    perl -ane 's/rev[1-3]_//g' > $unsup_lat_dir/uttlist.$n.$nj
  #done

  rm -f $unsup_lat_dir/lat_tmp.*.{ark,scp} 2>/dev/null

  # Copy the lattices temporarily
  norvb_nj=$(cat $norvb_unsup_lat_dir/num_jobs)
  $train_cmd --max-jobs-run $max_jobs_run JOB=1:$norvb_nj $unsup_lat_dir/log/copy_lattices.JOB.log \
    lattice-copy --write-compact=false "ark:gunzip -c $norvb_unsup_lat_dir/lat.JOB.gz |" \
    ark,scp:$unsup_lat_dir/lat_tmp.JOB.ark,$unsup_lat_dir/lat_tmp.JOB.scp || exit 1

  # Make copies of utterances for perturbed data
  for n in `seq 3`; do
    cat $unsup_lat_dir/lat_tmp.*.scp | awk -v n=$n '{print "rev"n"_"$1" "$2}'
  done | sort -k1,1 > $unsup_lat_dir/lat_rvb.scp

  # Copy and dump the lattices for perturbed data
  $train_cmd --max-jobs-run $max_jobs_run JOB=1:$nj $unsup_lat_dir/log/copy_rvb_lattices.JOB.log \
    lattice-copy --write-compact=false \
    "scp:utils/filter_scp.pl data/${unsupervised_set}_hires/split$nj/JOB/utt2spk $unsup_lat_dir/lat_rvb.scp |" \
    "ark:| gzip -c > $unsup_lat_dir/lat.JOB.gz" || exit 1

  rm $unsup_lat_dir/lat_tmp.* $unsup_lat_dir/lat_rvb.scp

  echo $nj > $unsup_lat_dir/num_jobs

  for f in cmvn_opts final.mdl splice_opts tree frame_subsampling_factor; do
    if [ -f $norvb_unsup_lat_dir/$f ]; then cp $norvb_unsup_lat_dir/$f $unsup_lat_dir/$f; fi 
  done
fi

ln -sf ../final.mdl $unsup_lat_dir/final.mdl

if [ $stage -le 12 ]; then
  steps/best_path_weights.sh --cmd "$decode_cmd" \
    ${norvb_train_data_dir} $decode_lang ${norvb_unsup_lat_dir} \
    $src_dir/best_path_${norvb_unsupervised_set}
fi

if [ $stage -le 13 ]; then
  norvb_weights_dir=$src_dir/best_path_${norvb_unsupervised_set}
  norvb_nj=$(cat $norvb_weights_dir/num_jobs)

  mkdir -p $src_dir/best_path_${unsupervised_set}
  for n in `seq 3`; do
    cat $norvb_weights_dir/weights.scp | awk -v n=$n '{print "rev"n"_"$1" "$2}'
  done | sort -k1,1 > $src_dir/best_path_${unsupervised_set}/weights.scp
fi

deriv_weights_scp=$src_dir/best_path_${unsupervised_set}/weights.scp

if [ $stage -le 14 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${treedir} ${src_dir}/best_path_${norvb_unsupervised_set} $dir
fi

# if [ $stage -le 15 ]; then
#   echo "$0: creating neural net configs using the xconfig parser";
# 
#   num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
#   learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
# 
#   lstm_opts="decay-time=40"
# 
#   mkdir -p $dir/configs
#   cat <<EOF > $dir/configs/network.xconfig
#   input dim=100 name=ivector
#   input dim=40 name=input
# 
#   # please note that it is important to have input layer with the name=input
#   # as the layer immediately preceding the fixed-affine-layer to enable
#   # the use of short notation for the descriptor
#   fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
# 
#   # the first splicing is moved before the lda layer, so no splicing here
#   relu-batchnorm-layer name=tdnn1 dim=$hidden_dim
#   relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$hidden_dim
#   relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$hidden_dim
# 
#   fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
#   relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
#   relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
#   fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
#   relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$hidden_dim
#   relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$hidden_dim
#   fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
#   relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=$hidden_dim
#   relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=$hidden_dim
#   fast-lstmp-layer name=lstm4 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 dropout-proportion=0.0 $lstm_opts
# 
#   ## adding the layers for chain branch
#   output-layer name=output input=lstm4 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5
# 
# 
#   # adding the layers for xent branch
#   # This block prints the configs for a separate output that will be
#   # trained with a cross-entropy objective in the 'chain' models... this
#   # has the effect of regularizing the hidden parts of the model.  we use
#   # 0.5 / args.xent_regularize as the learning rate factor- the factor of
#   # 0.5 / args.xent_regularize is suitable as it means the xent
#   # final-layer learns at a rate independent of the regularization
#   # constant; and the 0.5 was tuned so as to make the relative progress
#   # similar in the xent and regular final layers.
#   output-layer name=output-xent input=lstm4 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
# 
#   output name=output-0 input=output.affine@$label_delay skip-in-init=true
#   output name=output-1 input=output.affine@$label_delay skip-in-init=true
# 
#   output name=output-0-xent input=output-xent.log-softmax@$label_delay skip-in-init=true
#   output name=output-1-xent input=output-xent.log-softmax@$label_delay skip-in-init=true
# EOF
#   steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
# fi

. $src_dir/configs/vars

frame_subsampling_factor=3

left_context=$[model_left_context + chunk_left_context]
right_context=$[model_right_context + chunk_right_context]
left_context_initial=$model_left_context
right_context_final=$model_right_context

egs_left_context=`perl -e "print int($left_context + $frame_subsampling_factor / 2)"`
egs_right_context=`perl -e "print int($right_context + $frame_subsampling_factor / 2)"`
egs_left_context_initial=`perl -e "print int($left_context_initial + $frame_subsampling_factor / 2)"`
egs_right_context_final=`perl -e "print int($right_context_final + $frame_subsampling_factor / 2)"`

cmvn_opts=`cat $src_dir/cmvn_opts` || exit 1

if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set}
  frames_per_eg=$(cat $src_dir/egs/info/frames_per_eg)

  if [ $stage -le 16 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi

    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
      --frame-subsampling-factor $frame_subsampling_factor \
      --alignment-subsampling-factor $frame_subsampling_factor \
      --frames-per-eg $frames_per_eg \
      --frames-per-iter 1500000 \
      --cmvn-opts "$cmvn_opts" \
      --online-ivector-dir $(dirname $extractor)/ivectors_${supervised_set} \
      --generate-egs-scp true \
      data/${supervised_set}_hires $dir $sup_lat_dir $sup_egs_dir
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

if [ -z "$unsup_egs_dir" ]; then
  unsup_egs_dir=$dir/egs_${unsupervised_set}
  [ -z $unsup_frames_per_eg ] && unsup_frames_per_eg=$frames_per_eg
  if [ $stage -le 17 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $unsup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aspire-$(date +'%m_%d_%H_%M')/s5/$unsup_egs_dir/storage $unsup_egs_dir/storage
    fi

    steps/nnet3/chain/get_egs_split.sh --cmd "$decode_cmd" --alignment-subsampling-factor 1 \
      --left-tolerance 1 --right-tolerance 1 \
      --left-context $egs_left_context --right-context $egs_right_context \
      --left-context-initial $egs_left_context_initial --right-context-final $egs_right_context_final \
      --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
      --frame-subsampling-factor $frame_subsampling_factor \
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale 0.5 \
      --lattice-prune-beam 4.0 \
      --deriv-weights-scp $deriv_weights_scp \
      --online-ivector-dir $(dirname $extractor)/ivectors_${unsupervised_set} \
      --generate-egs-scp true $unsup_egs_opts \
      data/${unsupervised_set}_hires $dir \
      $unsup_lat_dir $unsup_egs_dir

    touch $unsup_egs_dir/.nodelete
  fi
fi

if [ $stage -le 18 ]; then
  steps/nnet3/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --minibatch-size 64 --frames-per-iter 1500000 \
    --lang2weight $supervision_weights --egs-prefix cegs. --lang2num-copies "$num_copies" \
    2 $sup_egs_dir $unsup_egs_dir $dir/egs_comb
fi

if [ $train_stage -le -1 ]; then
  train_stage=-1
fi

if [ $stage -le 19 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" --trainer.input-model $src_dir/final.mdl \
    --feat.online-ivector-dir $(dirname $extractor)/ivectors_${unsupervised_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights true \
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
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true $egs_opts --max-jobs-run $max_jobs_run" \
    --chain.right-tolerance 1 --chain.left-tolerance 1 \
    --chain.alignment-subsampling-factor 1 \
    --egs.chunk-width 160,140,110,80 \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir $dir/egs_comb \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${supervised_set}_hires \
    --tree-dir $treedir \
    --lat-dir $sup_lat_dir \
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph_pp
if [ $stage -le 20 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_pp_test $dir $graph_dir
fi

if [ $stage -le 21 ]; then
#%WER 27.8 | 2120 27217 | 78.2 13.6 8.2 6.0 27.8 75.9 | -0.613 | exp/chain/tdnn_7b/decode_dev_aspire_whole_uniformsegmented_win10_over5_v6_200jobs_iterfinal_pp_fg/score_9/penalty_0.0/ctm.filt.filt.sys
  local/nnet3/decode.sh --stage $test_stage --decode-num-jobs 30 --affix "$affix" \
   --acwt 1.0 --post-decode-acwt 10.0 \
   --window 10 --overlap 5 \
   --sub-speaker-frames 6000 --max-count 75 --ivector-scale 0.75 \
   --pass2-decode-opts "--min-active 1000" \
   --extra-left-context $extra_left_context \
   --extra-right-context $extra_right_context \
   --extra-left-context-initial 0 \
   --extra-right-context-final 0 \
   dev_aspire_ldc data/lang $dir/graph_pp $dir
fi

exit 0

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
