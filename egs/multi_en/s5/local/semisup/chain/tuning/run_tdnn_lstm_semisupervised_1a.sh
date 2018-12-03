#!/bin/bash

# Unsupervised set: train_unsup100k_250k
# unsup_frames_per_eg=150
# Deriv weights: Lattice posterior of best path pdf
# Unsupervised weight: 1.0
# Weights for phone LM (supervised, unsupervises): 3,2
# LM for decoding unsupervised data: 4gram

stage=-2
train_stage=-100
nj=400
split_nj=400
test_nj=50

# The following 3 options decide the output directory for semi-supervised 
# chain system
# dir=${exp_root}/chain${chain_affix}/tdnn${tdnn_affix}
multi=multi_a
chain_affix=
tdnn_affix=_semisup_1a

# Data directories
supervised_data=data/multi_a/tri5a
unsupervised_data=data/train_mixer6_1a_seg

# Input seed system
sup_gmm=tri5a
sup_chain_dir=exp/multi_a/chain/tdnn_lstm_1c2_sp_ld5
sup_lat_dir=exp/multi_a/tri5a_lats_nodup_sp
sup_tree_dir=exp/multi_a/chain/tri5a_tree_sp_7000
sup_ivector_dir=exp/multi_a/nnet3/ivectors_multi_a/tri5a_sp
sup_ivector_root_dir=exp/multi_a/nnet3

train_new_ivector=false
nnet3_affix=    # affix for nnet3 -- relates to i-vector used
                # Applicable if training a new i-vector extractor

# Unsupervised options
unsup_decode_opts="--frames-per-chunk 160 --extra-left-context 50 --extra-right-context 0"
unsup_frames_per_eg=150  # if empty will be equal to the supervised model's config -- you will need to change minibatch_size for comb training accordingly
lattice_lm_scale=0.5  # lm-scale for using the weights from unsupervised lattices
lattice_prune_beam=4.0  # If supplied will prune the lattices prior to getting egs for unsupervised data
tolerance=1
phone_insertion_penalty=

# Semi-supervised options
supervision_weights=1.0,1.0
lm_weights=3,1
num_copies=
sup_egs_dir=
unsup_egs_dir=
unsup_egs_opts=

remove_egs=false
common_egs_dir=

relu_dim=1024
cell_dim=1024
projection_dim=256

use_smart_splitting=true

# training options
num_epochs=2
initial_effective_lrate=0.001
final_effective_lrate=0.0001
num_jobs_initial=3
num_jobs_final=16
dropout_schedule='0,0@0.20,0.5@0.50,0'
chunk_left_context=40
chunk_right_context=0
xent_regularize=0.025
label_delay=5

# decode options
extra_left_context=50
extra_right_context=0
frames_per_chunk=

decode_iter=
decode_dir_affix=

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

set -u -e -o pipefail

exp_root=exp/$multi

RANDOM=0

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=$exp_root/chain${chain_affix}/tdnn_lstm${tdnn_affix}

unsup_decode_lang=data/lang_${multi}_${sup_gmm}_fsh_sw1_tg
unsup_rescore_lang=data/lang_${multi}_${sup_gmm}_fsh_sw1_fg
unsup_decode_graph_affix=_${multi}_${sup_gmm}_fsh_sw1
unsup_rescore_graph_affix=_fg

test_lang=data/lang_${multi}_${sup_gmm}_fsh_sw1_tg
test_graph_affix=_fsh_sw1_tg
test_rescore_lang=data/lang_${multi}_${sup_gmm}_fsh_sw1_fg
test_rescore_graph_affix=_fsh_sw1_fg

graphdir=$sup_chain_dir/graph${unsup_decode_graph_affix}
decode_affix=${unsup_decode_graph_affix}

if [ ! -f $graphdir/HCLG.fst ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 $unsup_decode_lang $sup_chain_dir $graphdir
fi

for f in ${supervised_data}_sp_hires/feats.scp; do 
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

if [ $stage -le 1 ]; then
  if [ -f ${unsupervised_data}_sp_hires/feats.scp ]; then
    echo "$0: ${unsupervised_data}_sp_hires/feats.scp exists. Remove it or re-run from next stage"
    exit 1
  fi

  utils/data/perturb_data_dir_speed_3way.sh $unsupervised_data ${unsupervised_data}_sp_hires || exit 1
  utils/data/perturb_data_dir_volume.sh ${unsupervised_data}_sp_hires || exit 1

  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
    --mfcc-config conf/mfcc_hires.conf ${unsupervised_data}_sp_hires || exit 1

  rm ${unsupervised_data}_sp_hires/cmvn.scp 2>/dev/null || true
  utils/fix_data_dir.sh ${unsupervised_data}_sp_hires || exit 1
fi

unsupervised_set=$(basename $unsupervised_data)
if [ $stage -le 2 ]; then
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 \
    ${unsupervised_data}_sp_hires ${unsupervised_data}_sp_max2_hires 

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${unsupervised_data}_sp_max2_hires $sup_ivector_root_dir/extractor \
    $sup_ivector_root_dir/ivectors_${unsupervised_set}_sp_hires || exit 1
fi

if [ $stage -le 5 ]; then
  echo "$0: getting the decoding lattices for the unsupervised subset using the chain model at: $sup_chain_dir"
  steps/nnet3/decode_semisup.sh --num-threads 4 --sub-split $nj --nj $split_nj --cmd "$decode_cmd" \
            --acwt 1.0 --post-decode-acwt 10.0 --write-compact false --skip-scoring true \
            --online-ivector-dir $sup_ivector_root_dir/ivectors_${unsupervised_set}_sp_hires \
            $unsup_decode_opts --keep-subsplit true \
            --scoring-opts "--min-lmwt 10 --max-lmwt 10" --word-determinize false \
            $graphdir ${unsupervised_data}_sp_hires $sup_chain_dir/decode${decode_affix}_${unsupervised_set}_sp || exit 1
fi

if [ $stage -le 6 ]; then
  steps/lmrescore_const_arpa_undeterminized.sh --cmd "$decode_cmd" \
    --scoring-opts "--min-lmwt 10 --max-lmwt 10" --skip-scoring true \
    --write-compact true --acwt 0.1 --beam 8.0 --keep-subsplit true \
    $unsup_decode_lang $unsup_rescore_lang \
    ${unsupervised_data}_sp_hires \
    $sup_chain_dir/decode${decode_affix}_${unsupervised_set}_sp \
    $sup_chain_dir/decode${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix} || exit 1
fi

ln -sf ../final.mdl $sup_chain_dir/decode${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix}/ || true

frame_subsampling_factor=1
if [ -f $sup_chain_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=`cat $sup_chain_dir/frame_subsampling_factor`
fi

if [ $stage -le 7 ]; then
  steps/nnet3/merge_subsplit_lattices.sh \
    --cmd "${train_cmd}" --skip-scoring true --skip-diagnostics true \
    $unsup_decode_lang \
    ${unsupervised_data}_sp_hires \
    $sup_chain_dir/decode${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix} || exit 1
fi

unsup_lat_dir=${sup_chain_dir}/decode${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix}


if [ $stage -le 8 ]; then
  steps/best_path_weights.sh --cmd "${train_cmd}" --acwt 0.1 \
    ${unsupervised_data}_sp_hires \
    $sup_chain_dir/decode${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix} \
    $sup_chain_dir/best_path${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix} || exit 1
fi
echo $frame_subsampling_factor > $sup_chain_dir/best_path${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix}/frame_subsampling_factor

cmvn_opts=`cat $sup_chain_dir/cmvn_opts` || exit 1

if [ ! -f $sup_tree_dir/final.mdl ]; then
  echo "$0: $sup_tree_dir/final.mdl does not exist."
  exit 1
fi

diff $sup_tree_dir/tree $sup_chain_dir/tree || { echo "$0: $sup_tree_dir/tree and $sup_chain_dir/tree differ"; exit 1; }

#if [ $stage -le 9 ]; then
#  steps/subset_ali_dir.sh --cmd "$train_cmd" \
#    data/${unsupervised_set} data/${unsupervised_set}_sp_hires \
#    $sup_chain_dir/best_path_${unsupervised_set}_sp${decode_affix} \
#    $sup_chain_dir/best_path_${unsupervised_set}${decode_affix}
#  echo $frame_subsampling_factor > $sup_chain_dir/best_path_${unsupervised_set}${decode_affix}/frame_subsampling_factor
#fi

if [ $stage -le 10 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights --cmd "$train_cmd" \
    ${sup_tree_dir} ${sup_chain_dir}/best_path${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix} \
    $dir
fi

if [ $stage -le 11 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $sup_tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  lstm_opts="dropout-proportion=0.0 decay-time=40"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$relu_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$relu_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$relu_dim

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$relu_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$relu_dim
  fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$relu_dim
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$relu_dim
  fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

  output name=output-0 input=output.affine@$label_delay
  output name=output-1 input=output.affine@$label_delay

  output name=output-0-xent input=output-xent.log-softmax@$label_delay
  output name=output-1-xent input=output-xent.log-softmax@$label_delay
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

. $dir/configs/vars

egs_left_context=`perl -e "print int($model_left_context + $chunk_left_context + $frame_subsampling_factor / 2)"`
egs_right_context=`perl -e "print int($model_right_context + $chunk_right_context + $frame_subsampling_factor / 2)"`
egs_left_context_initial=`perl -e "print int($model_left_context + $frame_subsampling_factor / 2)"`
egs_right_context_final=`perl -e "print int($model_right_context + $frame_subsampling_factor / 2)"`

supervised_set=$(basename $supervised_data)
if [ -z "$sup_egs_dir" ]; then
  sup_egs_dir=$dir/egs_${supervised_set}_sp
  frames_per_eg=$(cat $sup_chain_dir/egs/info/frames_per_eg)

  if [ $stage -le 12 ]; then
    if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $sup_egs_dir/storage ]; then
      utils/create_split_dir.pl \
       /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$sup_egs_dir/storage $sup_egs_dir/storage
    fi
    mkdir -p $sup_egs_dir/
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.

    echo "$0: generating egs from the supervised data"
    steps/nnet3/chain/get_egs.sh --cmd "$decode_cmd" \
               --left-context $egs_left_context --right-context $egs_right_context \
               --left-context-initial $egs_left_context_initial \
               --right-context-final $egs_right_context_final \
               --frame-subsampling-factor $frame_subsampling_factor \
               --alignment-subsampling-factor 3 \
               --frames-per-eg $frames_per_eg \
               --frames-per-iter 1500000 \
               --cmvn-opts "$cmvn_opts" \
               --online-ivector-dir $sup_ivector_dir \
               --generate-egs-scp true --constrained false \
               ${supervised_data}_sp_hires $dir \
               $sup_lat_dir $sup_egs_dir
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

if [ -z "$unsup_egs_dir" ]; then
  [ -z $unsup_frames_per_eg ] && [ ! -z "$frames_per_eg" ] && unsup_frames_per_eg=$frames_per_eg
  unsup_egs_dir=$dir/egs_${unsupervised_set}_sp

  if [ $stage -le 13 ]; then
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

    $get_egs_script --cmd "$decode_cmd --mem 6G" --alignment-subsampling-factor 1 \
               --left-tolerance $tolerance --right-tolerance $tolerance \
               --left-context $egs_left_context --right-context $egs_right_context \
               --left-context-initial $egs_left_context_initial \
               --right-context-final $egs_right_context_final \
               --frames-per-eg $unsup_frames_per_eg --frames-per-iter 1500000 \
               --frame-subsampling-factor $frame_subsampling_factor \
               --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
               --lattice-prune-beam "$lattice_prune_beam" \
               --phone-insertion-penalty "$phone_insertion_penalty" \
               --deriv-weights-scp $sup_chain_dir/best_path${decode_affix}_${unsupervised_set}_sp${unsup_rescore_graph_affix}/weights.scp \
               --online-ivector-dir $sup_ivector_root_dir/ivectors_${unsupervised_set}_sp_hires \
               --generate-egs-scp true --constrained false $unsup_egs_opts \
               ${unsupervised_data}_sp_hires $dir \
               $unsup_lat_dir $unsup_egs_dir
  fi
fi

comb_egs_dir=$dir/comb_egs

if [ $stage -le 14 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 64 \
    --lang2weight $supervision_weights --lang2num-copies "$num_copies" \
    2 $sup_egs_dir $unsup_egs_dir $comb_egs_dir
  touch $comb_egs_dir/.nodelete # keep egs around when that run dies.
fi

if [ $train_stage -le -4 ]; then
  train_stage=-4
fi

if [ $stage -le 15 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd --mem 4G" \
    --feat.online-ivector-dir $sup_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.dropout-schedule=$dropout_schedule \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context 0 \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$comb_egs_dir" \
    --cleanup.remove-egs false \
    --feat-dir ${supervised_data}_sp_hires \
    --tree-dir $sup_tree_dir \
    --lat-dir $sup_lat_dir \
    --dir $dir  || exit 1;
fi

graph_dir=$dir/graph${test_graph_affix}
if [ $stage -le 17 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 $test_lang $dir $graph_dir
fi

if [ $stage -le 18 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  if [ -z "$frames_per_chunk" ]; then
    frames_per_chunk=$(echo $frames_per_eg | cut -d ',' -f 1)
  fi

  rm $dir/.error 2>/dev/null || true
  for decode_set in eval2000 rt03; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj 50 --cmd "$decode_cmd" $iter_opts \
        --online-ivector-dir $sup_ivector_root_dir/ivectors_${decode_set} \
        --frames-per-chunk $frames_per_chunk \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        $graph_dir data/${decode_set}_hires \
        $dir/decode_${decode_set}${test_graph_affix} || exit 1;
      if [ ! -z "$test_rescore_lang" ]; then
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          $test_lang $test_rescore_lang data/${decode_set}_hires \
          $dir/decode_${decode_set}${test_graph_affix} \
          $dir/decode_${decode_set}${test_rescore_graph_affix} || exit 1;
      fi
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

exit 0

test_online_decoding=true
lang=data/lang_${multi}_${gmm}_fsh_sw1_tg
if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang $sup_ivector_root_dir/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in rt03 eval2000; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj 50 --cmd "$decode_cmd" $iter_opts \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires/test \
         ${dir}_online/decode_${decode_set}${test_graph_affix} || exit 1;
      if $rescore; then
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          $test_lang $test_rescore_lang data/${decode_set}_hires/test \
          ${dir}_online/decode_${decode_set}${test_graph_affix} \
          ${dir}_online/decode_${decode_set}${test_rescore_graph_affix} || exit 1;
      fi
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in online decoding"
    exit 1
  fi
fi

exit 0;
