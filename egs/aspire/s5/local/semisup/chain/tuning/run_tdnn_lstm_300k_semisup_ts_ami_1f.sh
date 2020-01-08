#!/bin/bash

# This script does MMI + KL training using TDNN + LSTM layers.
# The seed model is trained on 300 hours subset of Fisher.
# It is adapted to 80 hours of unsupervised AMI-IHM data.
# This script supports using different lattices 
# for KL training, usually generated using a unigram LM.
# This script is updates existing teacher model instead 
# of training from scratch.
# This script uses phone LM graph to compute numerator posteriors for 
# KL objective.
# This script is same as _e, but trains neural network from scratch.

set -e -o pipefail -u

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
nj=70
max_jobs_run=30

# seed model params
src_dir=exp/semisup300k/chain/tdnn_lstm_1b_sp
treedir=exp/semisup300k/chain/tree_bi_b
src_ivector_extractor=exp/nnet3/extractor

tgt_data_dir=data/ami_sdm1_train

student_mfcc_config=conf/mfcc_hires_16kHz.conf

student_graph_affix=_pp
student_lang=data/lang_pp_test

tgt_graph_affix=_ami
tgt_lang=data/lang_ami

lm_weights=1,0   # src, tgt weight

tdnn_affix=_1f
chain_affix=_semisup_ts_ami_sdm1
nnet3_affix=_semisup_ts_ami_sdm1

kl_factor_schedule="output=0.0,0.0"
mmi_factor_schedule="output=1.0,1.0" 

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

remove_egs=false
common_egs_dir=

egs_opts="--lattice-lm-scale 0.5 --lattice-prune-beam 4.0"
train_opts=

# decode options
test_sets="ami_sdm1_dev_16kHz ami_sdm1_eval_16kHz"

scoring_script=local/score.sh

extra_left_context=50
extra_right_context=0

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

tgt_dataset=$(basename $tgt_data_dir)

lat_dir=$src_dir/decode${student_graph_affix}_${tgt_dataset}_sp   # training lattices directory

dir=exp/chain${chain_affix}/tdnn_lstm${tdnn_affix}
graph_post_dir=$dir/den_post_${tgt_dataset}_sp   # training lattices directory

lang=data/lang_chain

for f in $src_ivector_extractor/final.ie $treedir/final.mdl $src_dir/final.mdl \
  $student_mfcc_config; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
done

utils/lang/check_phones_compatible.sh $lang/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $tgt_lang/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $student_lang/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $treedir/phones.txt data/lang/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_dir/phones.txt data/lang/phones.txt || exit 1

steps/nnet2/check_ivectors_compatible.sh $src_ivector_extractor $src_dir || exit 1
diff $treedir/tree $src_dir/tree || exit 1

teacher_data_dir=data/${tgt_dataset}_sp_hires
teacher_ivector_dir=$(dirname $src_ivector_extractor)/ivectors_${tgt_dataset}

if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh $tgt_data_dir data/${tgt_dataset}_hires
  utils/data/perturb_data_dir_speed_3way.sh data/${tgt_dataset}_hires \
    ${teacher_data_dir}
  utils/data/perturb_data_dir_volume.sh ${teacher_data_dir}

  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" --nj $nj \
    ${teacher_data_dir}
  steps/compute_cmvn_stats.sh ${teacher_data_dir}
  utils/fix_data_dir.sh ${teacher_data_dir}

  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${teacher_data_dir} ${teacher_data_dir}_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${teacher_data_dir}_max2 $src_ivector_extractor \
    $teacher_ivector_dir
fi

student_data_dir=${tgt_data_dir}_16kHz_sp_hires

local/semisup/nnet3/run_student_ivector_common.sh \
  --nnet3-affix "$nnet3_affix" \
  --orig-data-dir ${teacher_data_dir} \
  --student-data-dir $student_data_dir \
  --student-mfcc-config $student_mfcc_config \
  --stage $stage \
  --test-sets "$test_sets"

student_graph_dir=$src_dir/graph${student_graph_affix}

if [ $stage -le 9 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${student_lang} $src_dir $student_graph_dir
fi

if [ $stage -le 10 ]; then
  steps/nnet3/decode_semisup.sh --nj $nj --cmd "$decode_cmd" \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --write-compact false --word-determinize false \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial 0 \
    --extra-right-context-final 0 \
    --frames-per-chunk 160 \
    --online-ivector-dir $teacher_ivector_dir \
    --skip-scoring true \
    $student_graph_dir ${teacher_data_dir} $lat_dir || exit 1

  ln -sf ../final.mdl ${lat_dir}
fi

if [ $stage -le 11 ]; then
  steps/best_path_weights.sh --cmd "$decode_cmd" \
    ${teacher_data_dir} ${lat_dir} \
    $src_dir/best_path${student_graph_affix}_${tgt_dataset}_sp
fi

deriv_weights_scp=$src_dir/best_path${student_graph_affix}_${tgt_dataset}_sp/weights.scp

if [ $stage -le 12 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights \
    --cmd "$train_cmd" \
    $treedir $src_dir/best_path${student_graph_affix}_${tgt_dataset}_sp \
    $dir
fi

if [ $stage -le 13 ]; then
  steps/nnet3/chain/get_chain_graph_post_from_fst.sh --nj $nj --cmd "$decode_cmd" \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial 0 \
    --extra-right-context-final 0 \
    --frames-per-chunk 160 \
    --online-ivector-dir $teacher_ivector_dir \
    ${teacher_data_dir} $src_dir $dir $graph_post_dir || exit 1
fi

if [ $stage -le 14 ]; then
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

egs_opts="$egs_opts --deriv-weights-scp $deriv_weights_scp --graph-posterior-rspecifier scp:$graph_post_dir/numerator_post.scp"

if [ $train_stage -le -4 ]; then
  train_stage=-4
fi

if [ $stage -le 15 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_$(basename $student_data_dir) \
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
    --chain.kl-factor-schedule "$kl_factor_schedule" \
    --chain.mmi-factor-schedule "$mmi_factor_schedule" \
    --egs.stage $get_egs_stage --egs.get-egs-script "steps/nnet3/chain/get_egs_split.sh" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true $egs_opts --max-jobs-run $max_jobs_run" \
    --chain.right-tolerance 1 --chain.left-tolerance 1 \
    --chain.alignment-subsampling-factor 1 \
    --egs.chunk-width 160,140,110,80 \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $student_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir $train_opts || exit 1;
fi

graph_dir=$dir/graph${tgt_graph_affix}
if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${tgt_lang} $dir $graph_dir
fi

if [ $stage -le 17 ]; then
  for dset in $test_sets; do 
    (
      decode_dir=$dir/decode${tgt_graph_affix}_${dset}

      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk 160 \
        --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${dset} \
        --skip-scoring true \
        $graph_dir data/${dset}_hires $decode_dir || { echo "Failed decoding in $decode_dir"; touch $dir/.error; }

      $scoring_script --cmd "$decode_cmd" \
        data/${dset}_hires $graph_dir $decode_dir
    ) &
  done
  wait

  if [ -f $dir/.error ]; then
    echo "Failed decoding."
    exit 1
  fi
fi
