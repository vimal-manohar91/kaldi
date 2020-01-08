#!/bin/bash

# This script does MMI + KL training using TDNN + LSTM layers.
# The seed model is trained on 300 hours subset of Fisher.
# It is adapted to 80 hours of unsupervised AMI-IHM data and 300 hours of supervised Fisher data.
# This script trains network in a semi-supervised fashion.
# This script is computes numerator posteriors from the 
# unsplit lattices.
# This script is similar to _1k, but actual separate output-layers with 
# separate parameter matrices.

set -e -o pipefail -u

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
nj=70
max_jobs_run=30
exp=exp/semisup300k

# seed model params
src_dir=exp/semisup300k/chain/tdnn_lstm_1b_sp
treedir=exp/semisup300k/chain/tree_bi_b
src_ivector_extractor=exp/nnet3/extractor

tgt_data_dir=data/ami_ihm_train

student_mfcc_config=conf/mfcc_hires_16kHz.conf

student_graph_affix=_pp
student_lang=data/lang_pp_test

sup_lat_dir=exp/semisup300k/chain/tri5b_train_300k_rvb_sp_lats
supervised_set=train_300k_rvb_sp
supervision_weights=1,1
num_copies=1,1

tgt_graph_affix=_ami
tgt_lang=data/lang_ami

lm_weights=1,3   # src, tgt weight

tdnn_affix=_1l
chain_affix=_semisup_ts_ami_ihm
nnet3_affix=_semisup_ts_ami_ihm

kl_factor_schedule="output-0=0,0 output-1=0,0"
mmi_factor_schedule="output-0=1,1 output-1=1,1"

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
sup_egs_dir=
unsup_egs_dir=
unsup_frames_per_eg=150

lattice_lm_scale=0.5
kl_fst_scale=0.5
unsup_egs_opts=""
train_opts=

# decode options
test_sets="ami_ihm_dev_16kHz ami_ihm_eval_16kHz"
decode_iter=

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

unsup_lat_dir=$src_dir/decode${student_graph_affix}_${tgt_dataset}_sp   # training lattices directory

dir=$exp/chain${chain_affix}/tdnn_lstm${tdnn_affix}

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

if [ $stage -le -1 ]; then
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

if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh data/${supervised_set}_hires data/${supervised_set}_16kHz
  utils/data/resample_data_dir.sh 16000 data/${supervised_set}_16kHz

  utils/combine_data.sh data/${supervised_set}_16kHz_${tgt_dataset}_sp \
    data/${supervised_set}_16kHz ${teacher_data_dir}
fi

student_data_dir=${tgt_data_dir}_16kHz_sp_hires

local/semisup/nnet3/run_student_ivector_common.sh \
  --nnet3-affix "${nnet3_affix}_${tgt_dataset}" \
  --orig-data-dir data/${supervised_set}_16kHz_${tgt_dataset}_sp \
  --student-data-dir data/${supervised_set}_16kHz_${tgt_dataset}_sp_hires \
  --student-mfcc-config $student_mfcc_config \
  --stage $stage \
  --test-sets "$test_sets"

if [ $stage -le 8 ]; then
  utils/subset_data_dir.sh --utt-list ${teacher_data_dir}/utt2spk \
    data/${supervised_set}_16kHz_${tgt_dataset}_sp_hires \
    $student_data_dir

  utils/subset_data_dir.sh --utt-list data/${supervised_set}_16kHz/utt2spk \
    data/${supervised_set}_16kHz_${tgt_dataset}_sp_hires \
    data/${supervised_set}_16kHz_hires
fi

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

if [ $stage -le 12 ]; then
  steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats $lm_weights \
    --cmd "$train_cmd" \
    $treedir $src_dir/best_path${student_graph_affix}_${tgt_dataset}_sp \
    $dir
fi

deriv_weights_scp=$src_dir/best_path${student_graph_affix}_${tgt_dataset}_sp/weights.scp

if [ $stage -le 13 ]; then
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

  ## adding the layers for chain branch
  # Previous version of the script used output as output-0
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5
  output-layer name=output-0 input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5
  output name=output-1 input=output.affine@$label_delay 

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
  output-layer name=output-0-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
  output name=output-1-xent input=output-xent.log-softmax@$label_delay 
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

frame_subsampling_factor=1
if [ -f $src_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $src_dir/frame_subsampling_factor) || exit 1
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
      --online-ivector-dir exp/nnet3${nnet3_affix}_${tgt_dataset}/ivectors_${supervised_set}_16kHz_${tgt_dataset}_sp_hires \
      --generate-egs-scp true \
      data/${supervised_set}_16kHz_hires $dir $sup_lat_dir $sup_egs_dir
    touch $sup_egs_dir/.nodelete # keep egs around when that run dies.
  fi
else
  frames_per_eg=$(cat $sup_egs_dir/info/frames_per_eg)
fi

unsup_egs_opts="$unsup_egs_opts --deriv-weights-scp $deriv_weights_scp"

if [ -z "$unsup_egs_dir" ]; then
  unsup_egs_dir=$dir/egs_${tgt_dataset}_sp
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
      --cmvn-opts "$cmvn_opts" --lattice-lm-scale $lattice_lm_scale \
      --lattice-prune-beam 4.0 --kl-latdir $unsup_lat_dir --kl-fst-scale $kl_fst_scale \
      --deriv-weights-scp $deriv_weights_scp \
      --online-ivector-dir exp/nnet3${nnet3_affix}_${tgt_dataset}/ivectors_${supervised_set}_16kHz_${tgt_dataset}_sp_hires \
      --generate-egs-scp true $unsup_egs_opts \
      $student_data_dir $dir \
      $unsup_lat_dir $unsup_egs_dir

    touch $unsup_egs_dir/.nodelete
  fi
fi

if [ $stage -le 18 ]; then
  steps/nnet3/chain/multilingual/combine_egs.sh --cmd "$train_cmd" \
    --block-size 128 \
    --lang2weight $supervision_weights --lang2num-copies "$num_copies" \
    2 $sup_egs_dir $unsup_egs_dir $dir/egs_comb
fi

if [ $train_stage -le -4 ]; then
  train_stage=-4
fi

if [ $stage -le 19 ]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd --mem 4G" --train-queue-opt "--h-rt 00:20:00" \
    --combine-queue-opt "--h-rt 00:50:00" \
    --feat.online-ivector-dir exp/nnet3${nnet3_affix}_${tgt_dataset}/ivectors_${supervised_set}_16kHz_${tgt_dataset}_sp_hires \
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
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true $unsup_egs_opts --max-jobs-run $max_jobs_run" \
    --chain.right-tolerance 1 --chain.left-tolerance 1 \
    --chain.alignment-subsampling-factor 1 \
    --egs.chunk-width 160,140,110,80 \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$dir/egs_comb" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $student_data_dir \
    --tree-dir $treedir \
    --lat-dir $sup_lat_dir \
    --dir $dir $train_opts || exit 1;
fi

graph_dir=$dir/graph${tgt_graph_affix}
if [ $stage -le 20 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${tgt_lang} $dir $graph_dir
fi

if [ $stage -le 21 ]; then
  rm -f $dir/.error
  for dset in $test_sets; do
    (
      decode_dir=$dir/decode${tgt_graph_affix}_${dset}_iter${decode_iter}

      iter_opts=
      if [ ! -z "$decode_iter" ]; then
        iter_opts="--iter $decode_iter"
      fi

      steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
        --acwt 1.0 --post-decode-acwt 10.0 \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 --extra-right-context-final 0 \
        --frames-per-chunk 160 \
        --online-ivector-dir exp/nnet3${nnet3_affix}_${tgt_dataset}/ivectors_${dset} \
        --skip-scoring true $iter_opts \
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
