#!/bin/bash
set -e

# This is fisher chain recipe for training a model on a subset of around 100 hours.
# This is similar to _b, but uses a bi-phone tree with 7000 leaves

# configs for 'chain'
stage=0
tdnn_affix=7smbr_d
nj=80
train_stage=-10
get_egs_stage=-10
decode_iter=
train_set=train_sup_sp_silEx
ivector_train_set=train_sup
tree_affix=bi_d
nnet3_affix=
chain_affix=
exp=exp/semisup_100k
gmm=tri4a
hidden_dim=725
preserve_model_interval=10

# training options
num_epochs=4
xent_regularize=0.1
smbr_xent_regularize=
extra_opts="--chain.mmi-factor-schedule=1.0,1.0@0.1,0.5@0.2,0.5 --chain.smbr-factor-schedule=0.0,0.0@0.1,0.5@0.2,0.5"
extra_egs_opts=
chain_smbr_extra_opts=
smbr_leaky_hmm_coefficient=0.00001
leaky_hmm_coefficient=0.1
l2_regularize=0.0  # 00005
remove_egs=false
common_egs_dir=
apply_deriv_weights=false
xent_lrfactor=1.0

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

gmm_dir=$exp/$gmm   # used to get training lattices (for chain supervision)
treedir=$exp/chain${chain_affix}/tree_${tree_affix}
lat_dir=$exp/chain${chain_affix}/$(basename $gmm_dir)_${train_set}_unk_lats  # training lattices directory
dir=$exp/chain${chain_affix}/tdnn${tdnn_affix}_sp_silEx
train_data_dir=data/${train_set}_hires
train_ivector_dir=$exp/nnet3${nnet3_affix}/ivectors_${train_set}_hires
lang=data/lang_chain_unk

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

#local/semisup/nnet3/run_ivector_common.sh --stage $stage --exp $exp \
#                                  --speed-perturb true \
#                                  --train-set $ivector_train_set \
#                                  --ivector-train-set $ivector_train_set \
#                                  --nnet3-affix "$nnet3_affix" || exit 1

if [ $stage -le 6 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/${train_set}
  steps/compute_cmvn_stats.sh data/${train_set}
  utils/fix_data_dir.sh data/${train_set}
fi

if [ $stage -le 7 ]; then
  utils/copy_data_dir.sh data/${train_set} data/${train_set}_hires
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" --nj $nj \
    data/${train_set}_hires 
  steps/compute_cmvn_stats.sh data/${train_set}_hires
  utils/fix_data_dir.sh data/${train_set}_hires
fi

if [ $stage -le 8 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_hires data/${train_set}_hires/${train_set}_hires_max2

  steps/online/nnet2/extract_ivectors_online.sh --nj $nj \
    data/${train_set}_hires/${train_set}_hires_max2 \
    $exp/nnet3${nnet3_affix}/extractor \
    $train_ivector_dir
fi

cp data/${train_set}/allowed_lengths.txt data/${train_set}_hires

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    --generate-ali-from-lats true data/${train_set} \
    data/lang_unk $gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi

if [ ! -f $lang/topo ]; then
  echo "Could not find $lang/topo"
  exit 1
fi

if [ ! -f $treedir/final.mdl ]; then
  echo "Could not find $treedir/final.mdl"
  exit 1
fi


num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
num_jobs=$(cat $treedir/num_jobs)
if [ $stage -le 12 ]; then
  $train_cmd JOB=1:$num_jobs $dir/log/get_priors.JOB.log \
    gunzip -c $treedir/ali.JOB.gz \| ali-to-pdf $treedir/final.mdl ark:- ark:- \| \
    ali-to-post ark:- ark:- \| post-to-feats --post-dim=$num_targets ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| \
    vector-sum ark:- $dir/pdf_priors.JOB || exit 1

  vector-sum --binary=false $(for n in `seq $num_jobs`; do echo $dir/pdf_priors.$n; done | tr '\n' ' ') - | \
    awk 'BEGIN{a=" ["} {for (i=2; i<NF; i++) sum+=$i; for (i=2; i<NF; i++) a=a" "log(($i+0.00001)/(sum+0.00001)); a=a" ]"} END{print a}' > $dir/pdf_priors
fi


if [ $stage -le 13 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize * $xent_lrfactor)" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$hidden_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1,2) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$hidden_dim
  relu-batchnorm-layer name=tdnn6 input=Append(-6,-3,0) dim=$hidden_dim

  ## adding the layers for chain branch
  relu-batchnorm-layer name=tdnn7 dim=$hidden_dim target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=output.affine offset-file=$dir/pdf_priors learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 14 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_english-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  mkdir -p $dir/egs
  touch $dir/egs/.nodelete # keep egs around when that run dies.

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.smbr-xent-regularize=$smbr_xent_regularize \
    --chain.leaky-hmm-coefficient $leaky_hmm_coefficient \
    --chain.smbr-leaky-hmm-coefficient $smbr_leaky_hmm_coefficient \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights $apply_deriv_weights \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch "150=128,64/300=100,64,32/600=50,32,16/1200=16,8" \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true --no-chunking true $extra_egs_opts" \
    --egs.chunk-width "" \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --chain.smbr-extra-opts="$chain_smbr_extra_opts" \
    --cleanup.preserve-model-interval $preserve_model_interval \
    --dir $dir --lang $lang $extra_opts || exit 1;
fi

graph_dir=$dir/graph_poco_unk
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_poco_test_unk $dir $graph_dir
fi

decode_suff=
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in dev test; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $num_jobs --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}${decode_suff} || exit 1;
      ) &
  done
fi
wait;
exit 0;
