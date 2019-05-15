#!/bin/bash

set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
train_data_dir=data/apollo11_whole_legal_1a_seg_300k
src_dir=exp/chain/tdnn_lstm_1a
extractor=exp/nnet3/extractor
treedir=exp/chain/tree_bi_a
src_lang=data/lang
lang_test=data/lang_test

nj=100

nnet3_affix=
train_stage=-10
affix=_1a
get_egs_stage=-10

common_egs_dir=
xent_regularize=0.025
num_epochs=6

# training options
srand=0
remove_egs=false
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

utils/lang/check_phones_compatible.sh $src_lang/phones.txt $src_dir/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_lang/phones.txt $treedir/phones.txt || exit 1
utils/lang/check_phones_compatible.sh $src_lang/phones.txt $lang_test/phones.txt || exit 1

diff $treedir/tree $src_dir/tree || \
  { echo "$treedir/tree and $src_dir/tree are different!" && exit 1; }

if [ $stage -le 0 ]; then
  if [ -f ${train_data_dir}_hires/feats.scp ]; then
    echo "$0: ${train_data_dir}_hires/feats.scp exits. Remove it or skip this stage."
    exit 1
  fi

  utils/copy_data_dir.sh ${train_data_dir} ${train_data_dir}_hires
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf \
    --cmd "$train_cmd" --nj $nj \
    ${train_data_dir}_hires
  steps/compute_cmvn_stats.sh ${train_data_dir}_hires
  utils/fix_data_dir.sh ${train_data_dir}_hires
fi

train_id=$(basename $train_data_dir)
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_id}

if [ $stage -le 1 ]; then
  steps/online/nnet2/extract_ivectors_online.sh \
    --cmd "$train_cmd" --nj $nj \
    ${train_data_dir}_hires $extractor \
    exp/nnet3${nnet3_affix}/ivectors_${train_id} || exit 1

  for d in dev; do 
    if [ ! -f exp/nnet3${nnet3_affix}/ivectors_$d/ivector_online.scp ]; then
      steps/online/nnet2/extract_ivectors_online.sh \
        --cmd "$train_cmd" --nj $nj \
        data/${d}_hires $extractor \
        exp/nnet3${nnet3_affix}/ivectors_${d} || exit 1
    fi
  done
fi

decode_opts="--extra-left-context 50 --extra-right-context 0 --extra-left-context-initial 0 --extra-right-context-final 0 --frames-per-chunk 150"
  
graph_affix=
graph_dir=$src_dir/graph${graph_affix}
lat_dir=$src_dir/decode${graph_affix}_${train_id}

if [ $stage -le 2 ]; then
  if [ ! -f $graph_dir/HCLG.fst ]; then
    utils/mkgraph.sh --self-loop-scale 1.0 --transition-scale 1.0 \
      $lang_test $src_dir $graph_dir
  fi
  
  steps/nnet3/decode_semisup.sh --nj $nj --cmd "$decode_cmd" --num-threads 4 \
    --acwt 1.0 --post-decode-acwt 10.0 --write-compact false \
    $decode_opts \
    --online-ivector-dir $train_ivector_dir \
    $graph_dir ${train_data_dir}_hires $lat_dir
fi

best_paths_dir=$src_dir/best_path${graph_affix}_$train_id
if [ $stage -le 3 ]; then
  steps/best_path_weights.sh --cmd "$train_cmd" --acwt 0.1 \
    $lat_dir $best_paths_dir
fi

lang=data/lang_chain

if [ $stage -le 10 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt $src_lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r $src_lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

dir=exp/semisup_300k/tdnnf${affix}

cp $src_dir/phones.txt $best_paths_dir
mkdir -p $dir

if [ $stage -le 11 ]; then
  echo "$0: compute {den,normalization}.fst using weighted phone LM."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$train_cmd" \
    --lm-opts '--num-extra-lm-states=2000' \
    $best_paths_dir $dir || exit 1;
fi

if [ $stage -le 12 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  tdnn_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim-continuous=true"
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
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/fearless_steps-$(date +'%m_%d_%H_%M')/s5b/$dir/egs/storage $dir/egs/storage
  fi
  # exclude phone_LM and den.fst generation training stage
  if [ $train_stage -lt -4 ]; then train_stage=-4 ; fi

  # we use chain model from source to generate lats for target and the
  # tolerance used in chain egs generation using this lats should be 1 or 2 which is
  # (source_egs_tolerance/frame_subsampling_factor)
  # source_egs_tolerance = 5
  chain_opts=(--chain.alignment-subsampling-factor=1 --chain.left-tolerance=1 --chain.right-tolerance=1)

  steps/nnet3/chain/train.py --stage $train_stage ${chain_opts[@]} \
    --cmd="$train_cmd" \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.num-epochs $num_epochs \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --trainer.optimization.initial-effective-lrate=0.002 \
    --trainer.optimization.final-effective-lrate=0.0002 \
    --trainer.frames-per-iter=3000000 \
    --egs.dir "$common_egs_dir" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=true \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-chunk-per-minibatch=64,32 \
    --egs.chunk-width=150 \
    --egs.opts="--frames-overlap-per-eg 0 --generate-egs-scp true --deriv-weights-scp $best_paths_dir/weights.scp --lattice-prune-beam 4.0 --lattice-lm-scale 0.5" \
    --egs.get-egs-script steps/nnet3/chain/get_egs_split.sh \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --feat-dir=${train_data_dir}_hires \
    --tree-dir=$treedir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_test \
    $treedir $treedir/graph || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in dev; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 50 --extra-right-context 0 \
          --extra-left-context-initial 0 --extra-right-context-final 0 \
          --frames-per-chunk 150 \
          --nj $nspk --cmd "$decode_cmd --max-jobs-run 64"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data} \
          $treedir/graph data/${data}_hires ${dir}/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

#if [ $stage -le 16 ]; then
#   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" \
#     $lang_test data/lang_pp_test_fg/ \
#     data/dev_hires ${dir}/decode_dev ${dir}/decode_dev.rescored
#fi

exit 0;
