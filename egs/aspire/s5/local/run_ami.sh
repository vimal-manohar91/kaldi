#!/bin/bash

# Copyright 2018  Vimal Manohar
# Apache 2.0

stage=0
nj=40

semisup_stage=9
kl_stage=12
wt_stage=12
kl_wt_stage=12

semisup_train_stage=-10
semisup_wt_train_stage=-10
kl_train_stage=-10
kl_wt_train_stage=-10

mic=ihm

ivector_root_dir=exp/nnet3

chain_dir=exp/semisup300k/chain/tdnn_lstm_1b_sp

ami_lang_test=/exp/vmanohar/workspace_chain_ts/egs/ami/s5b/data/lang_ami_fsh.o3g.kn.pr1-7
ami_lang_suffix=_ami

extra_left_context=50
extra_right_context=0

student_lang=data/lang_ami 
student_graph_affix=_ami 
tdnn_affix=_1a_amilang

semisup_opts=
kl_opts=
semisup_wt_opts=
kl_wt_opts=

. ./path.sh
. ./cmd.sh

set -e -o pipefail -u

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $0"
  exit 1
fi

ami_train=/exp/vmanohar/workspace_chain_ts/egs/ami/s5b/data/$mic/train
ami_dev=/exp/vmanohar/workspace_chain_ts/egs/ami/s5b/data/$mic/dev
ami_eval=/exp/vmanohar/workspace_chain_ts/egs/ami/s5b/data/$mic/eval

if [ $stage -le -1 ]; then
  utils/copy_data_dir.sh $ami_dev data/ami_${mic}_dev_8kHz_hires
  utils/copy_data_dir.sh $ami_eval data/ami_${mic}_eval_8kHz_hires
fi

if [ $stage -le 1 ]; then
  for dset in ami_${mic}_dev_8kHz ami_${mic}_eval_8kHz; do
    this_nj=$(cat data/${dset}_hires/spk2utt | wc -l)
    if [ $this_nj -gt $nj ]; then
      this_nj=$nj
    fi

    steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" \
      --nj $this_nj data/${dset}_hires
    steps/compute_cmvn_stats.sh data/${dset}_hires
    utils/fix_data_dir.sh data/${dset}_hires
  done
fi

if [ $stage -le 2 ]; then
  for dset in ami_${mic}_dev_8kHz ami_${mic}_eval_8kHz; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" \
      --nj $nj data/${dset}_hires $ivector_root_dir/extractor \
      $ivector_root_dir/ivectors_${dset}
  done
fi

graph_dir=$chain_dir/graph${ami_lang_suffix}
if [ $stage -le 3 ]; then
  cp -rT $ami_lang_test data/lang${ami_lang_suffix}
  diff -i data/lang/phones.txt data/lang${ami_lang_suffix} || { echo "$0: phone-sets differ even ignoring case."; exit 1; }

  cp data/lang/phones.txt data/lang${ami_lang_suffix}

  for x in context_indep disambig extra_questions nonsilence optional_silence \
    roots sets silence word_boundary; do
    diff -i data/lang/phones/$x.txt data/lang${ami_lang_suffix}/phones/$x.txt || \
      { echo "$0: data/lang/phones/$x.txt differ from data/lang${ami_lang_suffix}/phones/$x.txt even ignoring case."; exit 1; }
    cp data/lang/phones/$x.txt data/lang${ami_lang_suffix}/phones
  done

  utils/int2sym.pl -f 3- data/lang/phones.txt \
    data/lang${ami_lang_suffix}/phones/align_lexicon.int | \
    utils/int2sym.pl -f 1-2 data/lang${ami_lang_suffix}/words.txt > \
    data/lang${ami_lang_suffix}/phones/align_lexicon.txt
  utils/validate_lang.pl data/lang${ami_lang_suffix}

  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang${ami_lang_suffix} $chain_dir $graph_dir
fi
scoring_script=local/score_ami.sh

if [ $stage -le 4 ]; then
  for dset in ami_${mic}_dev_8kHz ami_${mic}_eval_8kHz; do
    decode_dir=$chain_dir/decode${ami_lang_suffix}_${dset}

    steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
      --acwt 1.0 --post-decode-acwt 10.0 \
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --extra-left-context-initial 0 --extra-right-context-final 0 \
      --frames-per-chunk 160 \
      --online-ivector-dir $ivector_root_dir/ivectors_${dset} \
      --skip-scoring true \
      $graph_dir data/${dset}_hires $decode_dir

    $scoring_script --cmd "$decode_cmd" \
      data/${dset}_hires $graph_dir $decode_dir || exit 1
  done
fi

if [ $stage -le 5 ]; then
  utils/copy_data_dir.sh $ami_train data/ami_${mic}_train
  utils/copy_data_dir.sh $ami_dev data/ami_${mic}_dev_16kHz
  utils/copy_data_dir.sh $ami_eval data/ami_${mic}_eval_16kHz
fi
if [ $stage -le 6 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_300k_semisup_ts_ami_1a.sh \
    --src-dir exp/semisup300k/chain/tdnn_lstm_1b_sp \
    --treedir exp/semisup300k/chain/tree_bi_b \
    --src-ivector-extractor exp/nnet3/extractor \
    --tgt-data-dir data/ami_${mic}_train \
    --chain-affix _semisup_ts_ami_${mic} \
    --nnet3-affix _semisup_ts_ami_${mic} \
    --scoring-script $scoring_script $semisup_opts \
    --test-sets "ami_${mic}_dev_16kHz ami_${mic}_eval_16kHz" \
    --stage $semisup_stage --train-stage $semisup_train_stage \
    --student-lang ${student_lang} --student-graph-affix ${student_graph_affix} --tdnn-affix ${tdnn_affix}
fi

if [ $stage -le 7 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_300k_kl_ts_ami_1a.sh \
    --src-dir exp/semisup300k/chain/tdnn_lstm_1b_sp \
    --treedir exp/semisup300k/chain/tree_bi_b \
    --src-ivector-extractor exp/nnet3/extractor \
    --tgt-data-dir data/ami_${mic}_train \
    --chain-affix _kl_ts_ami_${mic} \
    --nnet3-affix _semisup_ts_ami_${mic} \
    --test-sets "ami_${mic}_dev_16kHz ami_${mic}_eval_16kHz" \
    --scoring-script $scoring_script $kl_opts \
    --stage $kl_stage --train-stage $kl_train_stage \
    --student-lang ${student_lang} --student-graph-affix ${student_graph_affix} --tdnn-affix ${tdnn_affix}
fi

if [ $stage -le 8 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_300k_semisup_wt_ami_1a.sh \
    --src-dir exp/semisup300k/chain/tdnn_lstm_1b_sp \
    --treedir exp/semisup300k/chain/tree_bi_b \
    --src-ivector-extractor exp/nnet3/extractor \
    --tgt-data-dir data/ami_${mic}_train \
    --chain-affix _semisup_wt_ami_${mic} \
    --nnet3-affix _semisup_ts_ami_${mic} \
    --scoring-script $scoring_script \
    --test-sets "ami_${mic}_dev_8kHz ami_${mic}_eval_8kHz" \
    --stage $wt_stage --train-stage $semisup_wt_train_stage \
    --student-lang ${student_lang} --student-graph-affix ${student_graph_affix} --tdnn-affix ${tdnn_affix} \
    $semisup_wt_opts
fi

exit 1

if [ $stage -le 9 ]; then
  local/semisup/chain/tuning/run_tdnn_lstm_300k_kl_wt_ami_1a.sh \
    --src-dir exp/semisup300k/chain/tdnn_lstm_1b_sp \
    --treedir exp/semisup300k/chain/tree_bi_b \
    --src-ivector-extractor exp/nnet3/extractor \
    --tgt-data-dir data/ami_${mic}_train \
    --chain-affix _kl_wt_ami_${mic} \
    --nnet3-affix _semisup_ts_ami_${mic} \
    --scoring-script $scoring_script \
    --test-sets "ami_${mic}_dev_8kHz ami_${mic}_eval_8kHz" \
    --stage $kl_wt_stage --train-stage $kl_wt_train_stage \
    --student-lang ${student_lang} --student-graph-affix ${student_graph_affix} --tdnn-affix ${tdnn_affix} \
    $kl_wt_opts
fi
