#!/bin/bash

nj=40
mic=sdm1

graph_affix=_pp
chain_dir=exp/semisup300k/chain/tdnn_lstm_1b_sp

ivector_root_dir=exp/nnet3

extra_left_context=50
extra_right_context=0

scoring_script=local/score_ami.sh

. ./cmd.sh
. ./path.sh

set -e -o pipefail -u

. utils/parse_options.sh

rm -f $chain_dir/.error

graph_dir=$chain_dir/graph_pp

for dset in ami_${mic}_dev_8kHz ami_${mic}_eval_8kHz; do
  (
  decode_dir=$chain_dir/decode_${dset}_pp

  steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.config \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context $extra_left_context \
    --extra-right-context $extra_right_context \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
    --frames-per-chunk 160 \
    --online-ivector-dir $ivector_root_dir/ivectors_${dset} \
    --skip-scoring true \
    $graph_dir data/${dset}_hires $decode_dir || touch $chain_dir/.error

  $scoring_script --cmd "$decode_cmd" \
    data/${dset}_hires $graph_dir $decode_dir || touch $chain_dir/.error
  ) &
done

wait
if [ -f $chain_dir/.error ]; then
  echo Decode failed
  exit 1
fi
