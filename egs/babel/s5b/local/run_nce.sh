#!/bin/bash

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

num_iters=2
train_stage=-10
alpha=1.0
datadir=dev2h.pem
dir=exp/tri5_nce
tau=400
weight_tau=10

. utils/parse_options.sh
dir=${dir}_a${alpha}_t${tau}_wt${weight_tau}

if [ ! -f $dir/.done ]; then
  steps/train_nce.sh --stage $train_stage --cmd "$train_cmd" \
    --num-iters $num_iters \
    --tau $tau --weight-tau $weight_tau \
    data/unsup.uem data/lang exp/tri5/decode_unsup.uem \
    $dir || exit 1
  touch $dir/.done
fi

dataset_dir=data/$datadir
dataset_id=$datadir
dataset_type=${datadir%%.*}

eval my_nj=\$${dataset_type}_nj  #for shadow, this will be re-set when appropriate

decode=$dir/decode_${dataset_id}_iter${num_iters}
if [ ! -f ${decode}/.done ]; then
  mkdir -p $decode
  steps/decode.sh --skip-scoring true \
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}" \
    --transform-dir exp/tri5/decode_${dataset_id} \
    exp/tri5/graph ${dataset_dir} ${decode} |tee ${decode}/decode.log
  touch ${decode}/.done
fi
  
local/run_kws_stt_task.sh --cer $cer --max-states $max_states \
  --skip-scoring false --extra-kws false --wip $wip \
  --cmd "$decode_cmd" --skip-kws true --skip-stt false \
  "${lmwt_plp_extra_opts[@]}" \
  ${dataset_dir} data/lang ${decode}

mkdir -p /home/vmanoha1/Results/LatEnt-Babel/
(
date;
grep "Sum" $decode/score_*/*.sys | grep $dir | utils/best_wer.sh 
echo `pwd`/$dir
) | tr '\n' ' ' >> /home/vmanoha1/Results/LatEnt-Babel/results

