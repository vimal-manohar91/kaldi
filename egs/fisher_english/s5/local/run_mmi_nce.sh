. path.sh
. cmd.sh

num_iters=4
tau=1600
weight_tau=10
alpha=10.0
dir=exp/tri5a_mmi_nce_100k_250k
do_decode=false
stage=-8

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

dir=${dir}_t${tau}_wt${weight_tau}_a${alpha}

if [ ! -f $dir/.done ]; then 
  steps/train_mmi_nce.sh --cmd "$train_cmd" --num-iters $num_iters \
    --tau $tau --weight-tau $weight_tau --alpha $alpha --stage $stage \
    --transform-dir-unsup exp/tri4a/decode_100k_unsup_100k_250k \
    data/train_100k data/unsup_100k_250k data/lang_100k_test exp/tri4a \
    exp/tri4a_denlats_100k exp/tri4a/decode_100k_unsup_100k_250k $dir || exit 1
  touch $dir/.done
fi

decode=$dir/decode_100k_dev_it${num_iters}
if $do_decode  && [ ! -f $decode/.done ]; then
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    --transform-dir exp/tri4a/decode_100k_dev \
    exp/tri4a/graph_100k data/dev $decode || exit 1
  touch $decode/.done
fi

