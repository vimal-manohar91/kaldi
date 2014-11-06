#!/bin/bash

. cmd.sh

# This runs on just the 100k data; the triphone baseline, tri4a, is
# also trained on that subset.

if [ ! -f exp/ubm5a/final.ubm ]; then
  steps/train_ubm.sh --cmd "$train_cmd" 700 data/train_100k data/lang \
    exp/tri4a exp/ubm5a || exit 1;
fi 

steps/train_sgmm2.sh --cmd "$train_cmd" \
  9000 30000 data/train_100k data/lang exp/tri4a \
  exp/ubm5a/final.ubm exp/sgmm2_5a || exit 1;

(
  utils/mkgraph.sh data/lang_test exp/sgmm2_5a exp/sgmm2_5a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
    exp/sgmm2_5a/graph data/dev exp/sgmm2_5a/decode_dev
)&

 # Now discriminatively train the SGMM system on 100k data.
steps/align_sgmm2.sh --nj 30 --cmd "$train_cmd" --transform-dir exp/tri4a \
  --use-graphs true --use-gselect true data/train_100k data/lang exp/sgmm2_5a exp/sgmm2_5a_ali_100k

  # Took the beam down to 10 to get acceptable decoding speed.
steps/make_denlats_sgmm2.sh --nj 30 --sub-split 30 --beam 9.0 --lattice-beam 6 --cmd "$decode_cmd" \
  --transform-dir exp/tri4a \
  data/train_100k data/lang exp/sgmm2_5a_ali_100k exp/sgmm2_5a_denlats_100k

steps/train_mmi_sgmm2.sh --cmd "$decode_cmd" --transform-dir exp/tri4a --boost 0.1 \
  data/train_100k data/lang exp/sgmm2_5a_ali_100k exp/sgmm2_5a_denlats_100k exp/sgmm2_5a_mmi_b0.1

for iter in 1 2 3 4; do
(
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" 
  --config conf/decode.config --iter $iter \
    exp/sgmm2_5a/graph data/dev exp/sgmm2_5a_mmi_b0.1/decode_dev_it${iter}
)&
done

