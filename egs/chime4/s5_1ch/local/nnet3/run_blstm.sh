# local/chime4_calc_wers.sh exp/nnet3_isolated_1ch_track/tdnn isolated_1ch_track exp/tri3b_tr05_multi_noisy/graph_tgpr_5k/
# compute dt05 WER for each location
#
# -------------------
# best overall dt05 WER 18.27% (language model weight = 10)
# -------------------
# dt05_simu WER: 17.98% (Average), 15.58% (BUS), 22.79% (CAFE), 14.03% (PEDESTRIAN), 19.54% (STREET)
# -------------------
# dt05_real WER: 18.56% (Average), 23.51% (BUS), 17.80% (CAFE), 12.75% (PEDESTRIAN), 20.19% (STREET)
# -------------------

local/nnet3/run_lstm.sh --affix bidirectional \
                        --stage 9 \
                        --train-stage 64 \
                        --lstm-delay " [-1,1] [-2,2] [-3,3] " \
                        --label-delay 0 \
                        --cell-dim 512  \
                        --recurrent-projection-dim 128 \
                        --non-recurrent-projection-dim 128 \
                        --chunk-left-context 40 \
                        --chunk-right-context 40 \
                        isolated_1ch_track
