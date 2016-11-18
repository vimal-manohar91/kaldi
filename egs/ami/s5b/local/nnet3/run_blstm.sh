#!/bin/bash

#    This is the standard "tdnn" system, built in nnet3.
# Please see RESULTS_* for examples of command lines invoking this script.


# local/nnet3/run_tdnn.sh --mic sdm1 --use-ihm-ali true

# local/nnet3/run_tdnn.sh --mic ihm --stage 11
# local/nnet3/run_tdnn.sh --mic ihm --train-set train --gmm tri3 --nnet3-affix "" &
#
# local/nnet3/run_tdnn.sh --mic sdm1 --stage 11 --affix _cleaned2 --gmm tri4a_cleaned2 --train-set train_cleaned2 &

# local/nnet3/run_tdnn.sh --use-ihm-ali true --mic sdm1 --train-set train --gmm tri3 --nnet3-affix "" &

# local/nnet3/run_tdnn.sh --use-ihm-ali true --mic mdm8 &

#  local/nnet3/run_tdnn.sh --use-ihm-ali true --mic mdm8 --train-set train --gmm tri3 --nnet3-affix "" &

# this is an example of how you'd train a non-IHM system with the IHM
# alignments.  the --gmm option in this case refers to the IHM gmm that's used
# to get the alignments.
# local/nnet3/run_tdnn.sh --mic sdm1 --use-ihm-ali true --affix _cleaned2 --gmm tri4a --train-set train_cleaned2 &



set -e -o pipefail -u

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
mic=ihm
nj=30
min_seg_len=1.55
use_ihm_ali=false
train_set=train_cleaned
gmm=tri3_cleaned  # this is the source gmm-dir for the data-type of interest; it
                  # should have alignments for the specified training data.
ihm_gmm=tri3      # Only relevant if $use_ihm_ali is true, the name of the gmm-dir in
                  # the ihm directory that is to be used for getting alignments.
num_threads_ubm=32
nnet3_affix=_cleaned  # cleanup affix for exp dirs, e.g. _cleaned
dnn_affix=bidirectional  #affix for TDNN directory e.g. "a" or "b", in case we change the configuration.

# BLSTM params
cell_dim=512
rp_dim=128
nrp_dim=128
chunk_left_context=40
chunk_right_context=40

# Options which are not passed through to run_ivector_common.sh
num_jobs_initial=2
num_jobs_final=12
samples_per_iter=20000
train_stage=-10

common_egs_dir=
remove_egs=true

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

local/nnet3/run_lstm.sh --min-seg-len $min_seg_len \
                         --train-set $train_set \
                         --gmm $gmm \
                         --ihm-gmm $ihm_gmm \
                         --nnet3-affix "$nnet3_affix" \
                         --dnn-affix "$dnn_affix" \
                         --stage $stage \
                         --train-stage $train_stage \
                         --lstm-delay " [-1,1] [-2,2] [-3,3] " \
                         --label-delay 0 \
                         --cell-dim $cell_dim \
                         --recurrent-projection-dim $rp_dim \
                         --non-recurrent-projection-dim $nrp_dim \
                         --common-egs-dir "$common_egs_dir" \
                         --chunk-left-context $chunk_left_context \
                         --chunk-right-context $chunk_right_context \
                         --mic $mic \
                         --num-jobs-initial $num_jobs_initial \
                         --num-jobs-final $num_jobs_final \
                         --samples-per-iter $samples_per_iter \
                         --use-ihm-ali $use_ihm_ali \
                         --remove-egs $remove_egs


exit 0;

