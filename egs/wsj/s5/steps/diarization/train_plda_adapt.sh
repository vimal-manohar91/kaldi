#! /bin/bash

# Copyright 2017 Vimal Manohar
# Apache 2.0.

cmd=run.pl

. path.sh

set -e -o pipefail

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: $0 <src-ivecdir> <dest-ivecdir> <plda-out-dir>"
  echo " e.g.: $0 exp/ivectors_s0.3_spkrid_i400_sre_8k exp/diarization/diarization_o_cp_s0.3_retry_i400_s0.3_mc0_eval97.seg_lstm_sad_music_1e/ivectors_perspk_spkrid_eval97.seg_lstm_sad_music_1e_lp_cp_s0.3 exp/diarization/diarization_o_cp_s0.3_retry_i400_s0.3_mc0_eval97.seg_lstm_sad_music_1e/ivectors_perspk_spkrid_eval97.seg_lstm_sad_music_1e_lp_cp_s0.3/plda_src_tx"
  exit 1
fi

src_ivecdir=$1
dest_ivecdir=$2
plda_out_dir=$3

for f in $dest_ivecdir/transform.mat $dest_ivecdir/mean.vec $src_ivecdir/mean.vec; do
  [ ! -f $f ] && echo "$0: Could not find file $f" && exit 1
done

mkdir -p $plda_out_dir

[ ! -f $src_ivecdir/ivector.scp ] && echo "$0: Could not find file $src_ivecdir/ivector.scp" && exit 1

ivectors="ark:copy-vector scp:$src_ivecdir/ivector.scp ark:- |"
ivectors="$ivectors ivector-subtract-global-mean $src_ivecdir/mean.vec ark:- ark:- | transform-vec $dest_ivecdir/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |"

$cmd $plda_out_dir/log/train_plda.log \
  ivector-compute-plda ark,t:$src_ivecdir/spk2utt \
  "$ivectors" $plda_out_dir/plda

cp $dest_ivecdir/transform.mat $plda_out_dir
cp $dest_ivecdir/mean.vec $plda_out_dir
