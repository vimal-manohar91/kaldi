#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

set -e -o pipefail

mfccdir=mfcc_spkrid_16k
nj=10
reco_nj=1
cmd=queue.pl
stage=-1

# Uniform segmentation options
frame_overlap=0.015
frame_shift=0.01
window=1.5
overlap=0.75
get_uniform_subsegments=true
do_change_point_detection=false
cp_suffix=_cp
change_point_split_opts="--use-full-covar --distance-metric=glr"
change_point_merge_opts="--use-full-covar --distance-metric=bic --bic-penalty=5.0"

# Clustering options
target_energy=0.5
threshold=
compartment_size=0
gmm_calibration_opts=
use_plda_clusterable=false
cluster_opts=

# Final segmentation options
max_segment_length=1000
overlap_length=100

plda_suffix=

. path.sh

. utils/parse_options.sh

if [ $# -ne 5 ]; then
  echo "Usage: $0 <data> <extractor> <pldadir> <dir> <out-data>"
  echo " e.g.: $0 data/eval97.seg_lstm_sad_music_1e exp/extractor_train_bn96_c1024_i128 exp/ivectors_spkrid_train_bn96 exp/diarization/diarization_eval97.seg_lstm_sad_music_1e{,/eval97.seg_lstm_sad_music_1e_diarized}"
  exit 1
fi

data=$1
extractor=$2
pldadir=$3
dir=$4
out_data=$5

dset=`basename $data`

num_frames=`perl -e "print int($window / $frame_shift + 0.5)"`
num_frames_overlap=`perl -e "print int($overlap/ $frame_shift + 0.5)"`

if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh $data ${data}_spkrid
  steps/make_mfcc.sh --mfcc-config conf/mfcc_spkrid_16k.conf --nj $nj \
    --cmd "$cmd" ${data}_spkrid
  steps/compute_cmvn_stats.sh ${data}_spkrid
  utils/fix_data_dir.sh ${data}_spkrid
fi
data=${data}_spkrid

if [ $stage -le 2 ]; then
  steps/diarization/extract_ivectors_nondense.sh --cmd "$cmd --mem 20G" \
    --nj $reco_nj --use-vad false --per-spk true \
    $extractor \
    ${data} $dir/ivectors_dense_spkrid_${dset}
fi

if [ $stage -le 3 ]; then
  steps/diarization/score_plda.sh --cmd "$cmd --mem 4G" \
    --nj $reco_nj --target-energy $target_energy --per-spk true \
    $pldadir $dir/ivectors_dense_spkrid_${dset} \
    $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix}
fi

if [ -z "$threshold" ]; then
  if [ $stage -le 4 ]; then
    steps/diarization/compute_plda_calibration.sh \
      --cmd "$cmd --mem 4G" --num-points 100000 \
      --gmm-calibration-opts "$gmm_calibration_opts"\
      $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix} \
      $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix}
  fi

  threshold=`cat $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix}/threshold.txt`
fi

if [ $stage -le 5 ]; then
  if ! $use_plda_clusterable; then
    steps/diarization/cluster.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $threshold --per-spk true \
      --compartment-size $compartment_size --cluster-opts "${cluster_opts}" \
      $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix} \
      $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold
  else
    steps/diarization/cluster_ivectors.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $threshold --per-spk true \
      --use-plda-clusterable true --target-energy $target_energy \
      --compartment-size $compartment_size --cluster-opts "${cluster_opts}" \
      $pldadir \
      $dir/ivectors_dense_spkrid_${dset} \
      $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold
  fi
fi

cat $data/reco2file_and_channel | \
  perl -ane 'if ($F[2] == "A") { $F[2] = "1"; } print(join(" ", @F) . "\n");' > \
  $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/reco2file_and_channel

export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH
if [ $stage -le 6 ]; then
  sort -k2,2 -k3,4n $dir/ivectors_dense_spkrid_${dset}/segments | \
    python steps/diarization/make_rttm.py --reco2file-and-channel $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/reco2file_and_channel \
    - $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/labels | \
    rttmSmooth.pl -s 0 | rttmSort.pl > \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/rttm  || exit 1
fi

if [ $stage -le 7 ]; then
  steps/diarization/convert_labels_to_data.sh \
    $data $dir/ivectors_dense_spkrid_${dset} \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/${dset}

  utils/copy_data_dir.sh \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/${dset} \
    $out_data 
fi

exit 0

