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
ivector_opts=

# Clustering options
target_energy=0.5
distance_threshold=
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
  if $do_change_point_detection; then
    utils/data/convert_data_dir_to_whole.sh $data ${data}_whole_lp
    steps/make_mfcc.sh --mfcc-config conf/mfcc_16k_lp.conf --nj $reco_nj \
      --cmd "$cmd" ${data}_whole_lp 
    steps/compute_cmvn_stats.sh ${data}_whole_lp 
    utils/fix_data_dir.sh ${data}_whole_lp
  
    utils/copy_data_dir.sh $data ${data}_lp
    steps/make_mfcc.sh --mfcc-config conf/mfcc_16k_lp.conf --nj $nj \
      --cmd "$cmd" ${data}_lp
    steps/compute_cmvn_stats.sh ${data}_lp 
    utils/fix_data_dir.sh ${data}_lp
  else
    utils/copy_data_dir.sh $data ${data}_spkrid
    steps/make_mfcc.sh --mfcc-config conf/mfcc_spkrid_16k.conf --nj $nj \
      --cmd "$cmd" ${data}_spkrid
    steps/compute_cmvn_stats.sh ${data}_spkrid
    utils/fix_data_dir.sh ${data}_spkrid
  fi
fi


data_whole=${data}_whole_lp

if $do_change_point_detection; then
  data=${data}_lp
else
  data=${data}_spkrid
fi

if $get_uniform_subsegments; then
  data_uniform_seg=$dir/${dset}_uniform_seg_window${window}_ovlp${overlap}

  if $do_change_point_detection; then
    data_uniform_seg=${data_uniform_seg}${cp_suffix}
  fi

  if [ $stage -le 1 ]; then
    rm -r ${data_uniform_seg} || true
    mkdir -p ${data_uniform_seg}

    if $do_change_point_detection; then
      utils/data/get_reco2utt.sh $data
      utils/split_data.sh --per-utt $data $nj
      $cmd JOB=1:$nj $dir/log/split_by_change_points.JOB.log \
        segmentation-init-from-segments --frame-overlap=$frame_overlap $data/split${nj}utt/JOB/segments ark:- \| \
        segmentation-split-by-change-points $change_point_split_opts ark:- scp:$data/split${nj}utt/JOB/feats.scp ark:$dir/temp_segmentation.JOB.ark
      
      $cmd $dir/log/get_subsegments.log \
        cat $dir/temp_segmentation.*.ark \| \
        segmentation-combine-segments ark:- \
        "ark:segmentation-init-from-segments --frame-overlap=$frame_overlap --shift-to-zero=false $data/segments ark:- |" \
        ark,t:$data/reco2utt ark:- \| \
        segmentation-cluster-adjacent-segments $change_point_merge_opts \
        ark:- scp:$data_whole/feats.scp ark:- \| \
        segmentation-post-process --merge-adjacent-segments ark:- ark:- \| \
        segmentation-to-segments --single-speaker --frame-overlap=0.0 ark:- ark:/dev/null \
        ${data_uniform_seg}/sub_segments
      
      utils/data/subsegment_data_dir.sh ${data_whole} ${data_uniform_seg}{/sub_segments,}
    else
      $cmd $dir/log/get_subsegments.log \
        segmentation-init-from-segments --frame-overlap=$frame_overlap $data/segments ark:- \| \
        segmentation-split-segments --max-segment-length=$num_frames --overlap-length=$num_frames_overlap ark:- ark:- \| \
        segmentation-to-segments --frame-overlap=0.0 --single-speaker ark:- ark:/dev/null \
        ${data_uniform_seg}/sub_segments

      utils/data/subsegment_data_dir.sh ${data} ${data_uniform_seg}{/sub_segments,}
    fi

    steps/make_mfcc.sh --mfcc-config conf/mfcc_spkrid_16k.conf --nj $nj \
      --cmd "$cmd" ${data_uniform_seg} \
      exp/make_mfcc_spkrid_16k/`basename $data_uniform_seg` $mfccdir
    steps/compute_cmvn_stats.sh ${data_uniform_seg}
    utils/fix_data_dir.sh ${data_uniform_seg}
  fi
  
  for f in reco2file_and_channel glm stm; do
    cp $data/$f $data_uniform_seg
  done

  dset=`basename $data_uniform_seg`
  data=${data_uniform_seg}
fi

if [ $stage -le 2 ]; then
  steps/diarization/extract_ivectors_nondense.sh --cmd "$cmd --mem 20G" \
    --nj $reco_nj --use-vad false $ivector_opts \
    $extractor \
    ${data} $dir/ivectors_dense_spkrid_${dset}
fi

if [ $stage -le 3 ]; then
  steps/diarization/score_plda.sh --cmd "$cmd --mem 4G" \
    --nj $reco_nj --target-energy $target_energy \
    $pldadir $dir/ivectors_dense_spkrid_${dset} \
    $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix}
fi

if [ -z "$distance_threshold" ]; then
  if [ $stage -le 4 ]; then
    steps/diarization/compute_plda_calibration.sh \
      --cmd "$cmd --mem 4G" --num-points 100000 \
      --gmm-calibration-opts "$gmm_calibration_opts"\
      $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix} \
      $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix}
  fi

  distance_threshold=`cat $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix}/threshold.txt | awk '{print -$1}'`
fi

if [ $stage -le 5 ]; then
  if ! $use_plda_clusterable; then
    steps/diarization/cluster.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $distance_threshold \
      --cluster-opts "${cluster_opts}" \
      $dir/ivectors_dense_spkrid_${dset}/plda_scores${plda_suffix} \
      $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold
  else
    steps/diarization/cluster_ivectors.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $distance_threshold \
      --use-plda-clusterable true --target-energy $target_energy \
      --cluster-opts "${cluster_opts}" \
      $pldadir \
      $dir/ivectors_dense_spkrid_${dset} \
      $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold
  fi
fi

cat $data/reco2file_and_channel | \
  perl -ane 'if ($F[2] == "A") { $F[2] = "1"; } print(join(" ", @F) . "\n");' > \
  $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold/reco2file_and_channel

export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH
if [ $stage -le 6 ]; then
  sort -k2,2 -k3,4n $dir/ivectors_dense_spkrid_${dset}/segments | \
    python steps/diarization/make_rttm.py --reco2file-and-channel $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold/reco2file_and_channel \
    - $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold/labels | \
    rttmSmooth.pl -s 0 | rttmSort.pl > \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold/rttm  || exit 1
fi

if [ $stage -le 7 ]; then
  steps/diarization/convert_labels_to_data.sh \
    $data $dir/ivectors_dense_spkrid_${dset} \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold/${dset}

  utils/copy_data_dir.sh \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$distance_threshold/${dset} \
    $out_data 
fi

exit 0
