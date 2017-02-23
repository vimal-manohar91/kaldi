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
    --cmd "$cmd" ${data}_spkrid \
    exp/make_mfcc_spkrid_16k/${dset} $mfccdir
  utils/fix_data_dir.sh ${data}_spkrid
fi
data=${data}_spkrid

if $get_uniform_subsegments; then
  data_uniform_seg=$dir/${dset}_uniform_seg_window${window}_ovlp${overlap}

  if $do_change_point_detection; then
    data_uniform_seg=${data_uniform_seg}_cp
  fi

  if [ $stage -le 1 ]; then
    rm -r ${data_uniform_seg} || true
    mkdir -p ${data_uniform_seg}

    if $do_change_point_detection; then
      $cmd $dir/log/get_subsegments.log \
        segmentation-init-from-segments --frame-overlap=$frame_overlap $data/segments ark:- \| \
        segmentation-split-segments --max-segment-length=$num_frames --overlap-length=$num_frames_overlap ark:- ark:- \| \
        segmentation-cluster-adjacent-segments --verbose=3 ark:- "scp:$data/feats.scp" ark:- \| \
        segmentation-post-process --merge-adjacent-segments ark:- ark:- \| \
        segmentation-to-segments --frame-overlap=0.0 ark:- ark:/dev/null \
        ${data_uniform_seg}/sub_segments
    else
      $cmd $dir/log/get_subsegments.log \
        segmentation-init-from-segments --frame-overlap=$frame_overlap $data/segments ark:- \| \
        segmentation-split-segments --max-segment-length=$num_frames --overlap-length=$num_frames_overlap ark:- ark:- \| \
        segmentation-to-segments --frame-overlap=0.0 ark:- ark:/dev/null \
        ${data_uniform_seg}/sub_segments
    fi

    utils/data/subsegment_data_dir.sh ${data} ${data_uniform_seg}{/sub_segments,}

    steps/make_mfcc.sh --mfcc-config conf/mfcc_spkrid_16k.conf --nj $nj \
      --cmd "$cmd" ${data_uniform_seg} \
      exp/make_mfcc_spkrid_16k/`basename $data_uniform_seg` $mfccdir
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
    --nj $reco_nj --use-vad false \
    $extractor \
    ${data} $dir/ivectors_dense_spkrid_${dset}
fi

if [ $stage -le 3 ]; then
  steps/diarization/score_plda.sh --cmd "$cmd --mem 4G" \
    --nj $reco_nj --target-energy $target_energy \
    $pldadir $dir/ivectors_dense_spkrid_${dset} \
    $dir/ivectors_dense_spkrid_${dset}/plda_scores
fi

if [ -z "$threshold" ]; then
  if [ $stage -le 4 ]; then
    steps/diarization/compute_plda_calibration.sh \
      --cmd "$cmd --mem 4G" --num-points 100000 \
      --gmm-calibration-opts "$gmm_calibration_opts"\
      $dir/ivectors_dense_spkrid_${dset}/plda_scores \
      $dir/ivectors_dense_spkrid_${dset}/plda_scores
  fi

  threshold=`cat $dir/ivectors_dense_spkrid_${dset}/plda_scores/threshold.txt`
fi

if [ $stage -le 5 ]; then
  if $use_plda_clusterable; then
    steps/diarization/cluster.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $threshold \
      --compartment-size $compartment_size \
      $dir/ivectors_dense_spkrid_${dset}/plda_scores \
      $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold
  else
    steps/diarization/cluster_ivectors.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold `perl -e "print 1.0 / (1.0 + exp($threshold))"` \
      --use-plda-clusterable true \
      --compartment-size $compartment_size ${cluster_opts} \
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
  python steps/diarization/make_rttm.py --reco2file-and-channel $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/reco2file_and_channel \
    $dir/ivectors_dense_spkrid_${dset}/segments \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/labels | \
    rttmSmooth.pl -s 0 | rttmSort.pl > \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/rttm  || exit 1
fi

if [ $stage -le 7 ]; then
  steps/diarization/convert_labels_to_data.sh \
    $data $dir/ivectors_dense_spkrid_${dset} \
    $dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold \
    $out_data
fi

exit 0

  rm -r $out_data || true
  mkdir -p $out_data

  for f in wav.scp reco2file_and_channel glm stm; do
    cp $data/$f $out_data
  done

  utils/data/get_reco2utt.sh $dir/ivectors_dense_spkrid_${dset}

  $cmd $dir/log/get_segments.log \
    segmentation-init-from-segments --shift-to-zero=false --frame-overlap=0.0 \
    --utt2label-rspecifier=ark,t:$dir/ivectors_dense_spkrid_${dset}/clusters_plda${plda_suffix}_th$threshold/labels \
    $dir/ivectors_dense_spkrid_${dset}/segments ark:- \| \
    segmentation-combine-segments-to-recordings ark:- \
    ark,t:$dir/ivectors_dense_spkrid_${dset}/reco2utt \
    ark:- \| \
    segmentation-post-process --merge-adjacent-segments ark:- ark:- \| \
    segmentation-split-segments \
    --max-segment-length=$max_segment_length \
    --overlap-length=$overlap_length ark:- ark:- \| \
    segmentation-to-segments --frame-overlap=0.0 ark:- \
    ark,t:$out_data/utt2spk $out_data/segments

  utils/utt2spk_to_spk2utt.pl $out_data/utt2spk > $out_data/spk2utt
  utils/fix_data_dir.sh $out_data
fi
