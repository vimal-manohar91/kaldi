#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

set -e -o pipefail

mfccdir=mfcc_spkrid_16k
mfcc_config_cp=conf/mfcc_16k_lp.conf
mfcc_config=conf/mfcc_spkrid_16k.conf

nj=10
reco_nj=1
cmd=queue.pl
stage=-1

# Uniform segmentation options
window=1.5              # Window for uniform segmentation
overlap=0.75            # Overlap between adjacent windows for overlap segmentation
get_uniform_subsegments=true      # Set to false if you want to use the data directory directly
get_whole_data_and_segment=false
sliding_cmvn_opts=

do_change_point_detection=false   # Do change point detection instead of creating uniform segmentation
cp_suffix=_cp
change_point_split_opts="--use-full-covar --distance-metric=glr"
change_point_merge_opts="--use-full-covar --distance-metric=bic --threshold=1.0"

# I-vector options
ivector_opts=            # Options for extracting i-vectors
per_spk=false            # Extract i-vector per-speaker instead of per-segment
use_vad=false

# PLDA options
use_src_mean=false       # Use mean from eval directory to mean normalize i-vectors
use_src_transform=false  # Use transform from eval directory to whiten i-vectors
plda_suffix=

calibration_method=kMeans
gmm_calibration_opts=    # Options for unsupervised calibration

# Clustering options
calibrate_per_reco=false
target_energy=0.5        # Energy retained by conversation-dependent PCA
distance_threshold=      # Threshold for AHC
cluster_method="plda-avg-scores"    # Method for AHC
cluster_opts=            # Options for clustering

# Final segmentation options
max_segment_length=1000
overlap_length=100

reco2num_spk=

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

frame_shift=`utils/data/get_frame_shift.sh $data` || frame_shift=0.01
num_frames=`perl -e "print int($window / $frame_shift + 0.5)"`
num_frames_overlap=`perl -e "print int($overlap/ $frame_shift + 0.5)"`

if [ $stage -le -1 ]; then
  if $do_change_point_detection; then
    utils/data/convert_data_dir_to_whole.sh $data ${data}_whole_lp
    steps/make_mfcc.sh --mfcc-config $mfcc_config_cp --nj $reco_nj \
      --cmd "$cmd" ${data}_whole_lp 
    steps/compute_cmvn_stats.sh ${data}_whole_lp 
    utils/fix_data_dir.sh ${data}_whole_lp
  
    utils/copy_data_dir.sh $data ${data}_lp
    steps/make_mfcc.sh --mfcc-config $mfcc_config_cp --nj $nj \
      --cmd "$cmd" ${data}_lp
    steps/compute_cmvn_stats.sh ${data}_lp 
    utils/fix_data_dir.sh ${data}_lp
  else
    utils/copy_data_dir.sh $data ${data}_spkrid
    steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $nj \
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
  if $do_change_point_detection; then
    data_uniform_seg=${data}${cp_suffix}
  else
    data_uniform_seg=$dir/${dset}_uniform_seg_window${window}_ovlp${overlap}
  fi

  mkdir -p $dir/change_point${cp_suffix}
  cp_dir=$dir/change_point${cp_suffix}

  if $do_change_point_detection; then
    if [ $stage -le 0 ]; then
      utils/data/get_reco2utt.sh $data
      utils/split_data.sh --per-utt $data $nj
      $cmd JOB=1:$nj $cp_dir/log/split_by_change_points${cp_suffix}.JOB.log \
        segmentation-init-from-segments --frame-overlap=0 $data/split${nj}utt/JOB/segments ark:- \| \
        segmentation-split-by-change-points $change_point_split_opts ark:- scp:$data/split${nj}utt/JOB/feats.scp ark:$cp_dir/temp_segmentation.JOB.ark
    fi

    if [ $stage -le 1 ]; then
      rm -r ${data_uniform_seg} || true
      mkdir -p ${data_uniform_seg}

      utils/data/get_reco2utt.sh $data
      $cmd $cp_dir/log/get_subsegments${cp_suffix}.log \
        cat $cp_dir/temp_segmentation.*.ark \| \
        segmentation-combine-segments ark:- \
        "ark:segmentation-init-from-segments --frame-overlap=0 --shift-to-zero=false $data/segments ark:- |" \
        ark,t:$data/reco2utt ark:- \| \
        segmentation-cluster-adjacent-segments --verbose=3 $change_point_merge_opts \
        ark:- scp:$data_whole/feats.scp ark:- \| \
        segmentation-post-process --merge-adjacent-segments ark:- ark:- \| \
        segmentation-to-segments --frame-overlap=0.0 ark:- ark,t:${data_uniform_seg}/utt2label \
        ${data_uniform_seg}/sub_segments
      
      utils/data/get_utt2dur.sh --nj $nj --cmd "$cmd" $data_whole 
      rm ${data_whole}/segments || true
      utils/data/get_segments_for_data.sh $data_whole >$data_whole/segments
      utils/data/subsegment_data_dir.sh ${data_whole} ${data_uniform_seg}/sub_segments $data_uniform_seg
      cp $data_uniform_seg/utt2label $data_uniform_seg/utt2spk
    fi
  else
    if [ $stage -le 0 ]; then
      rm -r ${data_uniform_seg} || true
      mkdir -p ${data_uniform_seg}

      $cmd $dir/log/get_subsegments.log \
        segmentation-init-from-segments --frame-overlap=0 $data/segments ark:- \| \
        segmentation-split-segments --max-segment-length=$num_frames --overlap-length=$num_frames_overlap ark:- ark:- \| \
        segmentation-to-segments --frame-overlap=0.0 --single-speaker ark:- ark:/dev/null \
        ${data_uniform_seg}/sub_segments

      utils/data/subsegment_data_dir.sh ${data} ${data_uniform_seg}{/sub_segments,}
      awk '{print $1" "$1}' $data_uniform_seg/segments > $data_uniform_seg/utt2spk
      cp $data_uniform_seg/utt2spk $data_uniform_seg/spk2utt
    fi
  fi
     
  if ! $get_whole_data_and_segment; then
    if [ $stage -le 1 ]; then
      utils/fix_data_dir.sh $data_uniform_seg
      steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $nj \
        --cmd "$cmd" ${data_uniform_seg} \
        exp/make_mfcc_spkrid_16k/`basename $data_uniform_seg` $mfccdir
      steps/compute_cmvn_stats.sh ${data_uniform_seg}
      utils/fix_data_dir.sh ${data_uniform_seg}
    fi
  fi

  for f in reco2file_and_channel glm stm; do
    [ -f $data/$f ] && cp $data/$f $data_uniform_seg
  done

  if [ $stage -le 2 ]; then
    steps/diarization/compute_cmvn_stats_perutt.sh --nj $nj --cmd "$cmd" \
      ${data_uniform_seg}
    steps/sid/compute_vad_decision.sh ${data_uniform_seg}
    steps/compute_cmvn_stats.sh ${data_uniform_seg}
    utils/fix_data_dir.sh ${data_uniform_seg}
  fi

  dset=`basename $data_uniform_seg`
  data=${data_uniform_seg}
fi

cmvn_affix=sliding_cmvn
if $get_whole_data_and_segment; then
  if [ $stage -le 2 ]; then
    utils/data/convert_data_dir_to_whole.sh $data ${data}_whole_spkrid
    steps/make_mfcc.sh --mfcc-config $mfcc_config --nj $reco_nj \
      --cmd "$cmd" ${data}_whole_spkrid exp/make_mfcc_spkrid_16k/${dset}_whole mfcc_spkrid_16k
    steps/compute_cmvn_stats.sh \
      ${data}_whole_spkrid exp/make_mfcc_spkrid_16k/${dset}_whole mfcc_spkrid_16k
    steps/diarization/compute_cmvn_stats_perutt.sh --nj $nj --cmd "$cmd" \
      ${data}_whole_spkrid exp/make_mfcc_spkrid_16k/${dset}_whole mfcc_spkrid_16k
    steps/sid/compute_vad_decision.sh ${data}_whole_spkrid
    utils/fix_data_dir.sh ${data}_whole_spkrid
  fi
  mfccdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $mfccdir ${PWD}`

  utils/data/get_reco2utt.sh $data
  utils/split_data.sh --per-reco $data $reco_nj

  sdata=$data/split${reco_nj}reco

  utils/copy_data_dir.sh $data ${data}_${cmvn_affix}

  if [ $stage -le 3 ]; then
    if [ -z "$sliding_cmvn_opts" ]; then
      echo "$0: Not provided --cmvn-opts for sliding_cmvn"
    fi

    $cmd JOB=1:$reco_nj exp/make_mfcc_spkrid_16k/${dset}_whole/apply_cmvn_sliding_${dset}.JOB.log \
      apply-cmvn-sliding ${sliding_cmvn_opts} "scp:utils/filter_scp.pl $sdata/JOB/reco2utt ${data}_whole_spkrid/feats.scp |" ark:- \| \
      extract-feature-segments ark:- $sdata/JOB/segments ark:- \| copy-feats ark:- \
      ark,scp:$mfccdir/${cmvn_affix}_mfcc_${dset}.JOB.ark,$mfccdir/${cmvn_affix}_mfcc_${dset}.JOB.scp
    
    for n in `seq $reco_nj`; do
      cat $mfccdir/${cmvn_affix}_mfcc_${dset}.$n.scp
    done | \
      sort -k1,1 > ${data}_${cmvn_affix}/feats.scp
  
    steps/compute_cmvn_stats.sh --fake ${data}_${cmvn_affix}
    utils/fix_data_dir.sh ${data}_${cmvn_affix}
  fi
  
  for f in reco2file_and_channel glm stm; do
    cp $data/$f ${data}_${cmvn_affix}
  done

  data=${data}_${cmvn_affix}
  dset=`basename $data`
fi

perspk_affix=
if $per_spk; then
  perspk_affix=_perspk
fi

ivectors_dir=$dir/ivectors${perspk_affix}_spkrid_${dset}
if $use_vad; then
  ivectors_dir=${ivectors_dir}_vad
fi

if [ $stage -le 4 ]; then
  steps/diarization/extract_ivectors_nondense.sh --cmd "$cmd --mem 20G" \
    --nj $nj --use-vad $use_vad $ivector_opts --per-spk $per_spk \
    $extractor ${data} $ivectors_dir
fi

if [ -f $pldadir/snn/transform_iter0.mat ]; then
  plda_suffix=${plda_suffix}_snn
elif [ -f $pldadir/efr/transform_iter0.mat ]; then
  plda_suffix=${plda_suffix}_efr
fi
if $use_src_mean; then
  plda_suffix=${plda_suffix}_src_mean

  if $use_src_transform; then
    plda_suffix=${plda_suffix}_tx
  fi
fi

plda_suffix=${plda_suffix}_e$target_energy
if [ $stage -le 5 ]; then
  steps/diarization/score_plda.sh --cmd "$cmd --mem 4G" \
    --nj $reco_nj --target-energy $target_energy --per-spk $per_spk \
    --use-src-mean $use_src_mean --use-src-transform $use_src_transform \
    $pldadir $ivectors_dir \
    $ivectors_dir/plda_scores${plda_suffix}
fi

pldadir=`cat $ivectors_dir/plda_scores${plda_suffix}/pldadir`
if [ -z "$distance_threshold" ]; then
  if [ $stage -le 6 ]; then
    steps/diarization/compute_plda_calibration.sh \
      --cmd "$cmd --mem 4G" --num-points 100000 --per-reco $calibrate_per_reco \
      --calibration-method "$calibration_method" \
      --gmm-calibration-opts "$gmm_calibration_opts" \
      $ivectors_dir/plda_scores${plda_suffix} \
      $ivectors_dir/plda_scores${plda_suffix}
  fi

  if $calibrate_per_reco; then
    cat $ivectors_dir/plda_scores${plda_suffix}/thresholds_per_reco.ark.txt | \
      awk '{if (NF > 0) {print $1" "(-$2)} }' > $ivectors_dir/plda_scores${plda_suffix}/distance_thresholds.ark.txt
    cluster_opts="$cluster_opts --thresholds-rspecifier=ark,t:$ivectors_dir/plda_scores${plda_suffix}/distance_thresholds.ark.txt"
  fi

  distance_threshold=`cat $ivectors_dir/plda_scores${plda_suffix}/threshold.txt | awk '{print -$1}'`
fi

cluster_affix=
cluster_suffix=_th$distance_threshold

if [ ! -z "$reco2num_spk" ]; then
  cluster_suffix=_num_spk
fi

if [ $cluster_method == "plda" ]; then
  cluster_affix=_plda_plda
  if [ $stage -le 7 ]; then
    steps/diarization/cluster_ivectors.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $distance_threshold --per-spk $per_spk \
      --use-plda-clusterable true --target-energy $target_energy \
      --cluster-opts "${cluster_opts}" --reco2num_spk "$reco2num_spk" \
      $pldadir $ivectors_dir \
      $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}
  fi
elif [ $cluster_method == "plda-avg-ivector" ]; then
  cluster_affix=_plda_avg_ivector
  if [ $stage -le 7 ]; then
    steps/diarization/cluster_ivectors.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $distance_threshold --per-spk $per_spk \
      --use-plda-clusterable true --target-energy $target_energy \
      --cluster-opts "${cluster_opts}" --reco2num_spk "$reco2num_spk" \
      $pldadir $ivectors_dir \
      $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}
  fi
elif [ $cluster_method == "plda-avg-scores" ]; then
  cluster_affix=_plda_avg_scores
  if [ $stage -le 7 ]; then
    steps/diarization/cluster.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $distance_threshold --per-spk $per_spk \
      --cluster-opts "${cluster_opts} --apply-sigmoid=false" --reco2num_spk "$reco2num_spk" \
      $ivectors_dir/plda_scores${plda_suffix} \
      $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}
  fi
elif [ $cluster_method == "plda-sigmoid" ]; then
  cluster_affix=_plda
  if [ $stage -le 7 ]; then
    steps/diarization/cluster.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $distance_threshold --per-spk $per_spk \
      --cluster-opts "${cluster_opts} --apply-sigmoid=true" --reco2num_spk "$reco2num_spk" \
      $ivectors_dir/plda_scores${plda_suffix} \
      $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}
  fi
elif [ $cluster_method == "kmeans" ]; then
  cluster_affix=_kmeans
  if [ $stage -le 7 ]; then
    steps/diarization/cluster_ivectors.sh --cmd "$cmd --mem 4G" \
      --nj $reco_nj --threshold $distance_threshold --per-spk $per_spk \
      --use-plda-clusterable false --target-energy $target_energy --reco2num_spk "$reco2num_spk" \
      --cluster-opts "${cluster_opts}" \
      $pldadir $ivectors_dir \
      $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}
  fi
else
  echo "$0: Unknown clustering method $cluster_method"
  exit 1
fi

cat $data/reco2file_and_channel | \
  perl -ane 'if ($F[2] == "A") { $F[2] = "1"; } print(join(" ", @F) . "\n");' > \
  $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/reco2file_and_channel

export PATH=$KALDI_ROOT/tools/sctk/bin:$PATH
if [ $stage -le 8 ]; then
  sort -k2,2 -k3,4n $ivectors_dir/segments | \
    python steps/diarization/make_rttm.py --reco2file-and-channel $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/reco2file_and_channel \
    - $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/labels | \
    rttmSmooth.pl -s 0 | rttmSort.pl > \
    $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/rttm  || exit 1
fi

if [ -f $data/rttm ]; then
  md-eval.pl -1 -c 0.25 -r $data/rttm \
    -s $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/rttm
fi

if [ $stage -le 9 ]; then
  steps/diarization/convert_labels_to_data.sh \
    $data $ivectors_dir \
    $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix} \
    $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/${dset}

  echo "$0: Created data directory in $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/${dset}"

  utils/copy_data_dir.sh \
    $ivectors_dir/clusters${cluster_affix}${plda_suffix}${cluster_suffix}/${dset} \
    $out_data 
fi

exit 0
