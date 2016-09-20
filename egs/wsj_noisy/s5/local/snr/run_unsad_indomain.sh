###############################################################################
## Prepare indomain VAD training data by decoding the data using 
## a GMM model without speaker adaptation
###############################################################################

dir=../../babel/403-dholuo-llp
orig_train_data=$dir/data/train 
corpus_id=babel_dholuo_train

local/snr/prepare_unsad_unsup_data.sh \
  --feat-config $dir/conf/plp.conf --feat-type plp \
  --pitch-config $dir/conf/pitch.conf --add-pitch true \
  --map-unknown-to-speech true --map-noise-to-sil false \
  --speed-perturb true \
  $orig_train_data $dir/lang $dir/exp/tri4 \
  data/${corpus_id}_unsad_unsup \
  exp/unsad_unsup_data_prep_${corpus_id}_sp

train_data_dir=data/${corpus_id}_unsad_unsup_sp_hires
utils/copy_data_dir.sh data/${corpus_id}_unsad_unsup_sp $train_data_dir
steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_bp.conf \
  --cmd "$train_cmd" --nj 32 \
  $train_data_dir exp/make_hires/${corpus_id}_unsad_unsup_sp mfcc_hires
steps/compute_cmvn_stats.sh \
  $train_data_dir exp/make_hires/${corpus_id}_unsad_unsup_sp mfcc_hires

###############################################################################
## Train SAD network
###############################################################################
  
unsad_dir=exp/unsad_unsup_data_prep_${corpus_id}_sp
local/snr/run_train_sad_indomain.sh \
  --num-epochs 3 \
  --train-data-dir $train_data_dir \
  --train-data-id ${corpus_id}_unsad_unsup_sp \
  --vad-scp $unsad_dir/reco_vad/vad.scp

###############################################################################
## Create segments using the SAD network
###############################################################################

nnet_dir=exp/nnet3_unsad/nnet_indomain_${corpus_id}_unsad_unsup_sp 
orig_test_data_dir=data/dev
testid=dev

local/snr/run_test_indomain.sh \
  --reco-nj 10 \
  --segmentation-config conf/segmentation.conf \
  --weights-segmentation-config conf/weights_segmentation.conf \
  --use-gpu true --do-downsampling true \
  --mfcc-config conf/mfcc_hires_bp.conf \
  --sad-nnet-iter final \
  $orig_test_data_dir data/${testid} $nnet_dir
