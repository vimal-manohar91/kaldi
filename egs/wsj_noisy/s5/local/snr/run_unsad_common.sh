
###############################################################################
## Prepare VAD training data for
## 10 hr   Babel assamese
## 10 hr   Babel zulu 
## 10 hr   Babel tamil 
## 10 hr   Babel cantonese
## 100 hrs Fisher english
###############################################################################

dir=/export/a14/vmanoha1/workspace_latent/babel_assamese
corpus_id=babel_assamese
local/snr/prepare_unsad_data.sh \
  --feat-config $dir/conf/plp.conf --feat-type plp \
  --pitch-config $dir/conf/pitch.conf --add-pitch true \
  --get-whole-recordings-and-weights true \
  --map-unknown-to-speech true --map-noise-to-sil false \
  --speed-perturb true \
  $dir/data/train $dir/data/lang $dir/exp/tri5_ali $dir/exp/tri4 \
  data/${corpus_id}_train_unsad_whole \
  exp/unsad_whole_data_prep_${corpus_id}_train_sp

dir=/export/a14/vmanoha1/workspace_latent/babel_tamil
corpus_id=babel_tamil
local/snr/prepare_unsad_data.sh \
  --feat-config $dir/conf/plp.conf --feat-type plp \
  --pitch-config $dir/conf/pitch.conf --add-pitch true \
  --get-whole-recordings-and-weights true \
  --map-unknown-to-speech true --map-noise-to-sil false \
  --speed-perturb true \
  $dir/data/train $dir/data/lang $dir/exp/tri5_ali $dir/exp/tri4 \
  data/${corpus_id}_train_unsad_whole \
  exp/unsad_whole_data_prep_${corpus_id}_train_sp

dir=/home/vmanoha1/workspace_snr/egs/babel/s5d_cantonese
corpus_id=babel_cantonese
local/snr/prepare_unsad_data.sh \
  --feat-config $dir/conf/plp.conf --feat-type plp \
  --pitch-config $dir/conf/pitch.conf --add-pitch true \
  --get-whole-recordings-and-weights true \
  --map-unknown-to-speech true --map-noise-to-sil false \
  --speed-perturb true \
  $dir/data/train $dir/data/lang $dir/exp/tri5_ali $dir/exp/tri4 \
  data/${corpus_id}_train_unsad_whole \
  exp/unsad_whole_data_prep_${corpus_id}_train_sp

dir=/export/a14/vmanoha1/workspace_uncertainty/babel_zulu
corpus_id=babel_zulu
local/snr/prepare_unsad_data.sh \
  --feat-config $dir/conf/plp.conf --feat-type plp \
  --pitch-config $dir/conf/pitch.conf --add-pitch true \
  --get-whole-recordings-and-weights true \
  --map-unknown-to-speech true --map-noise-to-sil false \
  --speed-perturb true \
  $dir/data/train $dir/data/lang $dir/exp/tri5_ali $dir/exp/tri4 \
  data/${corpus_id}_train_unsad_whole \
  exp/unsad_whole_data_prep_${corpus_id}_train_sp

dir=/export/a15/vmanoha1/workspace_snr/egs/aspire/s5
corpus_id=train_100k
local/snr/prepare_unsad_data.sh \
  --feat-config $dir/conf/mfcc.conf --feat-type mfcc \
  --get-whole-recordings-and-weights true \
  --map-unknown-to-speech true --map-noise-to-sil false \
  --speed-perturb true \
  $dir/data/train_100k $dir/data/lang_test \
  $dir/exp/tri4a_ali_100k \
  $dir/exp/tri3a \
  data/train_100k_unsad_whole \
  exp/unsad_whole_data_prep_train_100k_sp

###############################################################################
## Prepare corrupted data for 
## 10 hr   Babel assamese
## 10 hr   Babel zulu 
## 10 hr   Babel tamil 
## 10 hr   Babel cantonese
## 100 hrs Fisher english
###############################################################################

for corpus_id in babel_assamese babel_zulu babel_tamil babel_cantonese; do
  local/snr/run_corrupt.sh --data-only false --corrupt-only false --dry-run false \
    --data-dir data/${corpus_id}_train_unsad_whole \
    --speed-perturb true --num-data-reps 10 \
    --dest-wav-dir wavs_${corpus_id}_train_sp_unsad \
    --mfcc-config conf/mfcc_hires_bp.conf \
    --fbank-config conf/fbank_tiny.conf \
    --uncorrupted-vad-scp exp/unsad_whole_data_prep_${corpus_id}_sp/reco_vad/vad.scp 
done
  
corpus_id=train_100k
local/snr/run_corrupt.sh --data-only false --corrupt-only false --dry-run false \
  --data-dir data/${corpus_id}_unsad_whole \
  --speed-perturb true --num-data-reps 10 \
  --dest-wav-dir wavs_${corpus_id}_sp_unsad \
  --mfcc-config conf/mfcc_hires_bp.conf \
  --fbank-config conf/fbank_tiny.conf \
  --uncorrupted-vad-scp exp/unsad_whole_data_prep_${corpus_id}_sp/reco_vad/vad.scp 

###############################################################################
## Combine the prepared data and labels into single directory
###############################################################################

train_data_dir=data/train_azteec_unsad_whole_sp_multi_hires
utils/combine_data.sh --extra-files irm_targets.scp \
  data/train_azteec_unsad_whole_sp_multi_hires \
  data/babel_assamese_train_unsad_whole_sp_multi_hires \
  data/babel_zulu_train_unsad_whole_sp_multi_hires \
  data/babel_tamil_train_unsad_whole_sp_multi_hires \
  data/train_100k_unsad_whole_sp_multi_hires \
  data/babel_cantonese_train_unsad_whole_sp_multi_hires

cat exp/unsad_whole_data_prep_babel_{assamese,zulu,tamil,cantonese}_sp/reco_vad/vad_multi.scp \
  exp/unsad_whole_data_prep_train_100k_sp/reco_vad/vad_multi.scp > \
  exp/unsad_whole_data_prep_train_100k_sp/reco_vad/vad_azteec_multi.scp

cat exp/unsad_whole_data_prep_babel_{assamese,zulu,tamil,cantonese}_sp/final_vad/deriv_weights_multi_for_corrupted.scp \
  exp/unsad_whole_data_prep_train_100k_sp/final_vad/deriv_weights_multi_for_corrupted.scp > \
  exp/unsad_whole_data_prep_train_100k_sp/final_vad/deriv_weights_azteec_multi_for_corrupted.scp 

cat exp/unsad_whole_data_prep_babel_{assamese,zulu,tamil,cantonese}_sp/final_vad/deriv_weights_multi.scp \
  exp/unsad_whole_data_prep_train_100k_sp/final_vad/deriv_weights_multi.scp > \
  exp/unsad_whole_data_prep_train_100k_sp/final_vad/deriv_weights_azteec_multi.scp 

utils/subset_data_dir.sh \
  --utt-list <(cat $train_data_dir/utt2spk | grep "clean-corrupted1_") \
  $train_data_dir data/train_azteec_unsad_whole_sp_multi_lessreverb_hires

dataid=train_azteec_unsad_whole_sp_multi_lessreverb

utils/subset_data_dir.sh data/${dataid}_hires 1000 data/${dataid}_1k_hires
utils/subset_data_dir.sh data/${dataid}_fbank 1000 data/${dataid}_1k_fbank
