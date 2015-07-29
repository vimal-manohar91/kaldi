#!/bin/bash
# E.g.: > ./run-1-main.sh --tri5-only "true" conf/lang/101-cantonese-limitedLP.official.conf
# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.
# Base period (BP) languages: Pashto, Turkish, Tagalog, Cantonese, Vietnamese
# Optional Period One (OP1) languages: Haitian, Lao, Zulu, Assamese, Bengali, Tamil

tri5_only=false
sgmm5_only=false
data_only=false
sil="sil"          # how do you want to label the silence phones?
sys_type="phone"   # can be "phone" (eval on PER) or "word" (eval on WER)
use_mfcc="true"
remove_tags="true" # remove tags _[0-9], ", % from phones?

. path.sh
. utils/parse_options.sh

# bengali: "conf/lang/103-bengali-limitedLP.official.conf"
# assamese: "conf/lang/102-assamese-limitedLP.official.conf"
# cantonese: "conf/lang/101-cantonese-limitedLP.official.conf"
# pashto: "conf/lang/104-pashto-limitedLP.official.conf"
# tagalog: "conf/lang/106-tagalog-limitedLP.official.conf"
# turkish: "conf/lang/105-turkish-limitedLP.official.conf"
# vietnamese: "conf/lang/107-vietnamese-limitedLP.official.conf"
# haitian: "conf/lang/201-haitian-limitedLP.official.conf"
# lao: "conf/lang/203-lao-limitedLP.official.conf"
# zulu: "conf/lang/206-zulu-limitedLP.official.conf"
# tamil: "conf/lang/204-tamil-limitedLP.official.conf"
L=$1

case "$L" in
		BNG)
			langconf=conf/lang/103-bengali-limitedLP.official.conf
			;;
		ASM)			
			langconf=conf/lang/102-assamese-limitedLP.official.conf
			;;
		CNT)
			langconf=conf/lang/101-cantonese-limitedLP.official.conf
			;;
		PSH)
			langconf=conf/lang/104-pashto-limitedLP.official.conf
			;;
		TGL)
			langconf=conf/lang/106-tagalog-limitedLP.official.conf
			;;
		TUR)
			langconf=conf/lang/105-turkish-limitedLP.official.conf	
			;;
		VTN)
			langconf=conf/lang/107-vietnamese-limitedLP.official.conf
			;;
		HAI)
			langconf=conf/lang/201-haitian-limitedLP.official.conf
			;;
		LAO)
			langconf=conf/lang/203-lao-limitedLP.official.conf
			;;
		ZUL)
			langconf=conf/lang/206-zulu-limitedLP.official.conf	
			;;
		TAM)
			langconf=conf/lang/204-tamil-limitedLP.official.conf	
			;;
		*)
			echo "Unknown language code $L." && exit 1
esac

mkdir -p langconf/$L
rm -rf langconf/$L/*
cp $langconf langconf/$L/lang.conf
langconf=langconf/$L/lang.conf

[ ! -f $langconf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

[[ -f path.sh ]] && . ./path.sh

. conf/common_vars.sh || exit 1;
. $langconf || exit 1;

[ -f local.conf ] && . ./local.conf

[[ $sys_type == "phone" ]] && \
{ convert_word_to_phone="true"; oovSymbol="<oov>"; } || \
{ convert_word_to_phone="false"
  # here we retain the $oovSymbol defined in lang.conf file 
}

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will 
                 #return non-zero return code
#set -u           #Fail on an undefined variable
echo using "Language = $L, config = $langconf"

#Preparing dev2h and train directories
if [ ! -f data/$L/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./data/$L/raw_train_data
    train_data_dir=`readlink -f ./data/$L/raw_train_data`
    touch data/$L/raw_train_data/.done
fi
nj_max=`cat $train_data_list | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`readlink -f ./data/$L/raw_train_data`
echo "train_data_dir = $train_data_dir"

if [ ! -d data/$L/raw_dev2h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV2H set"
  echo ---------------------------------------------------------------------  
  local/make_corpus_subset.sh "$dev2h_data_dir" "$dev2h_data_list" ./data/$L/raw_dev2h_data || exit 1
fi

if [ ! -d data/$L/raw_dev10h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV10H set"
  echo ---------------------------------------------------------------------  
  local/make_corpus_subset.sh "$dev10h_data_dir" "$dev10h_data_list" ./data/$L/raw_dev10h_data || exit 1
fi

nj_max=`cat $dev2h_data_list | wc -l`
if [[ "$nj_max" -lt "$decode_nj" ]] ; then
  echo "The maximum reasonable number of jobs is $nj_max -- you have $decode_nj! (The training and decoding process has file-granularity)"
  exit 1
  decode_nj=$nj_max
fi

mkdir -p data/$L/local
if [[ ! -f data/$L/local/lexicon.txt || data/$L/local/lexicon.txt -ot "$lexicon_file" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/$L/local on" `date`
  echo ---------------------------------------------------------------------  
  local/make_lexicon_subset.sh $train_data_dir/transcription $lexicon_file data/$L/local/filtered_lexicon.txt
  [[ $remove_tags == "true" ]] && sed -E -i 's/_[0-9]|"|%//g' data/$L/local/filtered_lexicon.txt
  phoneme_mapping=$(cat conf/sampa2ipa.txt|sed '/^;/d'|awk '{print $1, " = ", $2, ";"}' |tr '\n' ' ')
  phoneme_mapping=$(echo $phoneme_mapping; echo $phoneme_mapping_overrides)
  local/prepare_lexicon.pl  --phonemap "$phoneme_mapping" --sil "$sil" \
    $lexiconFlags data/$L/local/filtered_lexicon.txt data/$L/local
fi

if [[ ! -f data/$L/train/wav.scp || data/$L/train/wav.scp -ot "$train_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic training lists in data/$L/train on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/$L/train
  # What are fragments? Frags include
  # a) Mispronunciations. e.g. "representive" (mispron. word in audio) -> *representative* (word transcribed with the right spelling in text)
  # b) Stumbling speech. e.g.  "to- tomorrow" (speaker stumbles midway in the word tomorrow) -> to- tomorrow (word transcribed up to the cut off point and hyphenated)
  # c) Truncated words at the start or end of a recording. e.g. "tisfactory" (truncated word in audio) -> ~satisfactory (word transcribed w/o truncation but marked with a ~ to denote truncation)
  local/prepare_acoustic_training_data.pl \
    --vocab data/$L/local/lexicon.txt --convert-word-to-phone  $convert_word_to_phone --fragmentMarkers \-\*\~ \
    $train_data_dir data/$L/train > data/$L/train/skipped_utts.log  
fi

if [[ ! -f data/$L/dev2h/wav.scp || data/$L/dev2h/wav.scp -ot ./data/$L/raw_dev2h_data/audio ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h data lists in data/$L/dev2h on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/$L/dev2h
  local/prepare_acoustic_training_data.pl \
    --vocab data/$L/local/lexicon.txt  --convert-word-to-phone $convert_word_to_phone  --fragmentMarkers \-\*\~ \
    `pwd`/data/$L/raw_dev2h_data data/$L/dev2h > data/$L/dev2h/skipped_utts.log || exit 1
fi

if [[ $convert_word_to_phone == "true" ]]; then
	cp data/$L/local/lexicon.txt data/$L/local/lexicon_words.txt    
	perl utils/extract_phones_from_lexicon.pl data/$L/local/lexicon_words.txt > data/$L/local/lexicon.txt
	#sed -i "s/\<oov\>/$oovSymbol/" data/local/lexicon.txt
fi

if [[ ! -f data/$L/dev2h/glm || data/$L/dev2h/glm -ot "$glmFile" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing dev2h stm files in data/$L/dev2h on" `date`
  echo ---------------------------------------------------------------------
  if [ -z $dev2h_stm_file ]; then 
    echo "WARNING: You should define the variable stm_file pointing to the IndusDB stm"
    echo "WARNING: Doing that, it will give you scoring close to the NIST scoring.    "
    local/prepare_stm.pl --fragmentMarkers \-\*\~ data/$L/dev2h || exit 1
  else
    local/augment_original_stm.pl $dev2h_stm_file data/$L/dev2h || exit 1
  fi
  [ ! -z $glmFile ] && cp $glmFile data/$L/dev2h/glm

fi

mkdir -p data/$L/lang
rm -rf data/$L/lang/*
if [[ ! -f data/$L/lang/L.fst || data/$L/lang/L.fst -ot data/$L/local/lexicon.txt ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in data/$L/lang on" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh \
    --share-silence-phones true --position_dependent_phones false \
    data/$L/local $oovSymbol data/$L/local/tmp.lang data/$L/lang
fi

# We will simply override the default G.fst by the G.fst generated using SRILM
if [[ ! -f data/$L/srilm/lm.gz || data/$L/srilm/lm.gz -ot data/$L/train/text ]]; then
  echo ---------------------------------------------------------------------
  echo "Training SRILM language models in data/$L/srilm on" `date`
  echo ---------------------------------------------------------------------
  local/train_lms_srilm.sh --sys-type $sys_type --dev-text data/$L/dev2h/text \
    --train-text data/$L/train/text data/$L data/$L/srilm 
fi

if [[ ! -f data/$L/lang/G.fst || data/$L/lang/G.fst -ot data/$L/srilm/lm.gz ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating G.fst data/$L/lang/G.fst on " `date`
  echo ---------------------------------------------------------------------
  local/arpa2G.sh data/$L/srilm/lm.gz data/$L/lang data/$L/lang
fi
decode_nj=$dev2h_nj


echo ---------------------------------------------------------------------
echo "Starting plp feature extraction for data/$L/train in plp on" `date`
echo ---------------------------------------------------------------------
#if [ ! -f data/train/.plp.done ]; then
#if $use_pitch; then
   #steps/make_plp_pitch.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp_pitch/train plp
#else
    #steps/make_plp.sh --cmd "$train_cmd" --nj $train_nj data/train exp/make_plp/train plp
#fi
  #utils/fix_data_dir.sh data/train
  #steps/compute_cmvn_stats.sh data/train exp/make_plp/train plp
  #utils/fix_data_dir.sh data/train
  #touch data/train/.plp.done
#fi

if [[ ! -f data/$L/train/.mfcc.done ]]; then
  (
	steps/make_mfcc.sh --nj $train_nj --cmd "$train_cmd" data/$L/train exp/$L/make_mfcc/train mfcc/$L
	utils/fix_data_dir.sh data/$L/train
	steps/compute_cmvn_stats.sh data/$L/train exp/$L/make_mfcc/train mfcc/$L
	utils/fix_data_dir.sh data/$L/train
	touch data/$L/train/.mfcc.done
  ) &
fi    
wait;

mkdir -p exp/$L

if [ ! -f data/$L/train_sub3/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting monophone training data in data/$L/train_sub[123] on" `date`
  echo ---------------------------------------------------------------------
  numutt=`cat data/$L/train/feats.scp | wc -l`;
  utils/subset_data_dir.sh data/$L/train  5000 data/$L/train_sub1
  if [ $numutt -gt 10000 ] ; then
    utils/subset_data_dir.sh data/$L/train 10000 data/$L/train_sub2
  else
    (cd data/$L; ln -s train train_sub2 )
  fi
  if [ $numutt -gt 20000 ] ; then
    utils/subset_data_dir.sh data/$L/train 20000 data/$L/train_sub3
  else
    (cd data/$L; ln -s train train_sub3 )
  fi

  touch data/$L/train_sub3/.done
fi

if $data_only; then
  echo "--data-only is true" && exit 0
fi

if [ ! -f exp/$L/mono/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) monophone training in exp/$L/mono on" `date`
  echo ---------------------------------------------------------------------
  steps/train_mono.sh \
    --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
    data/$L/train_sub1 data/$L/lang exp/$L/mono
  touch exp/$L/mono/.done
fi

if [ ! -f exp/$L/tri1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting (small) triphone training in exp/$L/tri1 on" `date`
  echo ---------------------------------------------------------------------
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
    data/$L/train_sub2 data/$L/lang exp/$L/mono exp/$L/mono_ali_sub2
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
    data/$L/train_sub2 data/$L/lang exp/$L/mono_ali_sub2 exp/$L/tri1
  touch exp/$L/tri1/.done
fi


echo ---------------------------------------------------------------------
echo "Starting (medium) triphone training in exp/$L/tri2 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/$L/tri2/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
    data/$L/train_sub3 data/$L/lang exp/$L/tri1 exp/$L/tri1_ali_sub3
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
    data/$L/train_sub3 data/$L/lang exp/$L/tri1_ali_sub3 exp/$L/tri2
  touch exp/$L/tri2/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (full) triphone training in exp/$L/tri3 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/$L/tri3/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/tri2 exp/$L/tri2_ali
  steps/train_deltas.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesTri3 $numGaussTri3 data/$L/train data/$L/lang exp/$L/tri2_ali exp/$L/tri3
  touch exp/$L/tri3/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (lda_mllt) triphone training in exp/$L/tri4 on" `date`
echo ---------------------------------------------------------------------
if [ ! -f exp/$L/tri4/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/tri3 exp/$L/tri3_ali
  steps/train_lda_mllt.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesMLLT $numGaussMLLT data/$L/train data/$L/lang exp/$L/tri3_ali exp/$L/tri4
  touch exp/$L/tri4/.done
fi

echo ---------------------------------------------------------------------
echo "Starting (SAT) triphone training in exp/$L/tri5 on" `date`
echo ---------------------------------------------------------------------

if [ ! -f exp/$L/tri5/.done ]; then
  steps/align_si.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/tri4 exp/$L/tri4_ali
  steps/train_sat.sh \
    --boost-silence $boost_sil --cmd "$train_cmd" \
    $numLeavesSAT $numGaussSAT data/$L/train data/$L/lang exp/$L/tri4_ali exp/$L/tri5
  touch exp/$L/tri5/.done
fi


################################################################################
# Ready to start SGMM training
################################################################################

if [ ! -f exp/$L/tri5_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/$L/tri5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_fmllr.sh \
    --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
    data/$L/train data/$L/lang exp/$L/tri5 exp/$L/tri5_ali
  touch exp/$L/tri5_ali/.done
fi

if $tri5_only ; then
  echo "Exiting after stage TRI5, as requested. "
  echo "Everything went fine. Done"
  exit 0;
fi

if [ ! -f exp/$L/ubm5/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/ubm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_ubm.sh \
    --cmd "$train_cmd" $numGaussUBM \
    data/$L/train data/$L/lang exp/$L/tri5_ali exp/$L/ubm5
  touch exp/$L/ubm5/.done
fi

if [ ! -f exp/$L/sgmm5/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/$L/sgmm5 on" `date`
  echo ---------------------------------------------------------------------
  steps/train_sgmm2.sh \
    --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
    data/$L/train data/$L/lang exp/$L/tri5_ali exp/$L/ubm5/final.ubm exp/$L/sgmm5
  #steps/train_sgmm2_group.sh \
  #  --cmd "$train_cmd" "${sgmm_group_extra_opts[@]-}" $numLeavesSGMM $numGaussSGMM \
  #  data/train data/lang exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5
  touch exp/$L/sgmm5/.done
fi

if $sgmm5_only ; then
  echo "Exiting after stage SGMM5, as requested. "
  echo "Everything went fine. Done"
  exit 0;
fi
################################################################################
# Ready to start discriminative SGMM training
################################################################################

if [ ! -f exp/$L/sgmm5_ali/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_ali on" `date`
  echo ---------------------------------------------------------------------
  steps/align_sgmm2.sh \
    --nj $train_nj --cmd "$train_cmd" --transform-dir exp/$L/tri5_ali \
    --use-graphs true --use-gselect true \
    data/$L/train data/$L/lang exp/$L/sgmm5 exp/$L/sgmm5_ali
  touch exp/$L/sgmm5_ali/.done
fi


if [ ! -f exp/$L/sgmm5_denlats/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_denlats on" `date`
  echo ---------------------------------------------------------------------
  steps/make_denlats_sgmm2.sh \
    --nj $train_nj --sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
    --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir exp/tri5_ali \
    data/$L/train data/$L/lang exp/$L/sgmm5_ali exp/$L/sgmm5_denlats
  touch exp/$L/sgmm5_denlats/.done
fi

if [ ! -f exp/$L/sgmm5_mmi_b0.1/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Starting exp/sgmm5_mmi_b0.1 on" `date`
  echo ---------------------------------------------------------------------
  steps/$L/train_mmi_sgmm2.sh \
    --cmd "$train_cmd" "${sgmm_mmi_extra_opts[@]}" \
    --drop-frames true --transform-dir exp/$L/tri5_ali --boost 0.1 \
    data/$L/train data/$L/lang exp/$L/sgmm5_ali exp/$L/sgmm5_denlats \
    exp/$L/sgmm5_mmi_b0.1
  touch exp/$L/sgmm5_mmi_b0.1/.done
fi

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
