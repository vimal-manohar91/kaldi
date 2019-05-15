#!/bin/bash
# Copyright (c) 2019, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
. ./path.sh || die "path.sh expected";

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

train_text=data/train/text
fisher_text=data/train_fisher/text
mission_transcripts="data/local/afj/afj_transcripts.txt data/local/nasa/alsj_transcripts.txt data/local/nasa/spacelog_transcripts.txt"
extra_transcripts="data/local/nasa/apollo_html_reports.txt data/local/nasa/a11_html_reports.txt"
dev_text=data/dev/text
dir=data/local/pocolm_nasa_notrain
lang=data/lang
stage=-1

bypass_metaparam_optim_opt=
num_ngrams_large=5000000
num_ngrams_small=2500000

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

lm_dir=${dir}/data

mkdir -p $dir
. ./path.sh || exit 1; # for KALDI_ROOT
export PATH=$KALDI_ROOT/tools/pocolm/scripts:$PATH
( # First make sure the pocolm toolkit is installed.
 cd $KALDI_ROOT/tools || exit 1;
 if [ -d pocolm ]; then
   echo Not installing the pocolm toolkit since it is already there.
 else
   echo "$0: Please install the PocoLM toolkit with: "
   echo " cd ../../../tools; extras/install_pocolm.sh; cd -"
   exit 1;
 fi
) || exit 1;

num_dev_sentences=10000

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  oov=$(cat $lang/oov.int) || exit 1

  for f in $mission_transcripts; do 
    cat $f | \
      utils/sym2int.pl --map-oov $oov $lang/words.txt | \
      utils/int2sym.pl $lang/words.txt
    done | shuf > ${dir}/data/mission_all.txt || exit
   
  set +o pipefail
  cat ${dir}/data/mission_all.txt | \
    head -n $num_dev_sentences > \
    ${dir}/data/text/dev.txt || exit 1
  
  cat ${dir}/data/mission_all.txt | \
    tail -n +$[num_dev_sentences+1] > \
    ${dir}/data/text/train.txt || exit 1
  set -o pipefail
  
  for f in $extra_transcripts; do
    cat $f | \
      utils/sym2int.pl --map-oov $oov $lang/words.txt | \
      utils/int2sym.pl $lang/words.txt | \
      gzip -c > ${dir}/data/text/$(basename $f).gz || exit 1;
  done

  cut -d ' ' -f 2- $fisher_text | \
    utils/sym2int.pl --map-oov $oov $lang/words.txt | \
    utils/int2sym.pl $lang/words.txt | \
    gzip -c > ${dir}/data/text/fisher.txt.gz || exit 1;

  # for reporting perplexities, we'll use the "real" dev set.
  # (a subset of the training data is used as ${dir}/data/text/dev.txt to work
  # out interpolation weights.
  # note, we can't put it in ${dir}/data/text/, because then pocolm would use
  # it as one of the data sources.
  cat $dev_text | cut -d " " -f 2- > ${dir}/data/real_dev_set.txt
  
  awk '{print $1}' $lang/words.txt | \
    grep -v '<s>' | grep -v '</s>' | \
    grep -v '<eps>' | grep -v '#0' | awk '
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s\n", $1);
  }' > $dir/data/wordlist || exit 1;
fi

order=4
wordlist=${dir}/data/wordlist

lm_name="`basename ${wordlist}`_${order}"
min_counts='fisher=2,3 default=1'
if [ -n "${min_counts}" ]; then
  lm_name+="_`echo ${min_counts} | tr -s "[:blank:]" "_" | tr "=" "-"`"
fi

unpruned_lm_dir=${lm_dir}/${lm_name}.pocolm

if [ $stage -le 1 ]; then
  # decide on the vocabulary.
  # Note: you'd use --wordlist if you had a previously determined word-list
  # that you wanted to use.
  # Note: if you have more than one order, use a certain amount of words as the
  # vocab and want to restrict max memory for 'sort',
  echo "$0: training the unpruned LM"
  train_lm.py  --wordlist=${wordlist} --num-splits=10 --warm-start-ratio=20  \
               --limit-unk-history=true \
               --fold-dev-into=train ${bypass_metaparam_optim_opt} \
               ${min_counts:+--min-counts="${min_counts}"} \
               ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir}

  get_data_prob.py ${dir}/data/real_dev_set.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity' | tee ${unpruned_lm_dir}/perplexity_real_dev_set.log
fi

if [ $stage -le 2 ]; then
  echo "$0: pruning the LM (to larger size)"
  # Using 5 million n-grams for a big LM for rescoring purposes.
  mkdir -p ${dir}/data/lm_${order}_prune_big
  prune_lm_dir.py --target-num-ngrams=$num_ngrams_large --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big \
    2> >(tee -a ${dir}/data/lm_${order}_prune_big/prune_lm.log >&2) || true

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_big/perplexity_real_dev_set.log

  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 3 ]; then
  echo "$0: pruning the LM (to smaller size)"
  mkdir -p ${dir}/data/lm_${order}_prune_small
  # Using 2.5 million n-grams for a smaller LM for graph building.  Prune from the
  # bigger-pruned LM, it'll be faster.
  prune_lm_dir.py --target-num-ngrams=$num_ngrams_small ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small \
    2> >(tee -a ${dir}/data/lm_${order}_prune_small/prune_lm.log >&2) || true

  get_data_prob.py ${dir}/data/real_dev_set.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_small/perplexity_real_dev_set.log

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
