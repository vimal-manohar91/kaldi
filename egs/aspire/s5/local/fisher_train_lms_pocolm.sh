#!/bin/bash

# Copyright 2016  Vincent Nguyen
#           2016  Johns Hopkins University (author: Daniel Povey)
#           2017  Vimal Manohar
# Apache 2.0
#
# It is based on the example scripts distributed with PocoLM

set -e
stage=0

text=data/train_all/text
lexicon=data/local/dict/lexicon.txt
dir=data/local/pocolm

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

for f in "$text" "$lexicon"; do
  [ ! -f $x ] && echo "$0: No such file $f" && exit 1;
done

num_dev_sentences=10000

#bypass_metaparam_optim_opt=
# If you want to bypass the metaparameter optimization steps with specific metaparameters
# un-comment the following line, and change the numbers to some appropriate values.
# You can find the values from output log of train_lm.py.
# These example numbers of metaparameters is for 4-gram model (with min-counts)
# running with train_lm.py.
# The dev perplexity should be close to the non-bypassed model.
#bypass_metaparam_optim_opt="--bypass-metaparameter-optimization=0.854,0.0722,0.5808,0.338,0.166,0.015,0.999,0.6228,0.340,0.172,0.999,0.788,0.501,0.406"
# Note: to use these example parameters, you may need to remove the .done files
# to make sure the make_lm_dir.py be called and tain only 3-gram model
#for order in 3; do
#rm -f ${lm_dir}/${num_word}_${order}.pocolm/.done

if [ $stage -le 0 ]; then
  mkdir -p ${dir}/data
  mkdir -p ${dir}/data/text

  echo "$0: Getting the Data sources"

  rm ${dir}/data/text/* 2>/dev/null || true

  cleantext=$dir/text_all.gz

  cut -d ' ' -f 2- $text | awk -v lex=$lexicon '
  BEGIN{
    while((getline<lex) >0) { seen[$1]=1; }
  }
  {
    for(n=1; n<=NF;n++) {  
      if (seen[$n]) { 
        printf("%s ", $n); 
      } else {
        printf("<unk> ");
      } 
    }
    printf("\n");
  }' | gzip -c > $cleantext || exit 1;

  # This is for reporting perplexities
  gunzip -c $dir/text_all.gz | head -n $num_dev_sentences > \
    ${dir}/data/test.txt

  # use a subset of the annotated training data as the dev set .
  # Note: the name 'dev' is treated specially by pocolm, it automatically
  # becomes the dev set.
  gunzip -c $dir/text_all.gz | tail -n +$[num_dev_sentences+1] | \
    head -n $num_dev_sentences > ${dir}/data/text/dev.txt

  gunzip -c $dir/text_all.gz | tail -n +$[2*num_dev_sentences+1] > \
    ${dir}/data/text/train.txt

  cat $lexicon | awk '{print $1}' | sort | uniq  | awk '
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
min_counts='train=1'
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
               --min-counts="${min_counts}" \
               ${dir}/data/text ${order} ${lm_dir}/work ${unpruned_lm_dir} | tee ${unpruned_lm_dir}/train_lm.log

  get_data_prob.py ${dir}/data/test.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity' | tee ${unpruned_lm_dir}/perplexity_test.log
fi

if [ $stage -le 2 ]; then
  rm ${dir}/data/arpa/${order}gram_big.arpa.gz 2>/dev/null || true
  echo "$0: pruning the LM (to larger size)"
  # Using 5 million n-grams for a big LM for rescoring purposes.
  prune_lm_dir.py --target-num-ngrams=$num_ngrams_large --initial-threshold=0.02 ${unpruned_lm_dir} ${dir}/data/lm_${order}_prune_big \
    2> >(tee -a ${dir}/data/lm_${order}_prune_big/prune_lm.log >&2) || true

  if [ ! -f ${dir}/data/lm_${order}_prune_big/metaparameters ]; then
    grep -q "can not do any pruning" ${dir}/data/lm_${order}_prune_big/prune_lm.log 
    if [ $? -eq 0 ]; then
      echo "$0: LM could not be pruned. Something went wrong!"
      exit 1
    fi

    mkdir -p ${dir}/data/arpa
    format_arpa_lm.py ${unpruned_lm_dir} | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
    echo "$0: No pruning necessary as num-ngrams is less than target"
    exit 0
  fi

  get_data_prob.py ${dir}/data/test.txt ${dir}/data/lm_${order}_prune_big 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_big/perplexity_test.log 

  mkdir -p ${dir}/data/arpa
  format_arpa_lm.py ${dir}/data/lm_${order}_prune_big | gzip -c > ${dir}/data/arpa/${order}gram_big.arpa.gz
fi

if [ $stage -le 3 ]; then
  rm ${dir}/data/arpa/${order}gram_small.arpa.gz 2>/dev/null || true
  echo "$0: pruning the LM (to smaller size)"
  # Using 3 million n-grams for a smaller LM for graph building.  Prune from the
  # bigger-pruned LM, it'll be faster.
  prune_lm_dir.py --target-num-ngrams=$num_ngrams_small ${dir}/data/lm_${order}_prune_big ${dir}/data/lm_${order}_prune_small \
    2> >(tee -a ${dir}/data/lm_${order}_prune_small/prune_lm.log >&2) || true

  if [ ! -f ${dir}/data/lm_${order}_prune_small/metaparameters ]; then
    grep -q "can not do any pruning" ${dir}/data/lm_${order}_prune_small/prune_lm.log
    if [ $? -eq 0 ]; then
      echo "$0: LM could not be pruned. Something went wrong!"
      exit 1
    fi

    ln -s ${order}gram_big.arpa.gz $dir/data/arpa/${order}gram_small.arpa.gz
    exit 0
  fi


  get_data_prob.py ${dir}/data/test.txt ${dir}/data/lm_${order}_prune_small 2>&1 | grep -F '[perplexity' | tee ${dir}/data/lm_${order}_prune_small/perplexity_test.log 

  format_arpa_lm.py ${dir}/data/lm_${order}_prune_small | gzip -c > ${dir}/data/arpa/${order}gram_small.arpa.gz
fi
