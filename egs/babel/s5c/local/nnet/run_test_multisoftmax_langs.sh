#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

# This example script trains DNN with <MultiSoftmax> output on top of FMLLR features.
# The network is trained on multiple languages simultaneously creating a separate softmax layer per language
# while sharing hidden layers across all languages.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

. ./path.sh ## Source the tools/utils (import the queue.pl)

stage=0
#. utils/parse_options.sh || exit 1;

set -u 
set -e
set -o pipefail
set -x

n=0
for i in "$@"
do
case $i in
    --l=*)
    lang="${i#*=}"
    shift; n=$((++n)); 
    ;;    
    --ali=*)
    ali="${i#*=}"  # tri3b_ali
    shift; n=$((++n)); 
    ;;        
    --data=*)
    data="${i#*=}"
    shift; n=$((++n)); 
    ;;
    --nnet=*)
    nnet="${i#*=}"
    shift; n=$((++n)); 
    ;; 
    --startnode=*)
    startnode="${i#*=}"
    shift; n=$((++n)); 
    ;;
    --outputdim=*)
    outputdim="${i#*=}"
    shift; n=$((++n)); 
    ;;
    --graphdir=*)  # tri3b/graph
    graphdir="${i#*=}"
    shift; n=$((++n)); 
    ;;
    *)
    echo "Unknown argument: ${i#*=}, exiting"; exit 1 
    ;;    
esac
[[ $(( n%4 )) -eq 0 ]] && break;
done

#nlangs=$(( n/3 - 1))

# Check if all the user i/p directories exist
echo "lang = ${lang}, alidir = ${ali}, datadir = ${data}, nnet = ${nnet}"
[ ! -e ${ali} ] && echo  "Missing  ${ali}" && exit 1
[ ! -e ${data} ] && echo "Missing ${data}" && exit 1
[ ! -e ${nnet} ] && echo "Missing ${nnet}" && exit 1
[ ! -e ${graphdir} ] && echo "Missing ${graphdir}" && exit 1

# Make the features
thisdata=data-fmllr-multisoftmax/test
testdir=$thisdata

if [ $stage -le 0 ]; then   
    # Store fMLLR features, so we can test on them easily,
	dir=$thisdata/${lang}; mkdir -p $dir
	gmm_ali=${ali}
	data_train=${data}	
	echo "Language = ${lang}: Generating features from datadir = ${data} and saving in $dir"	
	
	steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
     --transform-dir ${gmm_ali} \
     $dir ${data_train} ${gmm_ali} $dir/log $dir/data || exit 1         
fi



ali1_dim=$(hmm-info ${ali}/final.mdl | grep pdfs | awk '{ print $NF }')
ali1_pdf="ark:ali-to-pdf ${ali}/final.mdl 'ark:gunzip -c ${ali}/ali.*.gz |' ark:- |"
ali1_dir=${ali}

thisdir=exp/dnn4e-fmllr_multisoftmax
dir=$thisdir/test/$lang
# Test block softmax system,
if [ $stage -le 2 ]; then  
  # Create files used in decdoing, missing due to --labels use,
  analyze-counts --binary=false "$ali1_pdf" $dir/ali_train_pdf.counts || exit 1
  copy-transition-model --binary=false $ali1_dir/final.mdl $dir/final.mdl || exit 1
  cp $ali1_dir/tree $dir/tree || exit 1
  
  endnode=$(( startnode + ali1_dim - 1 ))
  # Rebuild network, <MultiSoftmax> is removed, and neurons from 1st block are selected,
  nnet-concat "nnet-copy --remove-last-components=1 $nnet - |" \
    "echo '<Copy> <InputDim> $output_dim <OutputDim> $ali1_dim <BuildVector> $startnode:$endnode </BuildVector>' | nnet-initialize - - |" \
    "echo '<Softmax> <InputDim> $ali1_dim <OutputDim> $ali1_dim' | nnet-initialize - - |" \
    $dir/final.nnet.lang1 || exit 1
    
  # Decode (reuse HCLG graph)
  steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $graphdir $testdir $dir/decode || exit 1;
    
  #steps/nnet/decode.sh --nj 20 --cmd "$decode_cmd" --config conf/decode_dnn.config --acwt 0.1 \
    --nnet $dir/final.nnet.lang1 \
    $gmm/graph_ug $dev $dir/decode_ug || exit 1;
fi

exit 0

# Getting results [see RESULTS file]
# for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
