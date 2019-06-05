#!/bin/bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey, Yenda Trmal)
# Apache 2.0

# See the script steps/scoring/score_kaldi_cer.sh in case you need to evalutate CER

[ -f ./path.sh ] && . ./path.sh

# begin configuration section.
cmd=run.pl
stage=0
decode_mbr=false
stats=true
beam=6
lmwt=10
iter=final
num_subset_per_job=100
scoring_affix=_subset100
#end configuration section.

echo "$0 $@"  # Print the command line for logging
[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [--cmd (run.pl|queue.pl...)] <data-dir> <lang-dir|graph-dir> <decode-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --stage (0|1|2)                 # start scoring script from part-way through."
  echo "    --decode_mbr (true/false)       # maximum bayes risk decoding (confusion network)."
  echo "    --min_lmwt <int>                # minumum LM-weight for lattice rescoring "
  echo "    --max_lmwt <int>                # maximum LM-weight for lattice rescoring "
  exit 1;
fi

data=$1
lang_or_graph=$2
dir=$3

symtab=$lang_or_graph/words.txt

for f in $symtab $dir/lat.1.gz $data/text; do
  [ ! -f $f ] && echo "score.sh: no such file $f" && exit 1;
done


ref_filtering_cmd="cat"
[ -x local/wer_output_filter ] && ref_filtering_cmd="local/wer_output_filter"
[ -x local/wer_ref_filter ] && ref_filtering_cmd="local/wer_ref_filter"
hyp_filtering_cmd="cat"
[ -x local/wer_output_filter ] && hyp_filtering_cmd="local/wer_output_filter"
[ -x local/wer_hyp_filter ] && hyp_filtering_cmd="local/wer_hyp_filter"


if $decode_mbr ; then
  echo "$0: scoring with MBR, word insertion penalty=$word_ins_penalty"
else
  echo "$0: scoring with word insertion penalty=$word_ins_penalty"
fi

nj=$(cat $dir/num_jobs)

utils/split_data.sh $data $nj

cdata=$dir/scoring${scoring_affix}/$(basename $data)_subset$num_subset_per_job
sdata=$dir/scoring${scoring_affix}/$(basename $data)_subset${num_subset_per_job}_split

rm -r $cdata 2>/dev/null || true

$cmd JOB=1:$nj $dir/subset_data.JOB.log \
  utils/subset_data_dir.sh --first $data/split$nj/JOB \
  $num_subset_per_job $sdata/JOB_subset$num_subset_per_job || exit 1

utils/combine_data.sh $cdata $sdata/*_subset$num_subset_per_job || exit 1

mkdir -p $dir/scoring${scoring_affix}
cat $cdata/text | \
  $ref_filtering_cmd > $dir/scoring${scoring_affix}/test_filt.txt || exit 1;
mkdir -p $dir/scoring${scoring_affix}/log

if [ $stage -le 0 ]; then
  if $decode_mbr ; then
    $cmd JOB=1:$nj $dir/scoring${scoring_affix}/log/best_path.JOB.log \
      lattice-copy --include=$sdata/JOB_subset$num_subset_per_job/utt2spk "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-scale --inv-acoustic-scale=$lmwt ark:- ark:- \| \
      lattice-prune --beam=$beam ark:- ark:- \| \
      lattice-mbr-decode  --word-symbol-table=$symtab \
      ark:- ark,t:- \| \
      utils/int2sym.pl -f 2- $symtab \| \
      $hyp_filtering_cmd '>' $dir/scoring${scoring_affix}/hyp.JOB.txt || exit 1
  else
    $cmd JOB=1:$nj $dir/scoring${scoring_affix}/log/best_path.JOB.log \
      lattice-copy --include=$sdata/JOB_subset$num_subset_per_job/utt2spk "ark:gunzip -c $dir/lat.JOB.gz|" ark:- \| \
      lattice-scale --inv-acoustic-scale=$lmwt ark:- ark:- \| \
      lattice-best-path --word-symbol-table=$symtab ark:- ark,t:- \| \
      utils/int2sym.pl -f 2- $symtab \| \
      $hyp_filtering_cmd '>' $dir/scoring${scoring_affix}/hyp.JOB.txt || exit 1
  fi

  for n in `seq $nj`; do
    cat $dir/scoring${scoring_affix}/hyp.$n.txt
  done > $dir/scoring${scoring_affix}/$lmwt.txt
fi

if [ $stage -le 1 ]; then
  $cmd $dir/scoring${scoring_affix}/log/score.log \
    cat $dir/scoring${scoring_affix}/$lmwt.txt \| \
    compute-wer --text --mode=present \
    ark:$dir/scoring${scoring_affix}/test_filt.txt  ark,p:- ">&" $dir/scoring${scoring_affix}/wer_$lmwt || exit 1;
fi


if [ $stage -le 2 ]; then
  wer_file=$dir/${scoring_affix}/wer_$lmwt

  if $stats; then
    mkdir -p $dir/scoring${scoring_affix}/wer_details

    $cmd $dir/scoring${scoring_affix}/log/stats1.log \
      cat $dir/scoring${scoring_affix}/$lmwt.txt \| \
      align-text --special-symbol="'***'" ark:$dir/scoring${scoring_affix}/test_filt.txt ark:- ark,t:- \|  \
      utils/scoring/wer_per_utt_details.pl --special-symbol "'***'" \| tee $dir/scoring${scoring_affix}/wer_details/per_utt \|\
       utils/scoring/wer_per_spk_details.pl $cdata/utt2spk \> $dir/scoring${scoring_affix}/wer_details/per_spk || exit 1;

    $cmd $dir/scoring${scoring_affix}/log/stats2.log \
      cat $dir/scoring${scoring_affix}/wer_details/per_utt \| \
      utils/scoring/wer_ops_details.pl --special-symbol "'***'" \| \
      sort -b -i -k 1,1 -k 4,4rn -k 2,2 -k 3,3 \> $dir/scoring${scoring_affix}/wer_details/ops || exit 1;

    $cmd $dir/scoring${scoring_affix}/log/wer_bootci.log \
      compute-wer-bootci --mode=present \
        ark:$dir/scoring${scoring_affix}/test_filt.txt ark:$dir/scoring${scoring_affix}/$lmwt.txt \
        '>' $dir/scoring${scoring_affix}/wer_details/wer_bootci || exit 1;

  fi
fi

exit 0;

