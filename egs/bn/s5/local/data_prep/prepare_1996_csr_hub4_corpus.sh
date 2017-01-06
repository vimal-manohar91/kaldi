#! /bin/bash

# Copyright 2016  Vimal Manohar
# Apache 2.0.

set -e
set -o pipefail
set -u

nj=4
cmd=run.pl
stage=0

. path.sh
. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 <SOURCE-DIR> <dir>"
  echo " e.g.: $0 /export/corpora/LDC/LDC98T31/ data/local/data/csr96_hub4"
  exit 1
fi

SOURCE_DIR=$1
dir=$2

mkdir -p $dir

ls $SOURCE_DIR/1996_csr_hub4_model/st_train/*.stZ \
  $SOURCE_DIR/1996_csr_hub4_model/st_test/*.stZ | sort > \
  $dir/filelist

mkdir -p $dir/split$nj/

if [ $stage -le 1 ]; then
  eval utils/split_scp.pl $dir/filelist $dir/split$nj/filelist.{`seq -s, $nj`}
  $cmd JOB=1:$nj $dir/log/process_text.JOB.log \
    local/data_prep/csr_hub4_utils/process_filelist.py \
    $dir/split$nj/filelist.JOB $dir
fi

for x in `ls $SOURCE_DIR/1996_csr_hub4_model/st_train/*.stZ`; do
  y=`basename $x`
  name=${y%.stZ}
  echo $dir/${name}.txt.gz
done > $dir/train.filelist

for x in `ls $SOURCE_DIR/1996_csr_hub4_model/st_test/*.stZ`; do
  y=`basename $x`
  name=${y%.stZ}
  echo $dir/${name}.txt.gz
done > $dir/test.filelist
