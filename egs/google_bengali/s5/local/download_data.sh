#!/bin/bash
# Copyright (c) 2018, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
datadir=./corpus
# End configuration section
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

mkdir -p $datadir/
if  [ ! -f $datadir/asr_bengali/utt_spk_text.tsv ] ; then
  [ ! -f $datadir/asr_bengali_0.zip ] && \
    wget -O $datadir/asr_bengali_0.zip http://www.openslr.org/resources/53/asr_bengali_0.zip
  [ ! -f $datadir/asr_bengali_1.zip ] && \
    wget -O $datadir/asr_bengali_1.zip http://www.openslr.org/resources/53/asr_bengali_1.zip
  [ ! -f $datadir/utt_spk_text.tsv ] && \
    wget -O $datadir/utt_spk_text.tsv http://www.openslr.org/resources/53/utt_spk_text.tsv

  unzip -o $datadir/asr_bengali_0.zip -d $datadir
  unzip -o $datadir/asr_bengali_1.zip -d $datadir
fi


