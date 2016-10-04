data=$1

cut -d ' ' -f 1,2 $data/segments | utils/utt2spk_to_spk2utt.pl > $data/reco2utt
