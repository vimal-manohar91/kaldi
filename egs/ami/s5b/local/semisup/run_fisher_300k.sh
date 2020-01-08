text=~/workspace_chain_ts/egs/aspire/s5/data/train_300k_dev/text

local/fisher_train_lms_pocolm.sh --text $text --lexicon data/local/fisher_dict/lexicon.txt \
  --dir data/local/fisher_300k_pocolm --num-ngrams-large 2500000

local/fisher_create_test_lang.sh --arpa-lm data/local/fisher_300k_pocolm/data/arpa/4gram_big.arpa.gz --lang data/lang_fisher --dir data/lang_fisher_300k_test
