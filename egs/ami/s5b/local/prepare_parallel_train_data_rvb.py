#!/usr/bin/env python3

map_file = "data/sdm1/train_ihmdata/ihmutt2utt"
num_data_reps=3

for line in open(map_file):
    parts = line.strip().split()

    for i in range(1, num_data_reps + 1):
        for sp in [0.9, 1.1]:
            sp_ihmutt = "sp{0}-{1}".format(sp, parts[0])
            sp_utt = "sp{0}-{1}".format(sp, parts[1])

            rev_ihmutt = "rev{0}_{1}".format(i, sp_ihmutt)
            rev_utt = "rev{0}_{1}".format(i, sp_utt)

            print ("{0} {1}".format(rev_ihmutt, rev_utt))

        rev_ihmutt = "rev{0}_{1}".format(i, parts[0])
        rev_utt = "rev{0}_{1}".format(i, parts[1])
        print ("{0} {1}".format(rev_ihmutt, rev_utt))
