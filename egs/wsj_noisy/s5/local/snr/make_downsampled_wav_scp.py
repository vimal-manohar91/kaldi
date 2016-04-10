#! /usr/bin/env python

# Copyright 2016    Vimal Manohar
# Apache 2.0

def GetArgs():
    parser = argparse.ArgumentParser(description = "Modifies wav.scp to "
                                                " add downsamping using sox")

    parser.add_argument("in_scp", metavar = "<in-wav-scp>",
                        help = "Original wav.scp")
    parser.add_argument("out_scp", metavar = "<out-wav-scp>",
                        help = "Output wav.scp with downsampling in pipe")
    parser.add_argument("--channel", default = 1, type = int,
                        help = "Channel of wav input")
    parser.add_argument("--sample-frequency", default = 8000, type = float,
                        help = "Sampling frequency of downsampled wavs")
    parser.add_argument("--extra-sox-opts", type = str,
                        help = "Additional options added to sox after "
                        "channel and sample-frequency e.g. bits"
 /usr/bin/sox -t wav - -r 8000 -c 1 -b 16 -t wav - downsample |
