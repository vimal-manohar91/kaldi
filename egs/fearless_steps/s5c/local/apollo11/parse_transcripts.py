#!/usr/bin/env python3

import os.path
import re
import sys


def normalize_text(text):
    words = text.split()
    for i, w in enumerate(words):
        w = re.sub(r"[?!,]", "", w)
        parts = w.split(".")
        if len(parts) > 2:
            # This is an abbreviation
            pass
        else:
            if len(parts) == 2 and parts[1] == "":
                w = parts[0]  # remove the period
        words[i] = w

    text = " ".join(words)

    text = re.sub(r"bio med", r"biomed", text)
    text = re.sub(r"aboutthe", r"about the", text)
    text = re.sub(r"intermittant", r"intermittent", text)
    text = re.sub(r"intothe", r"into the", text)
    text = re.sub(r"landingsite", r"landing site", text)

    text = text.lower()

    text = re.sub(r"\[unk\]", r"<unk>", text)
    #text = re.sub(r"\[laughter\]", r"[laughter]", text)

    return text


def main():
    for filename in map(str.rstrip, sys.stdin):
        file_id = os.path.splitext(os.path.basename(filename))[0]
        with open(filename, encoding='utf-8') as f:
            utt_counter = 0
            for line in f.readlines():
                parts = line.strip().split()
                utt_id = parts[0]
                text = " ".join(parts[1:])
                text = normalize_text(text)
                parts = utt_id.split('_')
                if len(parts) == 14:
                    begin_time = float(parts[12]) / 1000.0
                    end_time = float(parts[13]) / 1000.0
                    wav_id = "_".join(parts[0:10])
                elif len(parts) == 10:
                    begin_time = float(parts[8]) / 1000.0
                    end_time = float(parts[9]) / 1000.0
                    wav_id = "_".join(parts[0:6])
                elif len(parts) == 9:
                    begin_time = float(parts[7]) / 1000.0
                    end_time = float(parts[8]) / 1000.0
                    wav_id = "_".join(parts[0:6])
                else:
                    raise Exception("Could not parse utterance ID: {} in {}".format(line, filename))

                if wav_id != file_id:
                    raise Exception("{} != {}".format(wav_id, file_id))

                speaker_id = wav_id

                text = normalize_text(text)

                print('%s\t%s\t%.2f\t%.2f\t%s\t%s\t%s\t%s' % (
                    file_id, wav_id, begin_time, end_time,
                    utt_id, speaker_id, "U", text))


if __name__ == "__main__":
    main()
