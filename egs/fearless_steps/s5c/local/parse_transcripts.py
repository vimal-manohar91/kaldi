#!/usr/bin/env python3

import json
import os.path
import re
import sys


def normalize_text(text):
    words = text.split()
    for i, w in enumerate(words):
        w = re.sub(r'[?!,";]', '', w).strip()  # remove punctuations
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
    #text = re.sub(r"\[laughter\]", r"<laughter>", text)

    return text


def main():
    for filename in map(str.rstrip, sys.stdin):
        file_id = os.path.splitext(os.path.basename(filename))[0]
        with open(filename) as f:
            out = json.load(f, encoding='utf-8')
            utt_counter = 0
            for utt in out:
                wav_id = '{:0>20}'.format(file_id)
                speaker = '{}_{:0>17}'.format(file_id, utt.get('speakerID', 'MISSING'))
                assert len('{:0>17}'.format(utt.get('speakerID', 'MISSING'))) == 17
                speaker_id = '%s_%s' % (wav_id, speaker)
                utt_id = '%s_%04d' % (speaker_id, utt_counter)
                utt_counter += 1
                begin_time=utt['startTime']
                end_time=utt['endTime']
                text = utt['words']
                text.replace('\t', ' ')
                text = normalize_text(text)

                print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' % (
                    file_id, wav_id, begin_time, end_time,
                    utt_id, speaker_id, "U", text))


if __name__ == "__main__":
    main()
