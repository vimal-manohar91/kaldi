#!/usr/bin/env python3

import argparse
import json
import nltk
import re
import sys
import unidecode

sys.path.insert(0, 'local/nasa')
import normalize


def get_args():
    parser = argparse.ArgumentParser(
        description="Parse transcripts from Apollo Flight Journal")

    parser.add_argument("input_json_lines")
    parser.add_argument("output_transcripts")

    return parser.parse_args()


def normalize_transcript(trans):
    text = unidecode.unidecode(trans)

    text = re.sub(r'[<][^>]+[>]', '', text)
    text = re.sub(r'[</][^>]+[>]', '', text)  # remove html tags
    text = re.sub(r'([A-Z]-?[A-Z](-?[A-Z])*)\s\[[^\]]\]', r'\1', text)  # remove expansions of abbreviations which are inside square brackets after the abbreviations

    return normalize.normalize_text(text)


def parse_transcript(transcript):
    for text in transcript.strip().split('\n'):
        if len(text) == 0:
            continue

        for sent in nltk.sent_tokenize(text):
            sent = sent.strip()
            if len(sent) == 0:
                continue
            output = normalize_transcript(sent)
            if len(output) == 0:
                continue
            yield output


def main():
    args = get_args()

    with open(args.input_json_lines, encoding='utf-8') as json_reader, \
            open(args.output_transcripts, 'w', encoding='utf-8') as transcript_writer:
        for line in json_reader:
            j = json.loads(line)
            for text in parse_transcript(j['trans']):
                print (text, file=transcript_writer)


if __name__ == "__main__":
    main()
