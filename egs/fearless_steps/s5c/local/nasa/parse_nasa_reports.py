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

    parser.add_argument("input_file")
    parser.add_argument("output_file")

    return parser.parse_args()


def normalize_transcript(trans):
    text = unidecode.unidecode(trans)

    text = re.sub(r'[<][^>]+[>]', '', text)
    text = re.sub(r'[</][^>]+[>]', '', text)  # remove html tags

    return normalize.normalize_text(text)


def parse(para):
    para = re.sub(r'\n|\r', ' ', para)
    for sent in nltk.sent_tokenize(para):
        sent = sent.strip()
        if len(sent) == 0:
            continue
        output = normalize_transcript(sent)
        if len(output) == 0:
            continue
        yield output


def main():
    args = get_args()

    with open(args.input_file, encoding='utf-8') as reader, \
            open(args.output_file, 'w', encoding='utf-8') as writer:
        for line in reader:
            j = json.loads(line)
            for text in parse(j['trans']):
                print (text, file=writer)


if __name__ == "__main__":
    main()
