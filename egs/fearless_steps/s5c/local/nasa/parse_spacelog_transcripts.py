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

    parser.add_argument("input_transcripts")
    parser.add_argument("output_transcripts")

    return parser.parse_args()


def normalize_transcript(trans):
    text = unidecode.unidecode(trans)

    text = re.sub(r'\[glossary:[^\]\|]+\|([^\]]+)\]', r' \1 ', text, flags=re.IGNORECASE)  # Remove glossary, but keep the actual term e.g. [glossary:ASCS|ASCS] --> ASCS
    text = re.sub(r'\[time:[^\]\|]+\|([^\]]+)\]', r' \1 ', text, flags=re.IGNORECASE)  # Remove time, but keep the actual words or time said
    text = re.sub(r'\[glossary:([^\]]+)\]', r' \1 ', text, flags=re.IGNORECASE)  # Remove glossary
    text = re.sub(r'\[time:([^\]]+)\]', r' \1 ', text, flags=re.IGNORECASE)  # Remove time
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

    with open(args.input_transcripts, encoding='utf-8') as transcript_reader, \
            open(args.output_transcripts, 'w', encoding='utf-8') as transcript_writer:
        trans = []
        for line in transcript_reader:
            line = line.strip()
            if not len(line):
                if len(trans):
                    for text in parse_transcript(' '.join(trans)):
                        print (text, file=transcript_writer)
                    trans = []
                continue
            if re.match(r'\[\S+\]', line):
                continue
            if re.match(r'^_\S+', line):
                continue

            m = re.match(r'^[^:]+:\s*(.+)$', line)
            if m:
                this_trans = m.group(1)
            else:
                this_trans = line
            trans.append(this_trans)

        if len(trans):
            for text in parse_transcript(' '.join(trans)):
                print (text, file=transcript_writer)


if __name__ == "__main__":
    main()
