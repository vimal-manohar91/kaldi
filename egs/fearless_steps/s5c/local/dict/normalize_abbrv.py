#!/usr/bin/env python3

import argparse
import re
import sys

def read_word_list(word_list):
    words = {}
    for line in open(word_list, encoding='utf-8'):
        parts = line.strip().split()
        assert len(parts) >= 1
        word = parts[0]
        if word in ['<s>', '</s>', '<eps>'] or re.match(r'#\d+', word):
            continue
        words[parts[0]] = True
    return words


def normalize_abbrv(word):
    word = word.lower()
    word = re.sub(r'([a-z])\B', r'\1.', word)
    return word + '.'


def get_args():
    parser = argparse.ArgumentParser(
        description="""Reads lines from standard input, normalizes abbreviations
        and writes them to standard output.""")

    parser.add_argument("wordlist", help="Wordlist")

    return parser.parse_args()


def main():
    args = get_args()
    wordlist = read_word_list(args.wordlist)
    for line in sys.stdin.readlines():
        output = []
        for w in line.strip().split():
            if w.lower() == w:
                output.append(w)
            elif w.lower() in wordlist:
                output.append(w.lower())
            else:
                output.append(normalize_abbrv(w))
        print (' '.join(output))


if __name__ == "__main__":
    main()
