#!/usr/bin/env python3

import sys

utt2text = {}

for line in open(sys.argv[1], encoding="utf-8"):
  parts = line.strip().split()
  utt_id = parts[0]
  try:
    text = parts[1:]
  except IndexError:
    text = []
  utt2text[utt_id] = text

for line in sys.stdin.readlines():
  parts = line.strip().split()
  reco_id = parts[0]
  reco_text = []
  for utt in parts[1:]:
    text = utt2text[utt]
    if len(text) > 1 or (len(text) == 1 and text[0].lower() != "a"):
      reco_text.extend(text)

  print("{} {}".format(reco_id, " ".join(reco_text)))
