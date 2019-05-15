#!/usr/bin/env perl

if ($#ARGV ne 1) {
  die "Usage: $0 <abbreviation-wordlist> <non-abbreviation-wordlist> < <wordlist>";
}

open (ABBRV, ">", $ARGV[0]);
open (NORMAL, ">", $ARGV[1]);

while (<STDIN>) {
  chomp;
  if (m/[a-z]/) {
    # contains at least one letter
    if (m/[.]/) {
      # probably is an abbreviation
      print ABBRV ("$_\n");
    } else {
      # words that are not abbreviations
      print NORMAL ("$_\n");
    }
  }
}
