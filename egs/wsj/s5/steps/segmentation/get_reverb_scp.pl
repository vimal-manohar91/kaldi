#! /usr/bin/perl
use strict;
use warnings;

my $num_reps = $ARGV[0];

while (<STDIN>) {
  if (m/^(sp[0-9.]+-)(.+)$/) {
    for (my $i = 1; $i <= $num_reps; $i++) {
      print ($1 . "rev" . $i . "_" . $2 . "\n");
    }
  } else {
    for (my $i = 1; $i <= $num_reps; $i++) {
      print ("rev" . $i . "_" . $_);
    }
  }
}
