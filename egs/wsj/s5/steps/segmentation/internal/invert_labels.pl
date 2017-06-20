#! /usr/bin/perl 

# Copyright 2017  Vimal Manohar
# Apache 2.0

use strict;
use warnings;

@ARGV == 0 or die "Usage: invert_vector.pl < <vector-text-archive>";

while (<STDIN>) {
  chomp;
  my @F = split;

  my $str = "$F[0] [";
  for (my $i = 2; $i < $#F; $i++) {
    $str = "$str " . (1.0 - $F[$i]);
  }

  print ("$str ]\n");
}
