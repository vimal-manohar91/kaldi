#!/usr/bin/env perl
#===============================================================================
# Copyright 2018  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

use strict;
use warnings;
use utf8;

my %LEX;
my $field_begin;
my $field_end;
my $map_oov = "<UNK>";
for(my $x = 0; $x < 2; $x++) {
  if ($ARGV[0] eq "--map-oov") {
    shift @ARGV;
    $map_oov = shift @ARGV;
    if ($map_oov eq "-f" || $map_oov =~ m/words\.txt$/ || $map_oov eq "") {
      # disallow '-f', the empty string and anything ending in words.txt as the
      # OOV symbol because these are likely command-line errors.
      die "the --map-oov option requires an argument";
    }
  }
  if ($ARGV[0] eq "-f") {
    shift @ARGV;
    my $field_spec = shift @ARGV;
    if ($field_spec =~ m/^\d+$/) {
      $field_begin = $field_spec - 1; $field_end = $field_spec - 1;
    }
    if ($field_spec =~ m/^(\d*)[-:](\d*)/) { # accept e.g. 1:10 as a courtesty (properly, 1-10)
      if ($1 ne "") {
        $field_begin = $1 - 1;  # Change to zero-based indexing.
      }
      if ($2 ne "") {
        $field_end = $2 - 1;    # Change to zero-based indexing.
      }
    }
    if (!defined $field_begin && !defined $field_end) {
      die "Bad argument to -f option: $field_spec";
    }
  }
}

if (@ARGV < 1) {
  print STDERR "Usage: $0 [options] symtab [input transcriptions] > output transcriptions\n" .
    "options: [--map-oov <oov-symbol> ]  [-f <field-range> ]\n" .
      "note: <field-range> can look like 4-5, or 4-, or 5-, or 1.\n";
    die;
}

open(LEX, $ARGV[0]) or die "Cannot open the lexicon file $ARGV[0]: $!";
while(<LEX>) {
  chomp;
  next unless $_;
  my @F = split;
  $LEX{$F[0]} = 1;
}
close(LEX);
shift;

my $num_warning = 0;
my $max_warning = 2000;

while (<>) {
  my @A = split(" ", $_);
  my @B = ();
  for (my $n = 0; $n < @A; $n++) {
    my $w = $A[$n];
    my $i = $LEX{$w};
    if ( (!defined $field_begin || $n >= $field_begin)
         && (!defined $field_end || $n <= $field_end)) {
      if (!defined ($i)) {
        if ($num_warning++ < $max_warning) {
          print STDERR "$0: replacing $w with $map_oov\n";
          if ($num_warning == $max_warning) {
            print STDERR "$0: not warning for OOVs any more times\n";
          }
        }
        $w = $map_oov;
      }
    }
    push @B, $w;
  }
  print join(" ", @B);
  print "\n";
}

if ($num_warning > 0) {
  print STDERR "** Replaced $num_warning instances of OOVs with $map_oov\n";
}

exit(0);
