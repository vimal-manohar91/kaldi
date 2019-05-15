#!/usr/bin/perl

if ($#ARGV ne 0) {
  print STDERR "Usage: $0 <words> < <input-lexicon> > <output-lexicon>";
  exit(1);
}

open (WORDS, "<", $ARGV[0]);

my %words;
while (<WORDS>) {
  chomp; $words{$_} = 1;
}

while (<STDIN>) {
  chomp; my @F = split; 
  my $w = $F[0];
  if (defined $words{$w}) {
    print "$_\n";
  }
}
