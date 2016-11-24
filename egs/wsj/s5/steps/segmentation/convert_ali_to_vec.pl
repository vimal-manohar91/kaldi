#! /usr/bin/perl

while (<>) {
  chomp;
  my @F = split;

  printf ("$F[0] [ ");
  for (my $i = 1; $i <= $#F; $i++) {
    printf ("$F[$i] ");
  }
  print ("]"); 
}
