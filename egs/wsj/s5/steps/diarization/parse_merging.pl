#! /usr/bin/perl 

use strict;
use warnings;

if (scalar @ARGV != 2 && scalar @ARGV != 1) {
  die "Usage: $0 [ --per-spk ] <symbol-table> <ref-mapping>";
}

open UTTS, $ARGV[0];
open UTT2SPK, $ARGV[1];

my %int2utt;
while (<UTTS>) {
  chomp;
  my @F = split;
  $int2utt{$F[1]} = $F[0];
}

my %utt2spk;
while (<UTT2SPK>) {
  chomp;
  my @F = split;
  $utt2spk{$F[0]} = $F[1];
}

my $merge_line = 0;
my $line = "";
while (<STDIN>) {
  if (m/Merging/) {
    $merge_line = 1;
  } 

  if ($merge_line >= 1) {
    chomp;
    $line = "$line $_";
    $merge_line++;
  }

  if ($merge_line < 4) {
    next;
  }
  
  $merge_line = 0;
  #awk '/Merging/{i=1} {if (i >= 1) { printf $0" "; i++;} if (i == 4) { i = 0; print ""; } }'

  if ($line !~ m/GrCL \[ (\d+( \d+)*) \] .+ and .+GrCL \[ (\d+( \d+)*) \] .+ with distance (\S+)/) {
    die "Could not match line $line"
  }

  $line = "";
  my $dist = $5;

  my $cluster1 = $1;
  my $utt_cluster1 = "";
  my @F = split / /, $cluster1;
  my %Fspk;
  for (my $i = 0; $i <= $#F; $i++) {
    my $int = $F[$i] + 1;
    my $utt = $int2utt{$int};
    $F[$i] = $utt;
    $utt_cluster1 = "$utt_cluster1 $utt";

    if (scalar @ARGV == 2) {
      defined $utt2spk{$utt} or die "Could not find map for $utt in $ARGV[1]";
      my $spk = $utt2spk{$utt};
      $F[$i] = "$int-$utt-$spk";
      if (!defined $Fspk{$spk}) {
        $Fspk{$spk} = 0;
      }
      $Fspk{$spk}++;
    }
  }
  $cluster1 = join(" ", @F);
  
  my $cluster2 = $3;
  my $utt_cluster2 = "";
  my @G = split / /, $cluster2;
  my %Gspk;
  for (my $i = 0; $i <= $#G; $i++) {
    my $int = $G[$i] + 1;
    my $utt = $int2utt{$int};
    $G[$i] = $utt;
    $utt_cluster2 = "$utt_cluster2 $utt";

    if (scalar @ARGV == 2) {
      my $spk = $utt2spk{$utt};
      $G[$i] = "$int-$utt-$spk";
      if (!defined $Gspk{$spk}) {
        $Gspk{$spk} = 0;
      }
      $Gspk{$spk}++;
    }
  }
  $cluster2 = join(" ", @G);

  if (scalar @ARGV == 2) {
    my $score = 0;
    my $missing = 0;
    my $num_spk = 0;
    while (my ($spk, $count) = (each %Fspk)) {
      if (!defined $Gspk{$spk}) {
        $num_spk++;
        $missing++;
        next;
      }
      $score += $count * $Gspk{$spk};
      $num_spk++;
    }
    
    while (my ($spk, $count) = (each %Gspk)) {
      if (!defined $Fspk{$spk}) {
        $num_spk++;
        $missing++;
        next;
      }
    }

    print "$5 $num_spk -$missing [ $cluster1 ] [ $cluster2 ]\n";
  } else {
    print "$5 [ $cluster1 ] [ $cluster2 ]\n";
  }
}
