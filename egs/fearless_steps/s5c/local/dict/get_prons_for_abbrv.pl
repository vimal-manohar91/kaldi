#!/usr/bin/perl

if ($#ARGV ne 2) {
  print STDERR "This script reads an input lexicon and a ";
  print STDERR "list of abbreviations and compound words and ";
  print STDERR "tries to get pronunciations for them\n";
  print STDERR "Usage: $0 <input-lexicon> <oov-parts> <not-found-list> < <wordlist> > <output-lexicon>";
  exit(1);
}

open(LEX, "<", $ARGV[0]);   # input lexicon
open(OOVPART, ">", $ARGV[1]);  # file to write parts for which we don't have pronunciations
open(NOTFOUND, ">", $ARGV[2]);  # file to write the words for which we don't get pronunciations

my %oov_parts;
my %not_found_words;

my %lex;
while (<LEX>) {
  chomp;
  my @F = split;
  my $w = shift @F;
  $lex{$w} = join(" ", @F);
}
close(LEX);
$lex{"'s"} = "s";

while (<STDIN>) {
  # read words one by one -- each line is a word
  my $pron = "";
  my $found_pron = 1;
  chomp; my @F = split /-/, $_;
  
  if (defined($lex{$_})) {
    die "Word $_ is already in lexicon";
  }

  foreach my $w (@F) {
    # split word into compound word
    if (defined($lex{$w})) {
      $pron = $pron . $lex{$w}. " ";
    } else {
      my @parts = split /\./, $w;
      foreach my $part (@parts) {
        if (defined($lex{$part})) {
          # this part is a word
          $pron = $pron . $lex{$part} . " ";
        } elsif (defined($lex{$part . "."})) {
          # this part is a letter followed by a .
          $pron = $pron . $lex{$part . "."} . " ";
        } else {
          $found_pron = 0;
          $oov_parts{$part} = 1;
        }
      }
    }
  }

  if ($found_pron eq 1) {
    chomp($pron);
    print "$_ $pron\n";
  } else {
    $not_found_words{$_} = 1;
  }
}

foreach my $w (keys %not_found_words) {
  print NOTFOUND "$w\n";
}

foreach my $p (keys %oov_parts) {
  print OOVPART "$p\n";
}

close (NOTFOUND);
close (OOVPART);
