#! /usr/bin/perl 
#
# Given a lexicon in the format "word ph1 ph2 ph3 ... pN", this script
# extracts all the phones and prints the unique phones. (bash cut -f2 can do
# this too but it is dependent on correctly specifying the delimiter
# and ensuring your locale is set correctly to read unicode symbols. This
# script hides those dependencies from the user.)
# 
# amitdas@illinois.edu
# ============================================================================
# Revision History
# Date 		Author	Description of Change
# 06/17/15	ad 		Created 
# ============================================================================

my $usage = "Usage:
>perl $0 dict.txt > phones.txt
\n";

#use strict;
use Getopt::Long;
die "$usage" unless(@ARGV == 1);
binmode(STDOUT, ":encoding(UTF-8)");
binmode(STDERR, ":encoding(UTF-8)");


my ($dictf) = @ARGV;
%DICT = ();

# Read the dict file to populate the phones in a hash
open(DICTF,'<:encoding(UTF-8)', $dictf) || die "Unable to read from $dictf: $!";
foreach $line (<DICTF>) {
	($line =~ /^\;/) && next;	
	my(@recs) = split(/\s+/,$line);
	foreach $p (@recs[1..$#recs]) {
		if (!defined $DICT{$p}) {
			$DICT{$p} = 1;
			print STDERR "Added $p\n"; 
		}
	}
}
close (DICTF);

foreach $p (sort keys %DICT) {
	print STDOUT "$p $p\n";
}
