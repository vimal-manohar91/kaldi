#!/usr/bin/perl -an

my $id = shift @F;
my $text = join(' ', @F);
$text = lc $text;
print "$id $text\n";
