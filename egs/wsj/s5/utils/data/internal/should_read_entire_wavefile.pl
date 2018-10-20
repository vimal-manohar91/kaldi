#!/bin/perl

while (<>) {
  if (m/sox.*speed/ || m/sox.*trim/) {
    print "true";
    exit(0);
  }
}

print "false";
