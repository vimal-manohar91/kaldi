all:

include ../kaldi.mk

TESTFILES = segmentation-io-test information-bottleneck-clusterable-test

OBJFILES = segment.o segmentation.o segmentation-utils.o \
					 segmentation-post-processor.o \
					 information-bottleneck-clusterable.o \
					 information-bottleneck-cluster-utils.o \
					 adjacency-clusterable.o pair-clusterable.o \
					 gaussian-stats-clusterable.o segmentation-cluster-utils.o \
					 iterative-bottom-up-cluster.o plda-clusterable.o \
					 group-clusterable.o ivector-clusterable.o

LIBNAME = kaldi-segmenter

ADDLIBS = ../ivector/kaldi-ivector.a ../tree/kaldi-tree.a ../gmm/kaldi-gmm.a \
	../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

