
all:

EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = segmentation-copy segmentation-get-stats \
					 segmentation-init-from-ali segmentation-to-ali \
					 segmentation-init-from-segments segmentation-to-segments \
					 segmentation-combine-segments segmentation-merge-recordings \
					 segmentation-create-subsegments segmentation-intersect-ali \
					 segmentation-to-rttm segmentation-post-process \
					 segmentation-merge segmentation-split-segments \
					 segmentation-remove-segments \
					 segmentation-init-from-lengths \
					 segmentation-combine-segments-to-recordings \
					 segmentation-create-overlapped-subsegments \
					 segmentation-intersect-segments \
					 segmentation-filter-segments \
					 segmentation-init-from-additive-signals-info \
					 class-counts-per-frame-to-labels \
					 intersect-int-vectors \
					 gmm-global-init-models-from-feats \
					 gmm-global-init-gaussians-from-feats \
					 gmm-init-model-from-gmms \
					 segmentation-split-by-change-points \
					 segmentation-cluster-adjacent-segments \
					 agglomerative-cluster-plda agglomerative-group-cluster \
					 agglomerative-cluster agglomerative-cluster-vectors \
					 agglomerative-cluster-ib kmeans-cluster kmeans-cluster-vectors \
					 ib-scoring-dense \
					 agglomerative-cluster-plda-adjacency \
					 agglomerative-group-cluster-adjacency #\
					 agglomerative-cluster-vector-adjacency \
					 gmm-acc-pdf-stats-segmentation \
					 gmm-est-segmentation gmm-update-segmentation \
					 segmentation-init-from-diarization \
					 segmentation-compute-class-ctm-conf \
					 combine-vector-segments 

OBJFILES =



TESTFILES = 

ADDLIBS =  ../segmenter/kaldi-segmenter.a ../ivector/kaldi-ivector.a ../hmm/kaldi-hmm.a ../gmm/kaldi-gmm.a ../tree/kaldi-tree.a \
					../util/kaldi-util.a ../matrix/kaldi-matrix.a ../base/kaldi-base.a ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

