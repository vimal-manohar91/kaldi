#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, imp

import numpy as np

import sys

def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty() or type != AssertionError:
        # we are in interactive mode or we don't have a tty-like
        # device, so we call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        import traceback, pdb
        # we are NOT in interactive mode, print the exception...
        traceback.print_exception(type, value, tb)
        print
        # ...then start the debugger in post-mortem mode.
        pdb.pm()

sys.excepthook = info

# this is a bit like a struct, initialized from a string, which describes how to
# set up the statistics-pooling and statistics-extraction components.
# An example string is 'mean(-99:3:9::99)', which means, compute the mean of
# data within a window of -99 to +99, with distinct means computed every 9 frames
# (we round to get the appropriate one), and with the input extracted on multiples
# of 3 frames (so this will force the input to this layer to be evaluated
# every 3 frames).  Another example string is 'mean+stddev(-99:3:9:99)',
# which will also cause the standard deviation to be computed.
class StatisticsConfig:
    # e.g. c = StatisticsConfig('mean+stddev(-99:3:9:99)', 400, 'jesus1-forward-output-affine')
    def __init__(self, config_string, input_dim, input_name):
        self.input_dim = input_dim
        self.input_name = input_name

        m = re.search("(mean|mean\+stddev)\((-?\d+):(-?\d+):(-?\d+):(-?\d+)\)",
                      config_string)
        if m == None:
            raise Exception("Invalid splice-index or statistics-config string: " + config_string)
        self.output_stddev = (m.group(1) != 'mean')
        self.left_context = -int(m.group(2))
        self.input_period = int(m.group(3))
        self.stats_period = int(m.group(4))
        self.right_context = int(m.group(5))
        assert(self.left_context > 0 and self.right_context > 0)
        assert(self.input_period > 0 and self.stats_period > 0)
        assert(self.left_context % self.stats_period == 0)
        assert(self.right_context % self.stats_period == 0)
        assert(self.stats_period % self.input_period == 0)

    # OutputDim() returns the output dimension of the node that this produces.
    def OutputDim(self):
        return self.input_dim * (2 if self.output_stddev else 1)

    # OutputDims() returns an array of output dimensions, consisting of
    # [ input-dim ] if just "mean" was specified, otherwise
    # [ input-dim input-dim ]
    def OutputDims(self):
        return [ self.input_dim, self.input_dim ] if self.output_stddev else [ self.input_dim ]

    # Descriptor() returns the textual form of the descriptor by which the
    # output of this node is to be accessed.
    def Descriptor(self):
        return 'Round({0}-pooling-{1}-{2}, {3})'.format(self.input_name, self.left_context, self.right_context,
                                                       self.stats_period)

    # This function writes the configuration lines need to compute the specified
    # statistics, to the file f.
    def WriteConfigs(self, f):
        print('component name={0}-extraction-{1}-{2} type=StatisticsExtractionComponent input-dim={3} '
              'input-period={4} output-period={5} include-variance={6} '.format(
                self.input_name, self.left_context, self.right_context,
                self.input_dim, self.input_period, self.stats_period,
                ('true' if self.output_stddev else 'false')), file=f)
        print('component-node name={0}-extraction-{1}-{2} component={0}-extraction-{1}-{2} input={0} '.format(
                self.input_name, self.left_context, self.right_context), file=f)
        stats_dim = 1 + self.input_dim * (2 if self.output_stddev else 1)
        print('component name={0}-pooling-{1}-{2} type=StatisticsPoolingComponent input-dim={3} '
              'input-period={4} left-context={1} right-context={2} num-log-count-features=0 '
              'output-stddevs={5} '.format(self.input_name, self.left_context, self.right_context,
                                           stats_dim, self.stats_period,
                                           ('true' if self.output_stddev else 'false')),
              file=f)
        print('component-node name={0}-pooling-{1}-{2} component={0}-pooling-{1}-{2} input={0}-extraction-{1}-{2} '.format(
                self.input_name, self.left_context, self.right_context), file=f)




parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 0 -2,2 0 -4,4 0 -8,8'")
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--ivector-dim", type=int,
                    help="iVector dimension, e.g. 100", default=0)
parser.add_argument("--pnorm-input-dim", type=int,
                    help="input dimension to p-norm nonlinearities")
parser.add_argument("--pnorm-output-dim", type=int,
                    help="output dimension of p-norm nonlinearities")
parser.add_argument("--relu-dim", type=int,
                    help="dimension of ReLU nonlinearities")
parser.add_argument("--sigmoid-dim", type=int,
                    help="dimension of Sigmoid nonlinearities")
parser.add_argument("--pnorm-input-dims", type=str,
                    help="input dimension to p-norm nonlinearities")
parser.add_argument("--pnorm-output-dims", type=str,
                    help="output dimension of p-norm nonlinearities")
parser.add_argument("--relu-dims", type=str,
                    help="dimension of ReLU nonlinearities")
parser.add_argument("--sigmoid-dims", type=str,
                    help="dimension of Sigmoid nonlinearities")
parser.add_argument("--use-presoftmax-prior-scale", type=str,
                    help="if true, a presoftmax-prior-scale is added",
                    choices=['true', 'false'], default = "true")
parser.add_argument("--num-targets", type=int,
                    help="number of network targets (e.g. num-pdf-ids/num-leaves)")
parser.add_argument("--include-log-softmax", type=str,
                    help="add the final softmax layer ", default="true", choices = ["false", "true"])
parser.add_argument("--final-layer-normalize-target", type=float,
                    help="RMS target for final layer (set to <1 if final layer learns too fast",
                    default=1.0)
parser.add_argument("--skip-lda", type=str,
                    help="add lda matrix",
                    choices=['true', 'false'], default = "false")
parser.add_argument("--add-final-sigmoid", type=str,
                    help="add a sigmoid layer as the final layer. Applicable only if skip-final-softmax is true.",
                    choices=['true', 'false'], default = "false")
parser.add_argument("--objective-type", type=str, default="linear",
                    choices = ["linear", "quadratic", "xent"],
                    help = "the type of objective; i.e. quadratic or linear or cross-entropy")
parser.add_argument("config_dir",
                    help="Directory to write config files and variables")
parser.add_argument("--add-l2-regularizer", type=str,
                    help="add output node to do l2 regularization",
                    choices=['true', 'false'], default = "false")

print(' '.join(sys.argv))

args = parser.parse_args()

if not os.path.exists(args.config_dir):
    os.makedirs(args.config_dir)

## Check arguments.
if args.splice_indexes is None:
    sys.exit("--splice-indexes argument is required");
if args.feat_dim is None or not (args.feat_dim > 0):
    sys.exit("--feat-dim argument is required");
if args.num_targets is None or not (args.num_targets > 0):
    sys.exit("--num-targets argument is required");

if args.use_presoftmax_prior_scale == "true":
    use_presoftmax_prior_scale = True
else:
    use_presoftmax_prior_scale = False

if args.skip_lda == "true":
    skip_lda = True
else:
    skip_lda = False

if args.include_log_softmax == "true":
    include_log_softmax = True
else:
    include_log_softmax = False

if args.add_final_sigmoid == "true":
    add_final_sigmoid = True
else:
    add_final_sigmoid = False

delta_window=9
## Work out splice_array e.g. splice_array = [ [ -3,-2,...3 ], [0], [-2,2], .. [ -8,8 ] ]
splice_array = []
left_context = 0
right_context = 0
split_on_spaces = args.splice_indexes.split(" ");  # we already checked the string is nonempty.

if len(split_on_spaces) < 2:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)

for string in split_on_spaces:
    this_layer = len(splice_array)

    this_splices = string.split(",")
    splice_array.append(this_splices)
    # the rest of this block updates left_context and right_context, and
    # does some checking.
    leftmost_splice = 10000
    rightmost_splice = -10000

    for s in this_splices:
        try:
            n = int(s)
            if n < leftmost_splice:
                leftmost_splice = n
            if n > rightmost_splice:
                rightmost_splice = n
        except ValueError:
            if len(splice_array) == 1:
                sys.exit("First dimension of splicing array must not have averaging [yet]")
            x = StatisticsConfig(s, 100, 'foo')

    if leftmost_splice == 10000 or rightmost_splice == -10000:
        sys.exit("invalid element of --splice-indexes: " + string)
    left_context += -leftmost_splice
    right_context += rightmost_splice

left_context = max(0, left_context)
right_context = max(0, right_context)
num_hidden_layers = len(splice_array)
input_dim = len(splice_array[0]) * args.feat_dim  +  args.ivector_dim

if (sum([1 for x in [args.relu_dims, args.relu_dim, args.sigmoid_dims, args.sigmoid_dim, args.pnorm_input_dims, args.pnorm_input_dim] if x]) > 1
    or sum([1 for x in [args.relu_dims, args.relu_dim, args.sigmoid_dims, args.sigmoid_dim, args.pnorm_output_dims, args.pnorm_output_dim] if x]) > 1):
    sys.exit("only one of the dimension options must be provided")

if args.relu_dim is not None:
    nonlin_input_dims = [args.relu_dim] * num_hidden_layers
    nonlin_output_dims = nonlin_input_dims
if args.relu_dims is not None:
    nonlin_input_dims = args.relu_dims.strip().split()
    nonlin_output_dims = nonlin_input_dims
if args.sigmoid_dim is not None:
    nonlin_input_dims = [args.sigmoid_dim] * num_hidden_layers
    nonlin_output_dims = nonlin_input_dims
if args.sigmoid_dims is not None:
    nonlin_input_dims = args.sigmoid_dims.strip().split()
    nonlin_output_dims = nonlin_input_dims
if args.pnorm_input_dims is not None:
    assert(args.pnorm_output_dims is not None)
    nonlin_input_dims = args.pnorm_input_dims.strip().split()
    nonlin_output_dims = args.pnorm_output_dims.strip().split()
if args.pnorm_input_dim is not None:
    assert(args.pnorm_output_dim is not None)
    nonlin_input_dims = [args.pnorm_input_dim] * num_hidden_layers
    nonlin_output_dims = [args.pnorm_output_dim] * num_hidden_layers

nonlin_input_dims = [ int(x) for x in nonlin_input_dims ]
nonlin_output_dims = [ int(x) for x in nonlin_output_dims ]

assert len(nonlin_input_dims) == num_hidden_layers
assert len(nonlin_output_dims) == num_hidden_layers

f = open(args.config_dir + "/vars", "w")
print('left_context=' + str(left_context), file=f)
print('right_context=' + str(right_context), file=f)
# the initial l/r contexts are actually not needed.
# print('initial_left_context=' + str(splice_array[0][0]), file=f)
# print('initial_right_context=' + str(splice_array[0][-1]), file=f)
print('num_hidden_layers=' + str(num_hidden_layers), file=f)
f.close()

f = open(args.config_dir + "/init.config", "w")
print('# Config file for initializing neural network prior to', file=f)
print('# preconditioning matrix computation', file=f)
print('input-node name=input dim=' + str(args.feat_dim), file=f)
list=[ ('Offset(input, {0})'.format(n) if n != 0 else 'input' ) for n in splice_array[0] ]
if args.ivector_dim > 0:
    print('input-node name=ivector dim=' + str(args.ivector_dim), file=f)
    list.append('ReplaceIndex(ivector, t, 0)')
# example of next line:
# output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"
print('output-node name=output input=Append({0})'.format(", ".join(list)), file=f)

if (args.add_l2_regularizer == "true"):
  print('output-node name=output-l2reg input=Append({0})'.format(", ".join(list)), file=f)
f.close()

for l in range(1, num_hidden_layers + 1):
    f = open(args.config_dir + "/layer{0}.config".format(l), "w")
    print('# Config file for layer {0} of the network'.format(l), file=f)
    if l == 1:
        if not skip_lda:
            print('component name=lda type=FixedAffineComponent matrix={0}/lda.mat'.
                  format(args.config_dir), file=f)
        cur_dim = input_dim
    elif l > 1:
        cur_affine_output_dim = nonlin_output_dims[l-2]
        splices = []
        spliced_dims = []
        for s in splice_array[l-1]:
            # the connection from the previous layer
            try:
                offset = int(s)
                # it's an integer offset.
                splices.append('Offset({0}, {1})'.format(cur_output, offset))
                spliced_dims.append(cur_affine_output_dim)
            except:
                # it's not an integer offset, so assume it specifies the
                # statistics-extraction.
                stats = StatisticsConfig(s, cur_affine_output_dim, cur_output)
                splices.append(stats.Descriptor())
                spliced_dims.extend(stats.OutputDims())
        cur_input='Append({0})'.format(', '.join(splices))
        cur_dim = sum(spliced_dims)

    print('# Note: param-stddev in next component defaults to 1/sqrt(input-dim).', file=f)
    print('component name=affine{0} type=NaturalGradientAffineComponent '
          'input-dim={1} output-dim={2} bias-stddev=0'.
          format(l, cur_dim, nonlin_input_dims[l-1]), file=f)
    if args.relu_dims is not None:
        print('component name=nonlin{0} type=RectifiedLinearComponent dim={1}'.
              format(l, nonlin_input_dims[l-1]), file=f)
    elif args.sigmoid_dims is not None:
        print('component name=nonlin{0} type=SigmoidComponent dim={1}'.
              format(l, nonlin_input_dims[l-1]), file=f)
    else:
        print('# In nnet3 framework, p in P-norm is always 2.', file=f)
        print('component name=nonlin{0} type=PnormComponent input-dim={1} output-dim={2}'.
              format(l, nonlin_input_dims[l-1], nonlin_output_dims[l-1]), file=f)
    print('component name=renorm{0} type=NormalizeComponent dim={1} target-rms={2}'.format(
        l, nonlin_output_dims[l-1],
        (1.0 if l < num_hidden_layers else args.final_layer_normalize_target)), file=f)
    print('component name=final-affine type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} param-stddev=0 bias-stddev=0'.format(
          nonlin_output_dims[l-1], args.num_targets), file=f)

    if (args.add_l2_regularizer == "true"):
        print('component name=final-affine-l2reg type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} param-stddev=0 bias-stddev=0'.format(
          nonlin_output_dims[l-1], args.num_targets), file=f)


    if args.include_log_softmax == "true":
      # printing out the next two, and their component-nodes, for l > 1 is not
      # really necessary as they will already exist, but it doesn't hurt and makes
      # the structure clearer.
      if use_presoftmax_prior_scale:
          print('component name=final-fixed-scale type=FixedScaleComponent '
                'scales={0}/presoftmax_prior_scale.vec'.format(
                args.config_dir), file=f)
      print('component name=final-log-softmax type=LogSoftmaxComponent dim={0}'.format(
            args.num_targets), file=f)
    elif add_final_sigmoid:
        print('component name=final-sigmoid type=SigmoidComponent dim={0}'.format(
              args.num_targets), file=f)

    print('# Now for the network structure', file=f)
    if l == 1:
        splices = [ ('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_array[l-1] ]
        if not skip_lda:
            if args.ivector_dim > 0: splices.append('ReplaceIndex(ivector, t, 0)')
            orig_input='Append({0})'.format(', '.join(splices))
            # e.g. orig_input = 'Append(Offset(input, -2), ... Offset(input, 2), ivector)'
            print('component-node name=lda component=lda input={0}'.format(orig_input),
                  file=f)
            cur_input='lda'
        else:
            cur_input='Append({0})'.format(', '.join(splices))
    else:
        # e.g. cur_input = 'Append(Offset(renorm1, -2), renorm1, Offset(renorm1, 2))'
        splices = []
        spliced_dims = []
        for s in splice_array[l-1]:
            # the connection from the previous layer
            try:
                offset = int(s)
                # it's an integer offset.
                splices.append('Offset({0}, {1})'.format(cur_output, offset))
            except:
                # it's not an integer offset, so assume it specifies the
                # statistics-extraction.
                stats = StatisticsConfig(s, nonlin_output_dims[l-2], cur_output)
                stats.WriteConfigs(f)
                splices.append(stats.Descriptor())
        cur_input='Append({0})'.format(', '.join(splices))

    print('component-node name=affine{0} component=affine{0} input={1} '.
          format(l, cur_input), file=f)
    print('component-node name=nonlin{0} component=nonlin{0} input=affine{0}'.
          format(l), file=f)
    print('component-node name=renorm{0} component=renorm{0} input=nonlin{0}'.
          format(l), file=f)
    cur_output = 'renorm{0}'.format(l)

    print('component-node name=final-affine component=final-affine input=renorm{0}'.
          format(l), file=f)

    if (args.add_l2_regularizer == "true"):
        print('component-node name=final-affine-l2reg component=final-affine-l2reg input=renorm{0}'.
          format(l), file=f)

    if args.include_log_softmax == "true":
        if use_presoftmax_prior_scale:
            print('component-node name=final-fixed-scale component=final-fixed-scale input=final-affine',
                  file=f)
            print('component-node name=final-log-softmax component=final-log-softmax '
                  'input=final-fixed-scale', file=f)
        else:
            print('component-node name=final-log-softmax component=final-log-softmax '
                  'input=final-affine', file=f)
        print('output-node name=output input=final-log-softmax objective={0}'.format(args.objective_type), file=f)
    else:
        if add_final_sigmoid:
            print('component-node name=final-sigmoid component=final-sigmoid input=final-affine', file=f)
            print('output-node name=output input=final-sigmoid objective={0}'.format(args.objective_type), file=f)
        else:
            print('output-node name=output input=final-affine objective={0}'.format(args.objective_type), file=f)

    if (args.add_l2_regularizer == "true"):
        print('output-node name=output-l2reg input=final-affine-l2reg objective={0}'.format("quadratic"), file=f)
    f.close()


# component name=nonlin1 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm1 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component name=final-log-softmax type=LogSoftmaxComponent dim=$num_leaves


# ## Write file $config_dir/init.config to initialize the network, prior to computing the LDA matrix.
# ##will look like this, if we have iVectors:
# input-node name=input dim=13
# input-node name=ivector dim=100
# output-node name=output input="Append(Offset(input, -3), Offset(input, -2), Offset(input, -1), ... , Offset(input, 3), ReplaceIndex(ivector, t, 0))"

# ## Write file $config_dir/layer1.config that adds the LDA matrix, assumed to be in the config directory as
# ## lda.mat, the first hidden layer, and the output layer.
# component name=lda type=FixedAffineComponent matrix=$config_dir/lda.mat
# component name=affine1 type=NaturalGradientAffineComponent input-dim=$lda_input_dim output-dim=$pnorm_input_dim bias-stddev=0
# component name=nonlin1 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm1 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component name=final-log-softmax type=LogSoftmax dim=$num_leaves
# # InputOf(output) says use the same Descriptor of the current "output" node.
# component-node name=lda component=lda input=InputOf(output)
# component-node name=affine1 component=affine1 input=lda
# component-node name=nonlin1 component=nonlin1 input=affine1
# component-node name=renorm1 component=renorm1 input=nonlin1
# component-node name=final-affine component=final-affine input=renorm1
# component-node name=final-log-softmax component=final-log-softmax input=final-affine
# output-node name=output input=final-log-softmax


# ## Write file $config_dir/layer2.config that adds the second hidden layer.
# component name=affine2 type=NaturalGradientAffineComponent input-dim=$lda_input_dim output-dim=$pnorm_input_dim bias-stddev=0
# component name=nonlin2 type=PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim
# component name=renorm2 type=NormalizeComponent dim=$pnorm_output_dim
# component name=final-affine type=NaturalGradientAffineComponent input-dim=$pnorm_output_dim output-dim=$num_leaves param-stddev=0 bias-stddev=0
# component-node name=affine2 component=affine2 input=Append(Offset(renorm1, -2), Offset(renorm1, 2))
# component-node name=nonlin2 component=nonlin2 input=affine2
# component-node name=renorm2 component=renorm2 input=nonlin2
# component-node name=final-affine component=final-affine input=renorm2
# component-node name=final-log-softmax component=final-log-softmax input=final-affine
# output-node name=output input=final-log-softmax


# ## ... etc.  In this example it would go up to $config_dir/layer5.config.

