#!/usr/bin/env python

# tdnn or RNN with 'jesus layer'

#  inputs to jesus layer:
#      - for each spliced version of the previous layer the output (of dim  --jesus-forward-output-dim)

#  outputs of jesus layer:
#     for all layers:
#       --jesus-forward-output-dim


# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings


parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--splice-indexes", type=str,
                    help="Splice[:recurrence] indexes at each hidden layer, e.g. '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'. "
                    "Note: recurrence indexes are optional, may not appear in 1st layer, and must be "
                    "either all negative or all positive for any given layer.")
parser.add_argument("--feat-dim", type=int,
                    help="Raw feature dimension, e.g. 13")
parser.add_argument("--ivector-dim", type=int,
                    help="iVector dimension, e.g. 100", default=0)
parser.add_argument("--include-log-softmax", type=str,
                    help="add the final softmax layer ", default="true", choices = ["false", "true"])
parser.add_argument("--xent-regularize", type=float,
                    help="For chain models, if nonzero, add a separate output for cross-entropy "
                    "regularization (with learning-rate-factor equal to the inverse of this)",
                    default=0.0)
parser.add_argument("--xent-separate-forward-affine", type=str, 
                    help="if using --xent-regularize, gives it separate last-but-one weight matrix",
                    default=False, choices = ["false", "true"])
parser.add_argument("--use-repeated-affine", type=str,
                    help="if true use RepeatedAffineComponent, else BlockAffineComponent (i.e. no sharing)",
                    default="true", choices = ["false", "true"])
parser.add_argument("--final-layer-learning-rate-factor", type=float,
                    help="Learning-rate factor for final affine component",
                    default=1.0)
parser.add_argument("--self-repair-scale", type=float,
                    help="Small scale involved in fixing derivatives, if supplied (e.g. try 0.00001)",
                    default=0.00001)
parser.add_argument("--jesus-hidden-dim", type=int,
                    help="hidden dimension of Jesus layer.", default=10000)
parser.add_argument("--jesus-forward-output-dim", type=int,
                    help="part of output dimension of Jesus layer that goes to next layer",
                    default=1000)
parser.add_argument("--jesus-forward-input-dim", type=int,
                    help="Input dimension of Jesus layer that comes from affine projection "
                    "from the previous layer (same as output dim of forward affine transform)",
                    default=1000)
parser.add_argument("--final-hidden-dim", type=int,
                    help="Final hidden layer dimension-- or if <0, the same as "
                    "--jesus-forward-input-dim", default=-1)
parser.add_argument("--final-layer-normalize-target", type=float,
                        help="RMS target for final layer (set to <1 if final layer learns too fast",
                        default=1.0)
parser.add_argument("--num-jesus-blocks", type=int,
                    help="number of blocks in Jesus layer.  All configs of the form "
                    "--jesus-*-dim will be rounded up to be a multiple of this.",
                    default=100);
parser.add_argument("--jesus-stddev-scale", type=float,
                    help="Scaling factor on parameter stddev of Jesus layer (smaller->jesus layer learns faster)",
                    default=1.0)
parser.add_argument("--clipping-threshold", type=float,
                    help="clipping threshold used in ClipGradient components (only relevant if "
                    "recurrence indexes are specified).  If clipping-threshold=0 no clipping is done",
                    default=15)
parser.add_argument("--num-targets", type=int,
                    help="number of network targets (e.g. num-pdf-ids/num-leaves)")
parser.add_argument("config_dir",
                    help="Directory to write config files and variables");

## config for convolution layer ##
parser.add_argument("--conv-filter-dim", type=int,
                    help="The filt-x-dim used in convolution component.", default=250);
parser.add_argument("--conv-num-filters", type=int,
                    help="The number of filters used in convolution component.", default=100);
parser.add_argument("--conv-filter-step", type=int,
                    help="The filt-x-step used in convolution component.", default=10);
parser.add_argument("--max-shift", type=float, 
                    help="max shift used in ShiftInputComponent.", default=0.0);
parser.add_argument("--use-raw-wave", type=str,
                    help="if true use Convolution component in the 1st layer", default="true", choices=["false", "true"]);
parser.add_argument("--nonlin-type", type=int,
                    help="Type of nonlin used in the 1st layer", default=0);
parser.add_argument("--pnorm-block-dim", type=int,
                    help="block dimension for pnorm nonlinearity.", default=0);
parser.add_argument("--conv-jesus-hidden-dim", type=int,
                    help="hidden dimension of Jesus layer for convolution layer.", default=16000)
parser.add_argument("--conv-jesus-stddev-scale", type=float,
                    help="Scaling factor on parameter stddev of Jesus layer (smaller->jesus layer learns faster)",
                    default=1.0)
parser.add_argument("--conv-self-repair-scale", type=float,
                    help="Small scale involved in fixing derivatives, if supplied (e.g. try 0.00001)",
                    default=0.0)

parser.add_argument("--conv-use-shared-block", type=str,
                    help="If true RepeatedAffine applied as nonlinearity for convolution layer"
                     "otherwise BlockAffineComponent applied",
                    default="true", choices = ["false", "true"])
parser.add_argument("--conv-param-stddev-scale", type=float,
                    help="Scaling factor on parameter stddev of convolution layer.", default=1.0)

parser.add_argument("--conv-bias-stddev", type=float,
                    help="Scaling factor on bias stddev of convolution layer.", default=1.0)
parser.add_argument("--conv-num-nonlin", type=int,
                    help="number of nonlinearity layers for convolution in the network", default=2)
parser.add_argument("--add-log-stddev", type=str,
                    help="If ture add --add-log-stddev option to normalization layer.", 
                    default="false", choices= ["false", "true"]);
# options for adding ivector
parser.add_argument("--add-ivector-layer", type=int,
                    help="If true, the ivector is added to this layer.", default=3);
parser.add_argument("--ivector-output-dim", type=int,
                    help="the output dimension for separate affine component applied to ivector.", default=500);
parser.add_argument("--add-rbf-nonlin", type=str,
                    help="If true it adds radial basis nonlinearity to stats layer",
                    default="false", choices = ["false", "true"]);

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
if args.num_jesus_blocks < 1:
    sys.exit("invalid --num-jesus-blocks value");
if args.final_hidden_dim < 0:
    args.final_hidden_dim = args.jesus_forward_input_dim

for name in [ "jesus_hidden_dim", "jesus_forward_output_dim", "jesus_forward_input_dim",
              "final_hidden_dim" ]:
    old_val = getattr(args, name)
    if old_val % args.num_jesus_blocks != 0:
        new_val = old_val + args.num_jesus_blocks - (old_val % args.num_jesus_blocks)
        printable_name = '--' + name.replace('_', '-')
        print('Rounding up {0} from {1} to {2} to be a multiple of --num-jesus-blocks={3}: '.format(
                printable_name, old_val, new_val, args.num_jesus_blocks))
        setattr(args, name, new_val);

# this is a bit like a struct, initialized from a string, which describes how to
# set up the statistics-pooling and statistics-extraction components.
# An example string is 'mean(-99:3:9::99)', which means, compute the mean of
# data within a window of -99 to +99, with distinct means computed every 9 frames
# (we round to get the appropriate one), and with the input extracted on multiples
# of 3 frames (so this will force the input to this layer to be evaluated
# every 3 frames).  Another example string is 'mean+stddev(-99:3:9:99)',
# which will also cause the standard deviation to be computed; or
# 'mean+stddev+count(-99:3:9:99)'.
class StatisticsConfig:
    # e.g. c = StatisticsConfig('mean+stddev(-99:3:9:99)', 400, 100, 'jesus1-output-affine')
    # e.g. c = StatisticsConfig('mean+stddev+count(-99:3:9:99)', 400, 100, 'jesus1-output-affine')
    def __init__(self, config_string, input_dim, num_jesus_blocks, input_name):
        self.input_dim = input_dim
        self.num_jesus_blocks = num_jesus_blocks  # we need to know this because
                                                  # it's the dimension of the count
                                                  # features that we output.
        self.input_name = input_name

        m = re.search("mean(|\+stddev)(|\+count)\((-?\d+):(-?\d+):(-?\d+):(-?\d+)\)",
                      config_string)
        if m == None:
            sys.exit("Invalid splice-index or statistics-config string: " + config_string)
        self.output_stddev = (m.group(1) == '+stddev')
        self.output_count = (m.group(2) == '+count')

        self.left_context = -int(m.group(3))
        self.input_period = int(m.group(4))
        self.stats_period = int(m.group(5))
        self.right_context = int(m.group(6))
        if not (self.left_context >= 0 and self.right_context >= 0 and
                self.input_period > 0 and self.stats_period > 0 and
                self.num_jesus_blocks > 0 and
                self.left_context % self.stats_period == 0 and
                self.right_context % self.stats_period == 0 and
                self.stats_period % self.input_period == 0):
            sys.exit("Invalid configuration of statistics-extraction: " + config_string)

    # OutputDim() returns the output dimension of the node that this produces.
    def OutputDim(self):
        return self.input_dim * (2 if self.output_stddev else 1) + (self.num_jesus_blocks if self.output_count else 0)

    # OutputDims() returns an array of output dimensions... this node produces
    # one output node, but this array explains how it's split up into different types
    # of output (which will affect how we reorder the indexes for the jesus-layer).
    def OutputDims(self):
        ans = [ self.input_dim ]
        if self.output_stddev:
            ans.append(self.input_dim)
        if self.output_count:
            ans.append(self.num_jesus_blocks)
        return ans

    # Descriptor() returns the textual form of the descriptor by which the
    # output of this node is to be accessed.
    def Descriptor(self):
        return 'Round({0}-pooling-{1}-{2}, {3})'.format(self.input_name, self.left_context,
                                                        self.right_context, self.stats_period)

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
              'input-period={4} left-context={1} right-context={2} num-log-count-features={6} '
              'output-stddevs={5} '.format(self.input_name, self.left_context, self.right_context,
                                           stats_dim, self.stats_period,
                                           ('true' if self.output_stddev else 'false'),
                                           (self.num_jesus_blocks if self.output_count else 0)),
              file=f)
        print('component-node name={0}-pooling-{1}-{2} component={0}-pooling-{1}-{2} input={0}-extraction-{1}-{2} '.format(
                self.input_name, self.left_context, self.right_context), file=f)


####

## Work out splice_array
## e.g. for
## args.splice_indexes == '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'
## we would have
##   splice_array = [ [ -3,-2,...3 ], [-3,0] [-3,0] [-6,-3,0]


splice_array = []
left_context = 0
right_context = 0
split_on_spaces = args.splice_indexes.split(" ");  # we already checked the string is nonempty.
if len(split_on_spaces) < 2:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
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
            except:
                if len(splice_array) == 1:
                    sys.exit("First dimension of splicing array must not have averaging [yet]")
                try:
                    x = StatisticsConfig(s, 100, 100, 'foo')
                except:
                    if re.match("skip(-?\d+)$", s) == None:
                        sys.exit("The following element of the splicing array is not a valid specifier "
                                 "of statistics or of the form skipDDD: " + s)

        if leftmost_splice == 10000 or rightmost_splice == -10000:
            sys.exit("invalid element of --splice-indexes: " + string)
        left_context += -leftmost_splice
        right_context += rightmost_splice
except ValueError as e:
    sys.exit("invalid --splice-indexes argument " + args.splice_indexes + " " + str(e))
left_context = max(0, left_context)
right_context = max(0, right_context)
num_hidden_layers = len(splice_array)
input_dim = len(splice_array[0]) * args.feat_dim

f = open(args.config_dir + "/vars", "w")
print('left_context=' + str(left_context), file=f)
print('right_context=' + str(right_context), file=f)
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
f.close()


for l in range(1, num_hidden_layers + 1):
    # the following summarizes the structure of the layers:  Here, the Jesus component includes ReLU at its input and output, and renormalize
    #   at its output after the ReLU.
    # layer1: splice + LDA-transform + affine + ReLU + renormalize
    # layerX: splice + Jesus + affine + ReLU

    # Inside the jesus component is:
    #  [permute +] ReLU + repeated-affine + ReLU + repeated-affine
    # [we make the repeated-affine the last one so we don't have to redo that in backprop].
    # We follow this with a post-jesus composite component containing the operations:
    #  [permute +] ReLU + renormalize
    # call this post-jesusN.
    # After this we use dim-range nodes to split up the output into
    # [ jesusN-forward-output, jesusN-direct-output and jesusN-projected-output ]
    # parts;
    # and nodes for the jesusN-forward-affine.
    add_log_stddev_dim = 0
    use_stats = "false"
    if args.add_log_stddev == "true":
      add_log_stddev_dim = 1
    f = open(args.config_dir + "/layer{0}.config".format(l), "w")
    print('# Config file for layer {0} of the network'.format(l), file=f)
    if l == 1:
        if (args.use_raw_wave != "true"):
            print('component name=lda type=FixedAffineComponent matrix={0}/lda.mat'.
                  format(args.config_dir), file=f)
            splices = [ ('Offset(input, {0})'.format(n) if n != 0 else 'input') for n in splice_array[l-1] ]
            if args.add_ivector_layer == l:
              if args.ivector_dim > 0: 
                splices.append('ReplaceIndex(ivector, t, 0)')
                input_dim = len(splice_array[0]) * args.feat_dim  +  args.ivector_dim

            orig_input='Append({0})'.format(', '.join(splices))
            # e.g. orig_input = 'Append(Offset(input, -2), ... Offset(input, 2), ivector)'
            print('component-node name=lda component=lda input={0}'.format(orig_input),
                  file=f)
            # after the initial LDA transform, put a trainable affine layer and a ReLU, followed
            # by a NormalizeComponent.
            print('component name=affine1 type=NaturalGradientAffineComponent '
                  'input-dim={0} output-dim={1} bias-stddev=0'.format(
                    input_dim, args.jesus_forward_input_dim), file=f)
            print('component-node name=affine1 component=affine1 input=lda',
                  file=f)
            # the ReLU after the affine
            print('component name=relu1 type=RectifiedLinearComponent dim={1} self-repair-scale={2}'.format(
                    l, args.jesus_forward_input_dim, args.self_repair_scale), file=f)
            print('component-node name=relu1 component=relu1 input=affine1', file=f)
            # the renormalize component after the ReLU
            print ('component name=renorm1 type=NormalizeComponent dim={0} add-log-stddev={1}'.format(
                    args.jesus_forward_input_dim + add_log_stddev_dim, args.add_log_stddev), file=f)
            print('component-node name=renorm1 component=renorm1 input=relu1', file=f)
            cur_output = 'renorm1'
            cur_affine_output_dim = args.jesus_forward_input_dim
        else:
            # As input to layer after convolution component in 1st layer, we append the output
            # of previous layer to speed up computation and we splice floor(filter-dim/frame-input-dim)
            # frames together as input of convolution componenent in 1st layer and 
            # then we append output of convolution layer together.
            conv_num_inputs = int(args.conv_filter_dim / args.feat_dim + 1);
            conv_input_dim = conv_num_inputs * args.feat_dim;
            first_layer_split = [ int(x) for x in split_on_spaces[0].split(",")]

            conv_splice_in = []
            for i in range(first_layer_split[-1] - conv_num_inputs + 1, first_layer_split[-1]+1):
              conv_splice_in.append(i)

            conv_splice_out = []
            for i in range(-1 * (len(first_layer_split) - conv_num_inputs), 1):
              conv_splice_out.append(i);

            conv_in_list=[('Offset(input, {0})'.format(n) if n !=0 else 'input' ) for n in conv_splice_in ]
            conv_out_list=[('Offset(conv1, {0})'.format(n) if n !=0 else 'conv1' ) for n in conv_splice_out ]


            conv_out_len = (conv_input_dim - args.conv_filter_dim) / args.conv_filter_step + 1
            conv_column_map = []
            for x in range(0, args.conv_num_filters):
              for y in range(0, conv_out_len * len(conv_splice_out)):
                conv_column_map.append(y * args.conv_num_filters + x)
            if args.max_shift != 0.0:
              conv_in_list.append('Offset(input, {0})'.format(first_layer_split[-1] + 1))
              print("component name=shift1 type=ShiftInputComponent input-dim={0} output-dim={1} max-shift={2}".format(conv_input_dim + args.feat_dim, conv_input_dim, args.max_shift), file=f)
              print("component-node name=shift1 component=shift1 input=Append({0})".format(", ".join(conv_in_list)), file=f)
              print("component name=conv1 type=ConvolutionComponent input-x-dim={0} input-y-dim=1 input-z-dim=1 filt-x-dim={1} filt-y-dim=1 filt-x-step={2} filt-y-step=1 num-filters={3} input-vectorization-order=zyx param-stddev={4} bias-stddev={5}".format(conv_input_dim, args.conv_filter_dim, args.conv_filter_step, args.conv_num_filters, 
              args.conv_param_stddev_scale / math.sqrt(conv_input_dim), args.conv_bias_stddev), file=f)
              print("component-node name=conv1 component=conv1 input=shift1", file=f)
            else:
              print("component name=conv1 type=ConvolutionComponent input-x-dim={0} input-y-dim=1 input-z-dim=1 filt-x-dim={1} filt-y-dim=1 filt-x-step={2}, filt-y-step=1 num-filters={3} input-vectorization-order=zyx".format(conv_input_dim, args.conv_filter_dim, args.conv_filter_step, args.conv_num_filters), file=f)
              print("component-node name=conv1 component=conv1 input=Append({0})".format(", ".join(conv_in_list)), file=f)

            print("component name=permute1 type=PermuteComponent column-map={0}".format(",".join([str(x) for x in conv_column_map])), file=f)
            print("component-node name=permute1 component=permute1 input=Append({0})".format(", ".join(conv_out_list)), file=f)
            
            # nonlin_type: 0 for pnorm and 1 for jesus nonlinearity.
            nonlin_comp_name='pnorm1'
            pnorm_block_dim = args.pnorm_block_dim
            if pnorm_block_dim == 0:
              pnorm_block_dim = conv_out_len * len(conv_out_list) 
            nonlin_input_dim = args.conv_num_filters * conv_out_len * len(conv_out_list)
            nonlin_output_dim = nonlin_input_dim / pnorm_block_dim
            if args.nonlin_type == 0:
              print("component name=pnorm1 type=PnormComponent input-dim={0} output-dim={1}".format(nonlin_input_dim, nonlin_output_dim), file=f)
              print("component-node name=pnorm1 component=pnorm1 input=permute1", file=f)
            else:
              nonlin_comp_name='jesus1'
              need_input_permute_component = "false";
              num_conv_sub_components = 4
              if args.conv_use_shared_block == "true":
                comp_type="NaturalGradientRepeatedAffineComponent"
              else:
                comp_type="BlockAffineComponent"
              print("component name=jesus1 type=CompositeComponent num-components={0}".format(num_conv_sub_components), file=f)
              print(" component{0}='type={3} dim={1} self-repair-scale={2}'".format(1, 
                      nonlin_input_dim, args.conv_self_repair_scale, comp_type), file=f, end='')
              if args.conv_use_shared_block == "true":
                print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2}"
                      "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                        2,
                        nonlin_input_dim, args.conv_jesus_hidden_dim,
                        args.conv_num_filters,
                        args.conv_jesus_stddev_scale / math.sqrt(nonlin_input_dim / args.conv_num_filters),
                        0.5 * args.conv_jesus_stddev_scale, comp_type),
                      file=f, end='')
              else:
                print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2}"
                      "num-blocks={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                        2,
                        nonlin_input_dim, args.conv_jesus_hidden_dim,
                        args.conv_num_filters,
                        args.conv_jesus_stddev_scale / math.sqrt(nonlin_input_dim / args.conv_num_filters),
                        0.5 * args.conv_jesus_stddev_scale, comp_type),
                      file=f, end='')
              print(" component{0}='type=RectifiedLinearComponent dim={1}'".format(3,
                      args.conv_jesus_hidden_dim, args.conv_self_repair_scale), file=f, end='')


              if args.conv_use_shared_block == "true": 
                print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2}"
                      "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(4,
                        args.conv_jesus_hidden_dim,
                        nonlin_output_dim,
                        args.conv_num_filters,
                        args.jesus_stddev_scale / math.sqrt(args.conv_jesus_hidden_dim / args.conv_num_filters),
                        0.5 * args.jesus_stddev_scale),
                      file=f, end='')
              else:
                print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2}"
                      "num-blocks={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(4,
                        args.conv_jesus_hidden_dim,
                        nonlin_output_dim,
                        args.conv_num_filters,
                        args.jesus_stddev_scale / math.sqrt(args.conv_jesus_hidden_dim / args.conv_num_filters),
                        0.5 * args.jesus_stddev_scale),
                      file=f, end='')
               
              print("", file=f) # print newline. 
              print("component-node name=jesus1 component=jesus1 input=permute1", file=f)
            
            print("component name=log1 type=LogComponent dim={0}".format(nonlin_output_dim), file=f)
            print("component-node name=log1 component=log1 input={0}".format(nonlin_comp_name), file=f)
        cur_output = 'log1'
        cur_affine_output_dim = nonlin_output_dim


    else:
        use_repeat = args.use_repeated_affine
        num_blocks = args.num_jesus_blocks 
        if l <= (1 + args.conv_num_nonlin):
          use_repeat = args.conv_use_shared_block
          num_blocks = args.conv_num_filters

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
                # it's not an integer offset, so assume it either specifies the
                # statistics-extraction, or is of the form skipXX where XX is an
                # integer offset (this takes as input the previous post-jesus layer).
                m = re.match("skip(-?\d+)$", s)
                if m != None:
                    if l <= 2:
                        sys.exit("You cannot use skip-splicing for the 1st 2 layers")
                    offset = m.group(1)
                    splices.append("Offset(post-jesus{0}, {1})".format(l-1, offset))
                    spliced_dims.append(args.jesus_forward_output_dim)
                else:
                    stats = StatisticsConfig(s, cur_affine_output_dim, 
                                             num_blocks, cur_output)
                    stats.WriteConfigs(f)
                    use_stats = "true"
                    splices.append(stats.Descriptor())
                    spliced_dims.extend(stats.OutputDims())
                    if (args.add_rbf_nonlin == "true"):
                      splices.append("rbf-nonlin{0}".format(l))
                      spliced_dims.append(cur_affine_output_dim)

        # get the input to the Jesus layer.
        cur_input = 'Append({0})'.format(', '.join(splices))
        cur_dim = sum(spliced_dims)

        this_jesus_output_dim = args.jesus_forward_output_dim

        # As input to the Jesus component we'll append the spliced input and any
        # mean/stddev-stats input, and the first thing inside the component that
        # we do is rearrange the dimensions so that things pertaining to a
        # particular block stay together.

        column_map = []
        for x in range(0, num_blocks):
            dim_offset = 0
            for src_splice in spliced_dims:
                src_block_size = src_splice / num_blocks
                for y in range(0, src_block_size):
                    column_map.append(dim_offset + (x * src_block_size) + y)
                dim_offset += src_splice
        if sorted(column_map) != range(0, sum(spliced_dims)):
            print("column_map len is " + str(len(column_map)))
            print(" column_map - spliced dim range = " + str([x1 - x2 for (x1, x2) in zip(column_map, range(0, sum(spliced_dims)))]))
            print("column_map is " + str(column_map))
            print("num_jesus_blocks is " + str(num_blocks))
            print("spliced_dims is " + str(spliced_dims))
            sys.exit("code error creating new column order")

        need_input_permute_component = (column_map != range(0, sum(spliced_dims)))

        # Now add the jesus component.
        if use_stats == "true":
          if args.add_rbf_nonlin == "true":
            rbf_input = 'Append(jesus{0}-forward-output-affine, Round(jesus{0}-forward-output-affine-pooling-99-99, 9))'.format(l-1)
            print("component name=rbf-nonlin{0} type=RbfComponent dim={1}".format(l, 3*cur_affine_output_dim), file=f, end='')
            print("", file=f) # print newline.
            print("component-node name=rbf-nonlin{0} component=rbf-nonlin{0} input={1}".format(l, rbf_input), file=f, end='')
            print("", file=f) # print newline.

        num_sub_components = (5 if need_input_permute_component else 4);
        print('component name=jesus{0} type=CompositeComponent num-components={1}'.format(
                l, num_sub_components), file=f, end='')
        # print the sub-components of the CompositeComopnent on the same line.
        # this CompositeComponent has the same effect as a sequence of
        # components, but saves memory.
        if need_input_permute_component:
            print(" component1='type=PermuteComponent column-map={1}'".format(
                    l, ','.join([str(x) for x in column_map])), file=f, end='')
        print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
                (2 if need_input_permute_component else 1),
                cur_dim, args.self_repair_scale), file=f, end='')

        if use_repeat == "true":
            print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
                  "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                    (3 if need_input_permute_component else 2),
                    cur_dim, args.jesus_hidden_dim,
                    num_blocks,
                    args.jesus_stddev_scale / math.sqrt(cur_dim / num_blocks),
                    0.5 * args.jesus_stddev_scale),
                  file=f, end='')
        else:
            print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
                  "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                    (3 if need_input_permute_component else 2),
                    cur_dim, args.jesus_hidden_dim,
                    num_blocks,
                    args.jesus_stddev_scale / math.sqrt(cur_dim / num_blocks)),
                  file=f, end='')


        print(" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
                (4 if need_input_permute_component else 3),
                args.jesus_hidden_dim, args.self_repair_scale), file=f, end='')



        if use_repeat == "true":
            print(" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
                  "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                    (5 if need_input_permute_component else 4),
                    args.jesus_hidden_dim,
                    this_jesus_output_dim,
                    num_blocks,
                    args.jesus_stddev_scale / math.sqrt(args.jesus_hidden_dim / num_blocks),
                    0.5 * args.jesus_stddev_scale),
                  file=f, end='')
        else:
            print(" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
                  "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                    (5 if need_input_permute_component else 4),
                    args.jesus_hidden_dim,
                    this_jesus_output_dim,
                    num_blocks,
                    args.jesus_stddev_scale / math.sqrt((args.jesus_hidden_dim / num_blocks))),
                  file=f, end='')

        print("", file=f) # print newline.
        print('component-node name=jesus{0} component=jesus{0} input={1}'.format(
                l, cur_input), file=f)

        # now print the post-Jesus component which consists of ReLU +
        # renormalize.

        num_sub_components = 2
        print('component name=post-jesus{0} type=CompositeComponent num-components=2 '.format(l),
              file=f, end='')

        # still within the post-Jesus component, print the ReLU
        print(" component1='type=RectifiedLinearComponent dim={0} self-repair-scale={1}'".format(
                this_jesus_output_dim, args.self_repair_scale), file=f, end='')
        # still within the post-Jesus component, print the NormalizeComponent
        print(" component2='type=NormalizeComponent dim={0} add-log-stddev={1} '".format(
                this_jesus_output_dim, args.add_log_stddev), file=f, end='')
        print("", file=f) # print newline.
        print('component-node name=post-jesus{0} component=post-jesus{0} input=jesus{0}'.format(l),
              file=f)

        # handle the forward output, we need an affine node for this:
        cur_affine_output_dim = (args.jesus_forward_input_dim if l < num_hidden_layers else args.final_hidden_dim)
        cur_affine_input_dim = args.jesus_forward_output_dim + add_log_stddev_dim
        if (args.add_ivector_layer == l):
          if args.ivector_dim > 0:
            cur_affine_input_dim = args.jesus_forward_output_dim + add_log_stddev_dim + args.ivector_output_dim + add_log_stddev_dim
            print('component name=forward-ivector type=NaturalGradientAffineComponent input-dim={0} output-dim={1} bias-stddev=0'.format(args.ivector_dim, args.ivector_output_dim), file=f)
            print('component-node name=forward-ivector component=forward-ivector input=ReplaceIndex(ivector, t, 0)', file=f)
            print("component name=post-ivector type=CompositeComponent num-components=2 " 
                  "component1='type=RectifiedLinearComponent dim={0} self-repair-scale={1}' " 
                  "component2='type=NormalizeComponent dim={0} add-log-stddev={2} '".format(args.ivector_output_dim, args.self_repair_scale, args.add_log_stddev), file=f)
            print("component-node name=post-ivector component=post-ivector input=forward-ivector ", file=f)
            print('component name=forward-affine{0} type=NaturalGradientAffineComponent '
                  'input-dim={1} output-dim={2} bias-stddev=0'.format(l, cur_affine_input_dim, cur_affine_output_dim), file=f)
            print('component-node name=jesus{0}-forward-output-affine component=forward-affine{0} input=Append(post-jesus{0}, post-ivector)'.format(
                l), file=f)
        else:
          print('component name=forward-affine{0} type=NaturalGradientAffineComponent '
                'input-dim={1} output-dim={2} bias-stddev=0'.format(l, cur_affine_input_dim, cur_affine_output_dim), file=f)
          print('component-node name=jesus{0}-forward-output-affine component=forward-affine{0} input=post-jesus{0}'.format(
              l), file=f)
        # for each recurrence delay, create an affine node followed by a
        # clip-gradient node.  [if there are multiple recurrences in the same layer,
        # each one gets its own affine projection.]

        # The reason we set the param-stddev to 0 is out of concern that if we
        # initialize to nonzero, this will encourage the corresponding inputs at
        # the jesus layer to become small (to remove this random input), which
        # in turn will make this component learn slowly (due to small
        # derivatives).  we set the bias-mean to 0.001 so that the ReLUs on the
        # input of the Jesus layer are in the part of the activation that has a
        # nonzero derivative- otherwise with this setup it would never learn.

        cur_output = 'jesus{0}-forward-output-affine'.format(l)


    # with each new layer we regenerate the final-affine component, with a ReLU before it
    # because the layers we printed don't end with a nonlinearity.
    final_affine_input='log1'
    if (args.use_raw_wave == "false" or l != 1):
      final_affine_input='final-relu'
      print('component name=final-relu type=RectifiedLinearComponent dim={0} self-repair-scale={1}'.format(
              cur_affine_output_dim, args.self_repair_scale), file=f)
      print('component-node name=final-relu component=final-relu input={0}'.format(cur_output),
            file=f)
    print('component name=final-affine type=NaturalGradientAffineComponent '
          'input-dim={0} output-dim={1} learning-rate-factor={2} param-stddev=0.0 bias-stddev=0'.format(
            cur_affine_output_dim, args.num_targets,
            args.final_layer_learning_rate_factor), file=f)
    print('component-node name=final-affine component=final-affine input={0}'.format(final_affine_input),
          file=f)
    # printing out the next two, and their component-nodes, for l > 1 is not
    # really necessary as they will already exist, but it doesn't hurt and makes
    # the structure clearer.
    if args.include_log_softmax == "true":
        print('component name=final-log-softmax type=LogSoftmaxComponent dim={0}'.format(
                args.num_targets), file=f)
        print('component-node name=final-log-softmax component=final-log-softmax '
              'input=final-affine', file=f)
        print('output-node name=output input=final-log-softmax', file=f)
    else:
        print('output-node name=output input=final-affine', file=f)

    if args.xent_regularize != 0.0:
        # This block prints the configs for a separate output that will be
        # trained with a cross-entropy objective in the 'chain' models... this
        # has the effect of regularizing the hidden parts of the model.  we use
        # 0.5 / args.xent_regularize as the learning rate factor- the factor of
        # 1.0 / args.xent_regularize is suitable as it means the xent
        # final-layer learns at a rate independent of the regularization
        # constant; and the 0.5 was tuned so as to make the relative progress
        # similar in the xent and regular final layers.
        print('component name=final-affine-xent type=NaturalGradientAffineComponent '
              'input-dim={0} output-dim={1} param-stddev=0.0 bias-stddev=0 learning-rate-factor={2}'.format(
                cur_affine_output_dim, args.num_targets, 0.5 / args.xent_regularize), file=f)
        print('component-node name=final-affine-xent component=final-affine-xent input={0}'.format(final_affine_input),
              file=f)
        print('component name=final-log-softmax-xent type=LogSoftmaxComponent dim={0}'.format(
                args.num_targets), file=f)
        print('component-node name=final-log-softmax-xent component=final-log-softmax-xent '
              'input=final-affine-xent', file=f)
        print('output-node name=output-xent input=final-log-softmax-xent', file=f)

    f.close()
