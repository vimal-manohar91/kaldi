#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import shlex
import sys
import warnings
import copy
import imp
import ast
import math, re

nodes = imp.load_source('', 'steps/nnet3/components.py')
nnet3_train_lib = imp.load_source('ntl', 'steps/nnet3/nnet3_train_lib.py')
chain_lib = imp.load_source('ncl', 'steps/nnet3/chain/nnet3_chain_lib.py')

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

    def AddLayer(self, config_lines, name):
        components = config_lines['components']
        component_nodes = config_lines['component-nodes']

        components.append('component name={0}-extraction-{1}-{2} type=StatisticsExtractionComponent input-dim={3} '
              'input-period={4} output-period={5} include-variance={6} '.format(
                self.input_name, self.left_context, self.right_context,
                self.input_dim, self.input_period, self.stats_period,
                ('true' if self.output_stddev else 'false')))
        component_nodes.append('component-node name={0}-extraction-{1}-{2} component={0}-extraction-{1}-{2} input={0} '.format(
                self.input_name, self.left_context, self.right_context))
        stats_dim = 1 + self.input_dim * (2 if self.output_stddev else 1)
        component.append('component name={0}-pooling-{1}-{2} type=StatisticsPoolingComponent input-dim={3} '
              'input-period={4} left-context={1} right-context={2} num-log-count-features={6} '
              'output-stddevs={5} '.format(self.input_name, self.left_context, self.right_context,
                                           stats_dim, self.stats_period,
                                           ('true' if self.output_stddev else 'false'),
                                           (self.num_jesus_blocks if self.output_count else 0)))
        component_nodes.append('component-node name={0}-pooling-{1}-{2} component={0}-pooling-{1}-{2} input={0}-extraction-{1}-{2} '.format(
                self.input_name, self.left_context, self.right_context))

def GetArgs():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(description="Writes config files and variables "
                                                 "for TDNNs creation and training",
                                     epilog="See steps/nnet3/tdnn/train.sh for example.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter
                                     )

    # Only one of these arguments can be specified, and one of them has to
    # be compulsarily specified
    feat_group = parser.add_mutually_exclusive_group(required = True)
    feat_group.add_argument("--feat-dim", type=int,
                            help="Raw feature dimension, e.g. 13")
    feat_group.add_argument("--feat-dir", type=str,
                            help="Feature directory, from which we derive the feat-dim")

    # only one of these arguments can be specified
    ivector_group = parser.add_mutually_exclusive_group(required = False)
    ivector_group.add_argument("--ivector-dim", type=int,
                                help="iVector dimension, e.g. 100", default=0)
    ivector_group.add_argument("--ivector-dir", type=str,
                                help="iVector dir, which will be used to derive the ivector-dim  ", default=None)

    num_target_group = parser.add_mutually_exclusive_group(required = True)
    num_target_group.add_argument("--num-targets", type=int,
                                  help="number of network targets (e.g. num-pdf-ids/num-leaves)")
    num_target_group.add_argument("--ali-dir", type=str,
                                  help="alignment directory, from which we derive the num-targets")
    num_target_group.add_argument("--tree-dir", type=str,
                                  help="directory with final.mdl, from which we derive the num-targets")

    # CNN options
    parser.add_argument('--cnn.layer', type=str, action='append', dest = "cnn_layer",
                        help="CNN parameters at each CNN layer, e.g. --filt-x-dim=3 --filt-y-dim=8 "
                        "--filt-x-step=1 --filt-y-step=1 --num-filters=256 --pool-x-size=1 --pool-y-size=3 "
                        "--pool-z-size=1 --pool-x-step=1 --pool-y-step=3 --pool-z-step=1, "
                        "when CNN layers are used, no LDA will be added", default = None)
    parser.add_argument("--cnn.bottleneck-dim", type=int, dest = "cnn_bottleneck_dim",
                        help="Output dimension of the linear layer at the CNN output "
                        "for dimension reduction, e.g. 256."
                        "The default zero means this layer is not needed.", default=0)
    parser.add_argument("--cnn.cepstral-lifter", type=float, dest = "cepstral_lifter",
                        help="The factor used for determining the liftering vector in the production of MFCC. "
                        "User has to ensure that it matches the lifter used in MFCC generation, "
                        "e.g. 22.0", default=22.0)
    parser.add_argument("--cnn.param-stddev-scale", dest = "cnn_param_stddev_scale",
                        help="Scaling factor on parameter stddev of convolution layer.", default=1.0)
    parser.add_argument("--cnn.param-bias-stddev", type=float,
                        help="Scaling factor on bias stddev of convolution layer.", default=1.0)


    # General neural network options
    parser.add_argument("--splice-indexes", type=str, required = True,
                        help="Splice indexes at each layer, e.g. '-3,-2,-1,0,1,2,3' "
                        "If CNN layers are used the first set of splice indexes will be used as input "
                        "to the first CNN layer and later splice indexes will be interpreted as indexes "
                        "for the TDNNs.")
    parser.add_argument("--add-lda", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="If \"true\" an LDA matrix computed from the input features "
                        "(spliced according to the first set of splice-indexes) will be used as "
                        "the first Affine layer. This affine layer's parameters are fixed during training. "
                        "If --cnn.layer is specified this option will be forced to \"false\".",
                        default=True, choices = ["false", "true"])

    parser.add_argument("--include-log-softmax", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="add the final softmax layer ", default=True, choices = ["false", "true"])
    parser.add_argument("--add-final-sigmoid", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="add a final sigmoid layer as alternate to log-softmax-layer. "
                        "Can only be used if include-log-softmax is false. "
                        "This is useful in cases where you want the output to be "
                        "like probabilities between 0 and 1. Typically the nnet "
                        "is trained with an objective such as quadratic",
                        default=False, choices = ["false", "true"])

    parser.add_argument("--objective-type", type=str,
                        help = "the type of objective; i.e. quadratic or linear",
                        default="linear", choices = ["linear", "quadratic", "xent"])
    parser.add_argument("--xent-regularize", type=float,
                        help="For chain models, if nonzero, add a separate output for cross-entropy "
                        "regularization (with learning-rate-factor equal to the inverse of this)",
                        default=0.0)
    parser.add_argument("--xent-separate-forward-affine", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="if using --xent-regularize, gives it separate last-but-one weight matrix",
                        default=False, choices = ["false", "true"])
    parser.add_argument("--final-layer-normalize-target", type=float,
                        help="RMS target for final layer (set to <1 if final layer learns too fast",
                        default=1.0)
    parser.add_argument("--subset-dim", type=int, default=0,
                        help="dimension of the subset of units to be sent to the central frame")

    # Non-linearity options
    parser.add_argument("--pnorm-input-dim", type=int,
                        help="input dimension to p-norm nonlinearities")
    parser.add_argument("--pnorm-output-dim", type=int,
                        help="output dimension of p-norm nonlinearities")

    parser.add_argument("--relu-dim", type=int,
                        help="dimension of ReLU nonlinearities")

    # Jesus layer options
    parser.add_argument("--jesus.layer", dest = "jesus_layer", type = str,
                        help="paramters of jesus layer ")

    parser.add_argument("--self-repair-scale", type=float,
                        help="A non-zero value activates the self-repair mechanism in the sigmoid and tanh non-linearities of the LSTM", default=None)

    parser.add_argument("--use-presoftmax-prior-scale", type=str, action=nnet3_train_lib.StrToBoolAction,
                        help="if true, a presoftmax-prior-scale is added",
                        choices=['true', 'false'], default = True)

    parser.add_argument("--feat-type", type=str,
                        default = "mfcc", choices = ["mfcc", "fbank", "waveform"],
                        help="feature type used at input")

    parser.add_argument("--feature-extraction.pooling-type",
                        dest = "feat_extract_pooling_type", type=str,
                        choices = ["pnorm", "jesus", "none"], default = "jesus",
                        help="type of pooling used in feature extraction block")
    parser.add_argument("--feature-extraction.pnorm-block-dim",
                        dest = "feat_extract_pnorm_block_dim", type=int,
                        default = 0,
                        help="block dimension for pnorm pooling")
    parser.add_argument("--feature-extraction.jesus-layer",
                        dest = "feat_extract_jesus_layer", type=str,
                        help="parameters of jesus layer in feature extraction block")
    parser.add_argument("--feature-extraction.self-repair-scale",
                        dest = "feat_extract_self_repair_scale",
                        help="Small scale involved in fixing derivatives, if supplied (e.g. try 0.00001)", default = 0.0)
    parser.add_argument("--feature-extraction.conv-filter-dim", type=int, dest = 'conv_filter_dim',
                        help="The filt-x-dim used in convolution component.", default=250);
    parser.add_argument("--feature-extraction.conv-num-filters", type=int, dest = 'conv_num_filters',
                        help="The number of filters used in convolution component.", default=100);
    parser.add_argument("--feature-extraction.conv-filter-step", type=int, dest = 'conv_filter_step',
                        help="The filt-x-step used in convolution component.", default=10);
    parser.add_argument("--feature-extraction.conv-param-stddev-scale", type=float, dest = 'conv_param_stddev_scale',
                        help="Scaling factor on parameter stddev of convolution layer.", default=1.0)
    parser.add_argument("--feature-extraction-conv-bias-stddev", type=float, dest = 'conv_bias_stddev',
                        help="Scaling factor on bias stddev of convolution layer.", default=1.0)
    parser.add_argument("--feature-extraction.num-hidden-layers", type=int, dest = 'feat_extract_num_hidden_layers',
                        help="number of hidden layers in feature extraction block, excluding the convolution layer", default=2)
    parser.add_argument("--feature-extraction.max-shift", type=float, dest = 'max_shift',
                        help="max shift used in ShiftInputComponent.", default=0.0);

    parser.add_argument("--ivector-scale", type=float, default = 1.0,
                        help = "Scale ivector before adding to input")

    parser.add_argument("config_dir",
                        help="Directory to write config files and variables")

    print(' '.join(sys.argv), file = sys.stderr)

    args = parser.parse_args()
    args = CheckArgs(args)

    return args

def CheckArgs(args):
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    ## Check arguments.
    if args.feat_dir is not None:
        args.feat_dim = nnet3_train_lib.GetFeatDim(args.feat_dir)

    if args.ali_dir is not None:
        args.num_targets = nnet3_train_lib.GetNumberOfLeaves(args.ali_dir)
    elif args.tree_dir is not None:
        args.num_targets = chain_lib.GetNumberOfLeaves(args.tree_dir)

    if args.ivector_dir is not None:
        args.ivector_dim = nnet3_train_lib.GetIvectorDim(args.ivector_dir)

    if not args.feat_dim > 0:
        raise Exception("feat-dim has to be postive")

    if not args.num_targets > 0:
        print(args.num_targets)
        raise Exception("num_targets has to be positive")

    if not args.ivector_dim >= 0:
        raise Exception("ivector-dim has to be non-negative")

    if (args.subset_dim < 0):
        raise Exception("--subset-dim has to be non-negative")

    if (sum([ 1 for y in [ any(x is not None for x in [args.pnorm_input_dim, args.pnorm_output_dim]),
        args.relu_dim is not None, args.jesus_layer is not None ] if y ]) != 1):
        raise Exception("--relu-dim argument, "
                        "{--pnorm-input-dim, --pnorm-output-dim}, "
                        "{--jesus.hidden-dim, --jesus.forward-output-dim, --jesus.forward-input-dim, --jesus.num-blocks} "
                        "are mutually exclusive")

    args.jesus_config = None
    if args.relu_dim is not None:
        args.nonlin_input_dim = args.relu_dim
        args.nonlin_output_dim = args.relu_dim
    elif args.pnorm_input_dim is not None:
        args.nonlin_input_dim = args.pnorm_input_dim
        args.nonlin_output_dim = args.pnorm_output_dim
    else:
        args.nonlin_input_dim = None
        args.nonlin_output_dim = None
        args.jesus_config = ParseJesusString(args.jesus_layer)

    if args.add_final_sigmoid and args.include_log_softmax:
        raise Exception("--include-log-softmax and --add-final-sigmoid cannot both be true.")

    if args.xent_separate_forward_affine and args.add_final_sigmoid:
        raise Exception("It does not make sense to have --add-final-sigmoid=true when xent-separate-forward-affine is true")

    if args.add_lda and args.cnn_layer is not None:
        args.add_lda = False
        warnings.warn("--add-lda is set to false as CNN layers are used.")
    if args.add_lda and args.feat_type == "waveform":
        args.add_lda = False
        warnings.warn("--add-lda is set to false as feat-type is waveform")

    if args.add_lda:
        args.ivector_scale = 1.0
        warnings.warn("--ivector-scale is set to 1.0 as add-lda is True")

    args.feat_extract_config = argparse.Namespace()
    args.feat_extract_config.pooling_type = args.feat_extract_pooling_type
    args.feat_extract_config.pnorm_block_dim = args.feat_extract_pnorm_block_dim
    args.feat_extract_config.self_repair_scale = args.feat_extract_self_repair_scale
    args.feat_extract_config.conv_filter_dim = args.conv_filter_dim
    args.feat_extract_config.conv_filter_step = args.conv_filter_step
    args.feat_extract_config.conv_num_filters = args.conv_num_filters
    args.feat_extract_config.max_shift = args.max_shift
    args.feat_extract_config.num_hidden_layers = args.feat_extract_num_hidden_layers
    args.feat_extract_config.conv_param_stddev_scale = args.conv_param_stddev_scale
    args.feat_extract_config.conv_bias_stddev = args.conv_bias_stddev
    args.feat_extract_config.jesus_config = None

    if args.feat_extract_config.pooling_type == "jesus":
        if args.feat_extract_jesus_layer is None and args.jesus_config is None:
            raise Exception("feature-extraction.pooling-type is jesus; but jesus-layers is not specified")
        if args.feat_extract_jesus_layer is None:
            args.feat_extract_config.jesus_config = args.jesus_config
        else:
            args.feat_extract_config.jesus_config = ParseJesusString(args.feat_extract_jesus_layer)

    return args

def AddRelNormLayer(config_lines, name, input, norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, input['dimension'], self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, input['dimension'], norm_target_rms))
    component_nodes.append("component-node name={0}_relu component={0}_relu input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))

    return {'descriptor': '{0}_renorm'.format(name),
            'dimension': input['dimension']}

def AddPnormLayer(config_lines, name, input, pnorm_block_dim):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    output_dim = input['dimension'] / pnorm_block_dim;

    components.append("component name={0}_pnorm type=PnormComponent input-dim={1} output-dim={2}".format(name, input['dimension'], output_dim))
    component_nodes.append("component-node name={0}_pnorm component={0}_pnorm input={1}".format(name, input['descriptor']))

    return {'descriptor': '{0}_pnorm'.format(name),
            'dimension': output_dim}

def AddLogLayer(config_lines, name, input):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    components.append("component name={0}_log type=LogComponent dim={1}".format(name, input['dimension']))
    component_nodes.append("component-node name={0}_log component={0}_log input={1}".format(name, input['descriptor']))

    return {'descriptor': '{0}_log'.format(name),
            'dimension': input['dimension']}

def AddRelNormAffLayer(config_lines, name, input, output_dim, ng_affine_options = " bias-stddev=0 ", norm_target_rms = 1.0, self_repair_scale = None):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    self_repair_string = "self-repair-scale={0:.10f}".format(self_repair_scale) if self_repair_scale is not None else ''
    components.append("component name={0}_relu type=RectifiedLinearComponent dim={1} {2}".format(name, input['dimension'], self_repair_string))
    components.append("component name={0}_renorm type=NormalizeComponent dim={1} target-rms={2}".format(name, input['dimension'], norm_target_rms))
    components.append("component name={0}_affine type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input['dimension'], output_dim, ng_affine_options))

    component_nodes.append("component-node name={0}_relu component={0}_relu input={1}".format(name, input['descriptor']))
    component_nodes.append("component-node name={0}_renorm component={0}_renorm input={0}_relu".format(name))
    component_nodes.append("component-node name={0}_affine component={0}_affine input={0}_renorm".format(name))

    return {'descriptor':  '{0}_affine'.format(name),
            'dimension': output_dim}

def AddJesusLayer(config_lines, name, input, jesus_config, norm_target_rms = 1.0):
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    # get the input to the Jesus layer.
    cur_input = input['descriptor']
    cur_dim = input['dimension']
    spliced_dims = input['appended-dimensions']

    this_jesus_output_dim = jesus_config.forward_output_dim
    num_blocks = jesus_config.num_blocks

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
        error_string = "Exception: column_map len is {0:d}\n".format(len(column_map))
        error_string += "column_map - spliced dim range = {0}\n".format(str([x1 - x2 for (x1, x2) in zip(column_map, range(0, sum(spliced_dims)))]))
        error_string += "column_map is {0}\n".format(str(column_map))
        error_string += "num_jesus_blocks is {0:d}\n".format(num_blocks)
        error_string += "spliced_dims is {0}\n".format(str(spliced_dims))
        error_string += "code error creating new column order\n"
        raise Exception(error_string)

    need_input_permute_component = (column_map != range(0, sum(spliced_dims)))

    # Now add the jesus component.
    num_sub_components = (5 if need_input_permute_component else 4);
    jesus_component = 'component name={0}_jesus type=CompositeComponent num-components={1}'.format(
            name, num_sub_components)

    # print the sub-components of the CompositeComopnent on the same line.
    # this CompositeComponent has the same effect as a sequence of
    # components, but saves memory.
    if need_input_permute_component:
        jesus_component += " component1='type=PermuteComponent column-map={0}'".format(','.join([str(x) for x in column_map]))
    jesus_component += " component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
            (2 if need_input_permute_component else 1),
            cur_dim, jesus_config.self_repair_scale)

    if jesus_config.use_repeated_affine:
        jesus_component += (" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
              "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                (3 if need_input_permute_component else 2),
                cur_dim, jesus_config.hidden_dim,
                num_blocks,
                jesus_config.stddev_scale / math.sqrt(cur_dim / num_blocks),
                0.5 * jesus_config.stddev_scale))
    else:
        jesus_component += (" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
              "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                (3 if need_input_permute_component else 2),
                cur_dim, jesus_config.hidden_dim,
                num_blocks,
                jesus_config.stddev_scale / math.sqrt(cur_dim / num_blocks)))

    jesus_component += (" component{0}='type=RectifiedLinearComponent dim={1} self-repair-scale={2}'".format(
            (4 if need_input_permute_component else 3),
            jesus_config.hidden_dim, jesus_config.self_repair_scale))

    if jesus_config.use_repeated_affine:
        jesus_component += (" component{0}='type=NaturalGradientRepeatedAffineComponent input-dim={1} output-dim={2} "
              "num-repeats={3} param-stddev={4} bias-mean={5} bias-stddev=0'".format(
                (5 if need_input_permute_component else 4),
                jesus_config.hidden_dim,
                this_jesus_output_dim,
                num_blocks,
                jesus_config.stddev_scale / math.sqrt(jesus_config.hidden_dim / num_blocks),
                0.5 * jesus_config.stddev_scale))
    else:
        jesus_component += (" component{0}='type=BlockAffineComponent input-dim={1} output-dim={2} "
              "num-blocks={3} param-stddev={4} bias-stddev=0'".format(
                (5 if need_input_permute_component else 4),
                jesus_config.hidden_dim,
                this_jesus_output_dim,
                num_blocks,
                jesus_config.stddev_scale / math.sqrt((jesus_config.hidden_dim / num_blocks))))

    components.append(jesus_component)
    component_nodes.append('component-node name={0}_jesus component={0}_jesus input={1}'.format(
            name, cur_input))

    # now print the post-Jesus component which consists of ReLU +
    # renormalize.
    num_sub_components = 2
    post_jesus_component = 'component name={0}_post-jesus type=CompositeComponent num-components=2'.format(name)

    # still within the post-Jesus component, print the ReLU
    post_jesus_component += (" component1='type=RectifiedLinearComponent dim={0} self-repair-scale={1}'".format(
            this_jesus_output_dim, jesus_config.self_repair_scale))
    # still within the post-Jesus component, print the NormalizeComponent
    post_jesus_component += (" component2='type=NormalizeComponent dim={0}'".format(
                              this_jesus_output_dim))

    components.append(post_jesus_component)
    component_nodes.append('component-node name={0}_post-jesus component={0}_post-jesus input={0}_jesus'.format(name))

    return {'descriptor': '{0}_post-jesus'.format(name),
            'dimension': this_jesus_output_dim,
             'appended-dimensions': [this_jesus_output_dim]}

def AddPerDimAffineLayer(config_lines, name, input, input_window):
    filter_context = int((input_window - 1) / 2)
    filter_input_splice_indexes = range(-1 * filter_context, filter_context + 1)
    list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in filter_input_splice_indexes]
    filter_input_descriptor = 'Append({0})'.format(' , '.join(list))
    filter_input_descriptor = {'descriptor':filter_input_descriptor,
                               'dimension':len(filter_input_splice_indexes) * input['dimension']}


    # add permute component to shuffle the feature columns of the Append
    # descriptor output so that columns corresponding to the same feature index
    # are contiguous add a block-affine component to collapse all the feature
    # indexes across time steps into a single value
    num_feats = input['dimension']
    num_times = len(filter_input_splice_indexes)
    column_map = []
    for i in range(num_feats):
        for j in range(num_times):
            column_map.append(j * num_feats + i)
    permuted_output_descriptor = nodes.AddPermuteLayer(config_lines,
            name, filter_input_descriptor, column_map)

    # add a block-affine component
    output_descriptor = nodes.AddBlockAffineLayer(config_lines, name,
                                                  permuted_output_descriptor,
                                                  num_feats, num_feats)

    return [output_descriptor, filter_context, filter_context]

def AddMultiDimAffineLayer(config_lines, name, input, input_window, block_input_dim, block_output_dim):
    assert(block_input_dim % input_window== 0)
    filter_context = int((input_window - 1) / 2)
    filter_input_splice_indexes = range(-1 * filter_context, filter_context + 1)
    list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in filter_input_splice_indexes]
    filter_input_descriptor = 'Append({0})'.format(' , '.join(list))
    filter_input_descriptor = {'descriptor':filter_input_descriptor,
                               'dimension':len(filter_input_splice_indexes) * input['dimension']}


    # add permute component to shuffle the feature columns of the Append
    # descriptor output so that columns corresponding to the same feature index
    # are contiguous add a block-affine component to collapse all the feature
    # indexes across time steps into a single value
    num_feats = input['dimension']
    num_times = len(filter_input_splice_indexes)
    column_map = []
    for i in range(num_feats):
        for j in range(num_times):
            column_map.append(j * num_feats + i)
    permuted_output_descriptor = nodes.AddPermuteLayer(config_lines,
            name, filter_input_descriptor, column_map)
    # add a block-affine component
    output_descriptor = nodes.AddBlockAffineLayer(config_lines, name,
                                                  permuted_output_descriptor,
                                                  num_feats / (block_input_dim / input_window) * block_output_dim, num_feats / (block_input_dim/ input_window))

    return [output_descriptor, filter_context, filter_context]

def AddLpFilter(config_lines, name, input, rate, num_lpfilter_taps, lpfilt_filename, is_updatable = False):
    try:
        import scipy.signal as signal
        import numpy as np
    except ImportError:
        raise Exception(" This recipe cannot be run without scipy."
                        " You can install it using the command \n"
                        " pip install scipy\n"
                        " If you do not have admin access on the machine you are"
                        " trying to run this recipe, you can try using"
                        " virtualenv")
    # low-pass smoothing of input was specified. so we will add a low-pass filtering layer
    lp_filter = signal.firwin(num_lpfilter_taps, rate, width=None, window='hamming', pass_zero=True, scale=True, nyq=1.0)
    lp_filter = list(np.append(lp_filter, 0))
    nnet3_train_lib.WriteKaldiMatrix(lpfilt_filename, [lp_filter])
    filter_context = int((num_lpfilter_taps - 1) / 2)
    filter_input_splice_indexes = range(-1 * filter_context, filter_context + 1)
    list = [('Offset({0}, {1})'.format(input['descriptor'], n) if n != 0 else input['descriptor']) for n in filter_input_splice_indexes]
    filter_input_descriptor = 'Append({0})'.format(' , '.join(list))
    filter_input_descriptor = {'descriptor':filter_input_descriptor,
                               'dimension':len(filter_input_splice_indexes) * input['dimension']}

    input_x_dim = len(filter_input_splice_indexes)
    input_y_dim = input['dimension']
    input_z_dim = 1
    filt_x_dim = len(filter_input_splice_indexes)
    filt_y_dim = 1
    filt_x_step = 1
    filt_y_step = 1
    input_vectorization = 'zyx'

    tdnn_input_descriptor = nodes.AddConvolutionLayer(config_lines, name,
                                                     filter_input_descriptor,
                                                     input_x_dim, input_y_dim, input_z_dim,
                                                     filt_x_dim, filt_y_dim,
                                                     filt_x_step, filt_y_step,
                                                     1, input_vectorization,
                                                     filter_bias_file = lpfilt_filename,
                                                     is_updatable = is_updatable)


    return [tdnn_input_descriptor, filter_context, filter_context]

def AddConvMaxpLayer(config_lines, name, input, args):
    if '3d-dim' not in input:
        raise Exception("The input to AddConvMaxpLayer() needs '3d-dim' parameters.")

    input = nodes.AddConvolutionLayer(config_lines, name, input,
                              input['3d-dim'][0], input['3d-dim'][1], input['3d-dim'][2],
                              args.filt_x_dim, args.filt_y_dim,
                              args.filt_x_step, args.filt_y_step,
                              args.num_filters, input['vectorization'])

    if args.pool_x_size > 1 or args.pool_y_size > 1 or args.pool_z_size > 1:
      input = nodes.AddMaxpoolingLayer(config_lines, name, input,
                                input['3d-dim'][0], input['3d-dim'][1], input['3d-dim'][2],
                                args.pool_x_size, args.pool_y_size, args.pool_z_size,
                                args.pool_x_step, args.pool_y_step, args.pool_z_step)

    return input

# The ivectors are processed through an affine layer parallel to the CNN layers,
# then concatenated with the CNN output and passed to the deeper part of the network.
def AddCnnLayers(config_lines, cnn_layer, cnn_bottleneck_dim, cepstral_lifter, config_dir, feat_dim, splice_indexes=[0], ivector_dim=0, ivector_scale = 1.0, add_idct = True):
    cnn_args = ParseCnnString(cnn_layer)
    num_cnn_layers = len(cnn_args)
    # We use an Idct layer here to convert MFCC to FBANK features

    prev_layer_output = {'descriptor':  "input",
                         'dimension': feat_dim}

    if add_idct:
        nnet3_train_lib.WriteIdctMatrix(feat_dim, cepstral_lifter, config_dir.strip() + "/idct.mat")
        prev_layer_output = nodes.AddFixedAffineLayer(config_lines, "Idct", prev_layer_output, config_dir.strip() + '/idct.mat')

    list = [('Offset({0}, {1})'.format(prev_layer_output['descriptor'],n) if n != 0 else prev_layer_output['descriptor']) for n in splice_indexes]
    splice_descriptor = "Append({0})".format(", ".join(list))
    cnn_input_dim = len(splice_indexes) * feat_dim
    prev_layer_output = {'descriptor':  splice_descriptor,
                         'dimension': cnn_input_dim,
                         '3d-dim': [len(splice_indexes), feat_dim, 1],
                         'vectorization': 'yzx'}

    for cl in range(0, num_cnn_layers):
        prev_layer_output = AddConvMaxpLayer(config_lines, "L{0}".format(cl), prev_layer_output, cnn_args[cl])

    prev_layer_output['appended-dimensions'] = [ prev_layer_output['3d-dim'][2] for x in
            range(0, prev_layer_output['3d-dim'][0] * prev_layer_output['3d-dim'][1]) ]

    if cnn_bottleneck_dim > 0:
        prev_layer_output = nodes.AddAffineLayer(config_lines, "cnn-bottleneck", prev_layer_output, cnn_bottleneck_dim, "")
        prev_layer_output['appended-dimensions'] = [prev_layer_output['dimension']]

    if ivector_dim > 0:
        iv_layer_output = {'descriptor':  'ReplaceIndex(ivector, t, 0)',
                           'dimension': ivector_dim}
        iv_layer_output = nodes.AddAffineLayer(config_lines, "ivector", iv_layer_output, ivector_dim, "")

        if ivector_scale != 1.0:
            components.append('component name={0}-fixed-scale type=FixedScaleComponent dim={1} scale={2}'.format("ivector", ivector_dim, ivector_scale))
            component_nodes.append('component-node name={0}-fixed-scale component={0}-fixed-scale input={1}'.format("ivector",
                iv_layer_output['descriptor']))
            iv_layer_output['descriptor'] = "ivector-fixed-scale"

        prev_layer_output['appended-dimensions'].append(ivector_dim)


        prev_layer_output['descriptor'] = 'Append({0}, {1})'.format(prev_layer_output['descriptor'], iv_layer_output['descriptor'])
        prev_layer_output['dimension'] = prev_layer_output['dimension'] + iv_layer_output['dimension']

    return prev_layer_output

def AddFeatureExtractionBlock(config_lines, name, feat_extract_config, feat_dim, splice_indexes=[0]):
    # As input to layer after convolution component in 1st layer, we append the output
    # of previous layer to speed up computation and we splice floor(filter-dim/frame-input-dim)
    # frames together as input of convolution componenent in 1st layer and
    # then we append output of convolution layer together.
    components = config_lines['components']
    component_nodes = config_lines['component-nodes']

    conv_num_inputs = int(feat_extract_config.conv_filter_dim / feat_dim + 1);
    conv_input_dim = conv_num_inputs * feat_dim;
    first_layer_split = splice_indexes

    conv_splice_in = []
    for i in range(first_layer_split[-1] - conv_num_inputs + 1, first_layer_split[-1]+1):
        conv_splice_in.append(i)

    conv_in_list=[('Offset(input, {0})'.format(n) if n !=0 else 'input' ) for n in conv_splice_in ]

    prev_layer_output = {'descriptor': " ".join(conv_in_list),
                         'dimension': conv_input_dim}

    if feat_extract_config.max_shift != 0.0:
        conv_in_list.append('Offset(input, {0})'.format(first_layer_split[-1] + 1))

        components.append('component name={0}_shift type=ShiftInputComponent input-dim={1} output-dim={2} max-shift={3}'.format(name, conv_input_dim + feat_dim, conv_input_dim, feat_extract_config.max_shift))
        component_nodes.append('component-node name={0}_shift component={0}_shift input=Append({1})'.format(name, ", ".join(conv_in_list)))

        prev_layer_output['descriptor'] = '{0}_shift'.format(name)

    conv_output = nodes.AddConvolutionLayer(config_lines, name,
                        prev_layer_output,
                        input_x_dim = prev_layer_output['dimension'],
                        input_y_dim = 1, input_z_dim = 1,
                        filt_x_dim = feat_extract_config.conv_filter_dim,
                        filt_y_dim = 1,
                        filt_x_step = feat_extract_config.conv_filter_step,
                        filt_y_step = 1,
                        num_filters = feat_extract_config.conv_num_filters,
                        input_vectorization = "zyx",
                        param_stddev = feat_extract_config.conv_param_stddev_scale / math.sqrt(conv_input_dim),
                        bias_stddev = feat_extract_config.conv_bias_stddev)

    conv_splice_out = []
    for i in range(-1 * (len(first_layer_split) - conv_num_inputs), 1):
        conv_splice_out.append(i);

    conv_out_list=[('Offset({0}, {1})'.format(conv_output['descriptor'], n) if n != 0 else conv_output['descriptor']) for n in conv_splice_out ]
    prev_layer_output['descriptor'] = "Append({0})".format(", ".join(conv_out_list))
    prev_layer_output['dimension'] = len(conv_out_list) * conv_output['dimension']

    conv_out_len = (conv_input_dim - feat_extract_config.conv_filter_dim) / feat_extract_config.conv_filter_step + 1
    conv_column_map = []
    for x in range(0, feat_extract_config.conv_num_filters):
        for y in range(0, prev_layer_output['dimension'] / feat_extract_config.conv_num_filters):
            conv_column_map.append(y * feat_extract_config.conv_num_filters + x)

    components.append("component name={0}_permute type=PermuteComponent column-map={1}".format(name, ",".join([str(x) for x in conv_column_map])))
    component_nodes.append("component-node name={0}_permute component={0}_permute input={1}".format(name, prev_layer_output['descriptor']))

    prev_layer_output['descriptor'] = '{0}_permute'.format(name)

    if feat_extract_config.pooling_type == "pnorm":

        prev_layer_output = AddPnormLayer(config_lines, name,
                        prev_layer_output, feat_extract_config.pnorm_block_dim)
    else:
        prev_layer_output = AddJesusLayer(config_lines, name,
                                prev_layer_output,
                                feat_extract_config.jesus_config)

    prev_layer_output = AddPnormLayer(config_lines, "L{0}-feat-extract".format(0), prev_layer_output, 1)
    prev_layer_output = AddLogLayer(config_lines, "L{0}-feat-extract".format(0), prev_layer_output)

    prev_layer_output['appended-dimensions'] = [prev_layer_output['dimension']]

    return prev_layer_output

def PrintConfig(file_name, config_lines):
    f = open(file_name, 'w')
    print("", file = f)
    for line in config_lines['components']:
        print(line, file = f)
    print("", file = f)
    print("#Component nodes", file = f)
    print("", file = f)
    for line in config_lines['component-nodes']:
        print(line, file = f)
    print("", file = f)
    f.close()

def ParseCnnString(cnn_param_string_list):
    cnn_parser = argparse.ArgumentParser(description="cnn argument parser")

    cnn_parser.add_argument("--filt-x-dim", required=True, type=int)
    cnn_parser.add_argument("--filt-y-dim", required=True, type=int)
    cnn_parser.add_argument("--filt-x-step", type=int, default = 1)
    cnn_parser.add_argument("--filt-y-step", type=int, default = 1)
    cnn_parser.add_argument("--num-filters", required=True, type=int)
    cnn_parser.add_argument("--pool-x-size", type=int, default = 1)
    cnn_parser.add_argument("--pool-y-size", type=int, default = 1)
    cnn_parser.add_argument("--pool-z-size", type=int, default = 1)
    cnn_parser.add_argument("--pool-x-step", type=int, default = 1)
    cnn_parser.add_argument("--pool-y-step", type=int, default = 1)
    cnn_parser.add_argument("--pool-z-step", type=int, default = 1)

    cnn_args = []
    for cl in range(0, len(cnn_param_string_list)):
         cnn_args.append(cnn_parser.parse_args(shlex.split(cnn_param_string_list[cl])))

    return cnn_args

def ParseJesusString(jesus_param_string):
    jesus_parser = argparse.ArgumentParser(description = "jesus argument parser")

    jesus_parser.add_argument("--hidden-dim", type=int, required = True,
                              help="hidden dimension of Jesus layer.")
    jesus_parser.add_argument("--forward-output-dim", type=int, required = True,
                              help="part of output dimension of Jesus layer that goes to next layer")
    jesus_parser.add_argument("--forward-input-dim", type=int, required = True,
                              help="Input dimension of Jesus layer that comes from affine projection "
                              "from the previous layer (same as output dim of forward affine transform)")
    jesus_parser.add_argument("--num-blocks", type=int, required = True,
                              help="number of blocks in Jesus layer.  All configs of the form "
                              "--jesus.*-dim will be rounded up to be a multiple of this.")
    jesus_parser.add_argument("--stddev-scale", type=float,
                              help="Scaling factor on parameter stddev of Jesus layer (smaller->jesus layer learns faster)",
                              default=1.0)
    jesus_parser.add_argument("--use-repeated-affine", type=str,
                              action = nnet3_train_lib.StrToBoolAction, default = True,
                              help="if true use RepeatedAffineComponent, else BlockAffineComponent (i.e. no sharing)")
    jesus_parser.add_argument("--self-repair-scale",
                              help="Small scale involved in fixing derivatives, if supplied (e.g. try 0.00001)", default = 0.0)
    jesus_parser.add_argument("--final-hidden-dim", dest = "final_hidden_dim", type=int,
                              help="Final hidden layer dimension-- or if <0, the same as "
                              "--forward-input-dim", default=-1)


    jesus_config = jesus_parser.parse_args(shlex.split(jesus_param_string))

    for name in [ "hidden_dim", "forward_output_dim", "forward_input_dim", "final_hidden_dim" ]:
        old_val = getattr(jesus_config, name)
        if ((name != "final_hidden_dim" or (name == "final_hidden_dim" and old_val != -1))
            and  old_val % jesus_config.num_blocks != 0):
            new_val = old_val + jesus_config.num_blocks - (old_val % jesus_config.num_blocks)
            printable_name = '--' + name.replace('_', '-')
            print('Rounding up {0} from {1} to {2} to be a multiple of --num-jesus-blocks={3}: '.format(
                    printable_name, old_val, new_val, jesus_config.num_blocks))
            setattr(jesus_config, name, new_val);

    if jesus_config.final_hidden_dim == -1:
        jesus_config.final_hidden_dim = jesus_config.forward_output_dim

    return jesus_config

## Work out splice_array
## e.g. for
## args.splice_indexes == '-3,-2,-1,0,1,2,3 -3,0:-3 -3,0:-3 -6,-3,0:-6,-3'
## we would have
##   splice_array = [ [ -3,-2,...3 ], [-3,0] [-3,0] [-6,-3,0]

def ParseSpliceString(splice_indexes):
    splice_array = []
    left_context = 0
    right_context = 0
    split_on_spaces = splice_indexes.split(" ");  # we already checked the string is nonempty.

    if len(split_on_spaces) < 1:
        raise Exception("invalid splice-indexes argument, too short: "
                        + splice_indexes)
    try:
        for string in split_on_spaces:
            this_layer_len = len(splice_array)
            this_splices = string.split(",")

            # the rest of this block updates left_context and right_context, and
            # does some checking.
            leftmost_splice = 10000
            rightmost_splice = -10000

            if len(this_splices) < 1:
                raise Exception("invalid splice-indexes argument, too-short element: "
                                + splice_indexes)

            int_list = []
            all_list = []
            for s in this_splices:
                try:
                    n = int(s)
                    int_list.append(n)
                    all_list.append(n)
                    if n < leftmost_splice:
                        leftmost_splice = n
                    if n > rightmost_splice:
                        rightmost_splice = n
                except:
                    if len(splice_array) == 1:
                        raise Exception("First dimension of splicing array must not have averaging [yet]")
                    try:
                        x = StatisticsConfig(s, 100, 100, 'foo')
                        all_list.append(s)
                    except:
                        if re.match("skip(-?\d+)$", s) is None:
                            raise Exception("The following element of the splicing array is not a valid specifier "
                                 "of statistics or of the form skipDDD: " + s)

            if leftmost_splice == 10000 or rightmost_splice == -10000:
                raise Exception("invalid element of --splices-indexes: " + string)
            left_context += -leftmost_splice
            right_context += rightmost_splice

            if int_list != sorted(int_list):
                raise Exception("elements of splice-indexes must be sorted: "
                                + splice_indexes)
            splice_array.append(all_list)

    except ValueError as e:
        raise Exception("invalid splice-indexes argument " + splice_indexes + e)
    left_context = max(0, left_context)
    right_context = max(0, right_context)

    return {'left_context':left_context,
            'right_context':right_context,
            'splice_indexes':splice_array,
            'num_hidden_layers':len(splice_array)
            }

# The function signature of MakeConfigs is changed frequently as it is intended for local use in this script.
def MakeConfigs(config_dir, splice_indexes_string,
                cnn_layer, cnn_bottleneck_dim, cepstral_lifter,
                feat_dim, ivector_dim, ivector_scale, num_targets, add_lda,
                feat_type, feat_extract_config, jesus_config,
                nonlin_input_dim, nonlin_output_dim, subset_dim,
                use_presoftmax_prior_scale,
                final_layer_normalize_target,
                include_log_softmax,
                add_final_sigmoid,
                xent_regularize,
                xent_separate_forward_affine,
                self_repair_scale,
                objective_type):

    parsed_splice_output = ParseSpliceString(splice_indexes_string.strip())

    left_context = parsed_splice_output['left_context']
    right_context = parsed_splice_output['right_context']
    num_hidden_layers = parsed_splice_output['num_hidden_layers']
    splice_indexes = parsed_splice_output['splice_indexes']
    input_dim = len(parsed_splice_output['splice_indexes'][0]) + feat_dim + ivector_dim

    if xent_separate_forward_affine:
        if splice_indexes[-1] != [0]:
            raise Exception("--xent-separate-forward-affine option is supported only if the last-hidden layer has no splicing before it. Please use a splice-indexes with just 0 as the final splicing config.")

    prior_scale_file = '{0}/presoftmax_prior_scale.vec'.format(config_dir)

    config_lines = {'components':[], 'component-nodes':[]}

    config_files={}
    add_idct = not add_lda and feat_type == "mfcc" and feat_extract_config.pooling_type == "jesus"

    if add_idct:
        nnet3_train_lib.WriteIdctMatrix(feat_dim, cepstral_lifter, config_dir.strip() + "/idct.mat")

    prev_layer_output = nodes.AddInputLayer(config_lines, feat_dim, splice_indexes[0], ivector_dim, config_dir.strip() + "/idct.mat" if add_idct else None)

    # Add the init config lines for estimating the preconditioning matrices
    init_config_lines = copy.deepcopy(config_lines)
    init_config_lines['components'].insert(0, '# Config file for initializing neural network prior to')
    init_config_lines['components'].insert(0, '# preconditioning matrix computation')
    nodes.AddOutputLayer(init_config_lines, prev_layer_output)
    config_files[config_dir + '/init.config'] = init_config_lines

    # Add the feature extraction block and also the first layer in the case of waveform
    if feat_type != "waveform":
        if cnn_layer is not None:
            prev_layer_output = AddCnnLayers(config_lines, cnn_layer, cnn_bottleneck_dim, cepstral_lifter, config_dir,
                                             feat_dim, splice_indexes[0], ivector_dim, True if feat_type == "mfcc" else False)
        else:
            if add_lda:
                prev_layer_output = nodes.AddLdaLayer(config_lines, "L0", prev_layer_output, config_dir + '/lda.mat')
                # There is an LDA; so no point in doing permute
                prev_layer_output['appended-dimensions'] = [prev_layer_output['dimension']]
            else:
                prev_layer_output['appended-dimensions'] = [feat_dim for x in range(0, len(splice_indexes[0]))] + ([ivector_dim] if ivector_dim > 0 else [])

        if feat_extract_config.pooling_type != "jesus":
            prev_layer_output = nodes.AddAffineLayer(config_lines, "L0", prev_layer_output, nonlin_output_dim)
            # There is an affine component; so no point in doing permute
            prev_layer_output['appended-dimensions'] = [prev_layer_output['dimension']]

    else:
        # For raw waveform, we always have a feature extraction block
        prev_layer_output = AddFeatureExtractionBlock(config_lines, "L0-feat-extract", feat_extract_config,
                                                      feat_dim)
        #appended_dimensions = [feat_dim for x in range(0, len(splice_indexes[0]))] + ([ivector_dim] if ivector_dim > 0 else [])

        if (num_hidden_layers < feat_extract_config.num_hidden_layers or
                feat_extract_config.num_hidden_layers < 1):
            raise Exception("num-hidden-layers must be larger than --feature-extraction.num-hidden-layers "
                            "and --feature-extraction.num-hidden-layers must be larger than 0")

    # we moved the first splice layer to before the LDA..
    # so the input to the first affine layer is going to [0] index
    splice_indexes[0] = [0]

    for i in range(0, num_hidden_layers):
        config_lines['components'].append("# Components for layer {0}".format(i))

        # make the intermediate config file for layerwise discriminative training
        # if specified, pool the input from the previous layer

        #######################################################################
        # prepare the spliced input
        #######################################################################
        prev_prev_layer_output = prev_layer_output

        if i > 0:
            prev_layer_output['appended-dimensions'] = []

        if (not (feat_type == "waveform" and i == 0) and
                not (len(splice_indexes[i]) == 1 and splice_indexes[i][0] == 0)):
            try:
                zero_index = splice_indexes[i].index(0)
            except ValueError:
                zero_index = None

            # I just assume the prev_layer_output_descriptor is a simple forwarding descriptor
            prev_layer_output_descriptor = prev_layer_output['descriptor']
            subset_output = prev_layer_output
            if subset_dim > 0:
                # if subset_dim is specified the script expects a zero in the splice indexes
                assert(zero_index is not None)
                subset_node_config = "dim-range-node name=Tdnn_input_{0} input-node={1} dim-offset={2} dim={3}".format(i, prev_layer_output_descriptor, 0, subset_dim)
                subset_output = {'descriptor' : 'Tdnn_input_{0}'.format(i),
                                 'dimension' : subset_dim}
                config_lines['component-nodes'].append(subset_node_config)

            appended_descriptors = []
            for j in range(len(splice_indexes[i])):
                if j == zero_index:
                    appended_descriptors.append(prev_layer_output['descriptor'])
                    prev_layer_output['appended-dimensions'].append(prev_layer_output['dimension'])
                    continue

                if type(splice_indexes[i][j]) is int:
                    appended_descriptors.append('Offset({0}, {1})'.format(subset_output['descriptor'], splice_indexes[i][j]))
                    prev_layer_output['appended-dimensions'].append(subset_output['dimension'])
                else:
                    m = re.match("skip(-?\d+)$", splice_indexes[i][j])
                    if m is not None:
                        if i <= 1:
                            raise Exception("You cannot use skip-splicing for the 1st 2 layers")
                        offset = m.group(1)
                        appended_descriptors.append('Offset({0}, {1})'.format(prev_prev_layer_output['descriptor'], offset))
                        prev_layer_output['appended-dimensions'].append(subset_output['dimension'])
                    else:
                        stats = StatisticsConfig(splice_indexes[i][j], subset_output['dimension'],
                                                 jesus_config.num_blocks if jesus_config is not None else 1,
                                                 subset_output['descriptor'])
                        stats.AddLayer(config_lines, name)
                        appended_descriptors.append(stats.Descriptor())
                        prev_layer_output['appended-dimensions'].extend(stats.OutputDims())
            prev_layer_output = {'descriptor' : "Append({0})".format(" , ".join(appended_descriptors)),
                                 'dimension'  : sum(prev_layer_output['appended-dimensions']),
                                 'appended-dimensions': prev_layer_output['appended-dimensions']}
        else:
            # this is a normal affine node
            if i > 0:
                prev_layer_output['appended-dimensions'].append(prev_layer_output['dimension'])
        #######################################################################

        #######################################################################
        # add hidden layers
        #######################################################################
        if xent_separate_forward_affine and i == num_hidden_layers - 1:
            # Final layers when xent forward regularization is used
            if xent_regularize == 0.0:
                raise Exception("xent-separate-forward-affine=True is valid only if xent-regularize is non-zero")

            if (feat_type != "waveform" or i != 0):
                final_layer_output = AddRelNormLayer(config_lines, "Tdnn_pre_final", prev_layer_output, self_repair_scale = self_repair_scale, norm_target_rms = 1.0)
            else:
                final_layer_output = prev_layer_output
            #else:
            #    final_layer_output = AddPnormLayer(config_lines, "L{0}-feat-extract".format(i), prev_layer_output, 1)
            #    final_layer_output = AddLogLayer(config_lines, "L{0}-feat-extract".format(i), final_layer_output)

            prev_layer_output_chain = nodes.AddAffRelNormLayer(config_lines, "Tdnn_pre_final_chain",
                                                    final_layer_output, nonlin_output_dim,
                                                    self_repair_scale = self_repair_scale,
                                                    norm_target_rms = final_layer_normalize_target)


            nodes.AddFinalLayer(config_lines, prev_layer_output_chain, num_targets,
                               use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                               prior_scale_file = prior_scale_file,
                               include_log_softmax = include_log_softmax)


            prev_layer_output_xent = nodes.AddAffRelNormLayer(config_lines, "Tdnn_pre_final_xent",
                                                    final_layer_output, nonlin_output_dim,
                                                    self_repair_scale = self_repair_scale,
                                                    norm_target_rms = final_layer_normalize_target)

            nodes.AddFinalLayer(config_lines, prev_layer_output_xent, num_targets,
                                ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                    0.5 / xent_regularize),
                                use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                prior_scale_file = prior_scale_file,
                                include_log_softmax = True,
                                name_affix = 'xent')
        else:
            if feat_type != "waveform" or i > 0:
                # Non-final layers
                # For waveform features, hidden layer 0 is added as part of the feature extraction block
                if ( (i < feat_extract_config.num_hidden_layers and feat_extract_config.jesus_config is None)
                    or (i >= feat_extract_config.num_hidden_layers and jesus_config is None) ):
                    prev_layer_output = AddRelNormAffLayer(config_lines, "Tdnn_{0}".format(i),
                                                           prev_layer_output, nonlin_output_dim,
                                                           self_repair_scale = self_repair_scale,
                                                           norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target)
                else:
                    if i < feat_extract_config.num_hidden_layers:
                        prev_layer_output = AddJesusLayer(config_lines, "L{0}-feat-extract".format(i),
                                            prev_layer_output,
                                            feat_extract_config.jesus_config,
                                            norm_target_rms = 1.0 if i < num_hidden_layers - 1 else final_layer_normalize_target)
                        prev_layer_output = nodes.AddAffineLayer(config_lines, "L{0}-jesus-forward-output".format(i),
                                                           prev_layer_output,
                                                           feat_extract_config.jesus_config.forward_input_dim)
                    else:
                        prev_layer_output = AddJesusLayer(config_lines, "L{0}".format(i),
                                                      prev_layer_output, jesus_config,
                                                      norm_target_rms = 1.0 if i < num_hidden_layers -1 else final_layer_normalize_target)

                        prev_layer_output = nodes.AddAffineLayer(config_lines, "L{0}-jesus-forward-output".format(i),
                                                           prev_layer_output,
                                                           jesus_config.forward_input_dim if i < num_hidden_layers - 1 else jesus_config.final_hidden_dim)

            if i == feat_extract_config.num_hidden_layers - 1 and ivector_dim > 0:
                iv_layer_output = {'descriptor':  'ReplaceIndex(ivector, t, 0)',
                                   'dimension': ivector_dim}
                iv_layer_output = nodes.AddAffRelNormLayer(config_lines, "ivector", iv_layer_output, ivector_dim, "")

                if ivector_scale != 1.0:
                    components = config_lines['components']
                    component_nodes = config_lines['component-nodes']
                    components.append('component name={0}-fixed-scale type=FixedScaleComponent dim={1} scale={2}'.format("ivector", ivector_dim, ivector_scale))
                    component_nodes.append('component-node name={0}-fixed-scale component={0}-fixed-scale input={1}'.format("ivector",
                                                                                            iv_layer_output['descriptor']))
                    iv_layer_output['descriptor'] = "ivector-fixed-scale"

                prev_layer_output['descriptor'] = 'Append({0}, {1})'.format(prev_layer_output['descriptor'], iv_layer_output['descriptor'])
                prev_layer_output['dimension'] = prev_layer_output['dimension'] + iv_layer_output['dimension']

            # with each new layer we regenerate the final-affine component, with a ReLU before it
            # because the layers we printed don't end with a nonlinearity.
            if (feat_type != "waveform" or i != 0):
                final_layer_output = AddRelNormLayer(config_lines, "Final", prev_layer_output, self_repair_scale = self_repair_scale, norm_target_rms = 1.0)
            else:
                final_layer_output = prev_layer_output

            # a final layer is added after each new layer as we are generating
            # configs for layer-wise discriminative training

            # add_final_sigmoid adds a sigmoid as a final layer as alternative
            # to log-softmax layer.
            # http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression#Softmax_Regression_vs._k_Binary_Classifiers
            # This is useful when you need the final outputs to be probabilities between 0 and 1.
            # Usually used with an objective-type such as "quadratic".
            # Applications are k-binary classification such Ideal Ratio Mask prediction.
            nodes.AddFinalLayer(config_lines, final_layer_output, num_targets,
                               use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                               prior_scale_file = prior_scale_file,
                               include_log_softmax = include_log_softmax,
                               add_final_sigmoid = add_final_sigmoid,
                               objective_type = objective_type)
            if xent_regularize != 0.0:
                nodes.AddFinalLayer(config_lines, final_layer_output, num_targets,
                                    ng_affine_options = " param-stddev=0 bias-stddev=0 learning-rate-factor={0} ".format(
                                          0.5 / xent_regularize),
                                    use_presoftmax_prior_scale = use_presoftmax_prior_scale,
                                    prior_scale_file = prior_scale_file,
                                    include_log_softmax = True,
                                    name_affix = 'xent')

        config_files['{0}/layer{1}.config'.format(config_dir, i+1)] = config_lines
        config_lines = {'components':[], 'component-nodes':[]}

    left_context = int(parsed_splice_output['left_context'])
    right_context = int(parsed_splice_output['right_context'])

    # write the files used by other scripts like steps/nnet3/get_egs.sh
    f = open(config_dir + "/vars", "w")
    print('model_left_context=' + str(left_context), file=f)
    print('model_right_context=' + str(right_context), file=f)
    print('num_hidden_layers=' + str(num_hidden_layers), file=f)
    print('num_targets=' + str(num_targets), file=f)
    print('add_lda=' + ('true' if add_lda else 'false'), file=f)
    print('include_log_softmax=' + ('true' if include_log_softmax else 'false'), file=f)
    print('objective_type=' + objective_type, file=f)
    f.close()

    # printing out the configs
    # init.config used to train lda-mllt train
    for key in config_files.keys():
        PrintConfig(key, config_files[key])

def Main():
    args = GetArgs()

    MakeConfigs(config_dir = args.config_dir,
                splice_indexes_string = args.splice_indexes,
                feat_dim = args.feat_dim, ivector_dim = args.ivector_dim,
                ivector_scale = args.ivector_scale,
                feat_type = args.feat_type,
                feat_extract_config = args.feat_extract_config,
                jesus_config = args.jesus_config,
                num_targets = args.num_targets,
                add_lda = args.add_lda,
                cnn_layer = args.cnn_layer,
                cnn_bottleneck_dim = args.cnn_bottleneck_dim,
                cepstral_lifter = args.cepstral_lifter,
                nonlin_input_dim = args.nonlin_input_dim,
                nonlin_output_dim = args.nonlin_output_dim,
                subset_dim = args.subset_dim,
                use_presoftmax_prior_scale = args.use_presoftmax_prior_scale,
                final_layer_normalize_target = args.final_layer_normalize_target,
                include_log_softmax = args.include_log_softmax,
                add_final_sigmoid = args.add_final_sigmoid,
                xent_regularize = args.xent_regularize,
                xent_separate_forward_affine = args.xent_separate_forward_affine,
                self_repair_scale = args.self_repair_scale,
                objective_type = args.objective_type)

if __name__ == "__main__":
    Main()

