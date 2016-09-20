#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

import numpy as np

def ComputeLifterCoeffs(Q, dim):
    coeffs = np.zeros((dim))
    for i in range(0, dim):
        coeffs[i] = 1.0 + 0.5 * Q * math.sin(math.pi * i / Q);

    return coeffs

def ComputeIDctMatrix(K, N, cepstral_lifter=0):
    matrix = np.zeros((K, N))
    # normalizer for X_0
    normalizer = math.sqrt(1.0 / N);
    for j in range(0, N):
        matrix[0, j] = normalizer;
    # normalizer for other elements
    normalizer = math.sqrt(2.0 / N);
    for k in range(1, K):
      for n in range(0, N):
        matrix[k, n] = normalizer * math.cos(math.pi/N * (n + 0.5) * k);

    if cepstral_lifter != 0:
        lifter_coeffs = ComputeLifterCoeffs(cepstral_lifter, K)
        for k in range(0, K):
            matrix[k, :] = matrix[k, :] / lifter_coeffs[k];

    return matrix.T

parser = argparse.ArgumentParser(description="Writes config files and variables "
                                 "for TDNNs creation and training",
                                 epilog="See steps/nnet3/train_tdnn.sh for example.");
parser.add_argument("--cnn-indexes", type=str,
                    help="CNN indexes at each CNN layer, e.g. '3,8,1,1,256,1,3,1,1,3,1'")
parser.add_argument("--cnn-reduced-dim", type=int,
                    help="Output dimension of the linear layer at the CNN output "
                    "for dimension reduction, e.g. 256", default=256)
parser.add_argument("--add-delta", type=str,
                    help="add a convolution layer at the front to generate delta and delta-delta "
                    "to the rest of the network ", default="false", choices = ["false", "true"])
parser.add_argument("--use-mfcc", type=str,
                    help="use MFCC features as the CNN input, if true, an IDCT matrix "
                    "is added to convert them to FBANK", default="true", choices = ["false", "true"])
parser.add_argument("--cepstral-lifter", type=float,
                    help="Here we need the scaling factor on cepstra in the production of MFCC"
                    "to cancel out the effect of lifter, e.g. 22.0", default=22.0)
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
if args.cnn_indexes is None:
    sys.exit("--cnn-indexes argument is required");
if args.cnn_reduced_dim is None or not (args.cnn_reduced_dim > 0):
    sys.exit("--cnn-reduced-dim argument is required");
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
split1 = args.splice_indexes.split();  # we already checked the string is nonempty.
input_dim = args.feat_dim + args.ivector_dim

if len(split1) < 1:
    sys.exit("invalid --splice-indexes argument, too short: "
             + args.splice_indexes)
try:
    for string in split1:
        split2 = string.split(",")
        if len(split2) < 1:
            sys.exit("invalid --splice-indexes argument, too-short element: "
                     + args.splice_indexes)
        int_list = []
        for int_str in split2:
            int_list.append(int(int_str))
        if not int_list == sorted(int_list):
            sys.exit("elements of --splice-indexes must be sorted: "
                     + args.splice_indexes)
        if args.add_delta == "true" and len(splice_array) == 0:
            leftmost = int_list[0]
            rightmost = int_list[-1]
            for i in range(1, delta_window / 2 + 1):
                int_list.append(leftmost - i)
                int_list.append(rightmost + i)
            int_list = sorted(int_list)
        left_context += -int_list[0]
        right_context += int_list[-1]
        splice_array.append(int_list)
except ValueError as e:
    sys.exit("invalid --splice-indexes argument " + args.splice_indexes + e)
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

## Work out cnn_array e.g. cnn_array = [ [8,8,1,1,256,1,3,1,1,3,1], [4,3,1,1,256,1,3,1,1,3,1 ] ]
cnn_array = []
split1 = args.cnn_indexes.split(" ");  # we already checked the string is nonempty.
if len(split1) < 1:
    sys.exit("invalid --cnn-indexes argument, too short: "
             + args.cnn_indexes)
try:
    for string in split1:
        split2 = string.split(",")
        if len(split2) != 11:
            sys.exit("invalid --cnn-indexes argument, must contain 11 positive integers: "
                     + args.cnn_indexes)
        int_list = []
        for int_str in split2:
            if int(int_str) < 1:
                sys.exit("non-positive number in --cnn-indexes argument: {0}".format(int(int_str)))
            int_list.append(int(int_str))
        cnn_array.append(int_list)
except ValueError as e:
    sys.exit("invalid --cnn-indexes argument " + args.cnn_indexes + str(e))
num_cnn_layers = len(cnn_array)

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
    if l == 1 and skip_lda:
        if args.ivector_dim > 0:
            print('component name=ivector-affine type=NaturalGradientAffineComponent '
                  'input-dim={0} output-dim={0}'.format(args.ivector_dim), file=f)

        if args.use_mfcc == "true":
            # generate the IDCT matrix and write to the file
            idct_matrix = ComputeIDctMatrix(args.feat_dim, args.feat_dim, args.cepstral_lifter)
            # append a zero column to the matrix, this is the bias of the fixed affine component
            zero_col = np.zeros((args.feat_dim, 1))
            f2 = open(args.config_dir + "/idct.mat", "w")
            print('[ ', file=f2)
            np.savetxt(f2, np.append(idct_matrix, zero_col, 1), fmt='%.6e')
            print(' ]', file=f2)
            f2.close()

            print('component name=idct type=FixedAffineComponent matrix={0}/idct.mat'.
                  format(args.config_dir), file=f)

        if args.add_delta == "true":
            conv_input_x_dim = len(splice_array[0]);
            conv_input_y_dim = args.feat_dim;
            conv_input_z_dim = 1;
            conv_vectorization = "yzx";
            conv_filt_x_dim = delta_window;
            conv_filt_y_dim = 1;
            conv_filt_x_step = 1;
            conv_filt_y_step = 1;
            conv_num_filters = 3;

            print('component name={0}_conv type=ConvolutionComponent '
                  'input-x-dim={1} input-y-dim={2} input-z-dim={3} '
                  'filt-x-dim={4} filt-y-dim={5} '
                  'filt-x-step={6} filt-y-step={7} '
                  'num-filters={8} input-vectorization-order={9}'.
                  format("add-delta", conv_input_x_dim, conv_input_y_dim, conv_input_z_dim,
                         conv_filt_x_dim, conv_filt_y_dim,
                         conv_filt_x_step, conv_filt_y_step,
                         conv_num_filters, conv_vectorization), file=f)


        for cl in range(0, num_cnn_layers):
            if cl == 0:
                if args.add_delta == "true":
                    conv_input_x_dim = len(splice_array[0]) - delta_window + 1;
                    conv_input_y_dim = args.feat_dim;
                    conv_input_z_dim = 3;
                    conv_vectorization = "zyx";
                else:
                    conv_input_x_dim = len(splice_array[0]);
                    conv_input_y_dim = args.feat_dim;
                    conv_input_z_dim = 1;
                    conv_vectorization = "yzx";
            else:
                conv_input_x_dim = maxp_num_pools_x;
                conv_input_y_dim = maxp_num_pools_y;
                conv_input_z_dim = maxp_num_pools_z;
                conv_vectorization = "zyx";

            conv_filt_x_dim = cnn_array[cl][0];
            conv_filt_y_dim = cnn_array[cl][1];
            conv_filt_x_step = cnn_array[cl][2];
            conv_filt_y_step = cnn_array[cl][3];
            conv_num_filters = cnn_array[cl][4];

            if conv_filt_x_dim > conv_input_x_dim or conv_filt_y_dim > conv_input_y_dim:
                sys.exit("invalid convolution filter size vs. input size")
            if conv_filt_x_step > conv_filt_x_dim or conv_filt_y_step > conv_filt_y_dim:
                sys.exit("invalid convolution filter step vs. filter size")

            print('component name=L{0}_conv type=ConvolutionComponent '
                  'input-x-dim={1} input-y-dim={2} input-z-dim={3} '
                  'filt-x-dim={4} filt-y-dim={5} '
                  'filt-x-step={6} filt-y-step={7} '
                  'num-filters={8} input-vectorization-order={9}'.
                  format(cl, conv_input_x_dim, conv_input_y_dim, conv_input_z_dim,
                         conv_filt_x_dim, conv_filt_y_dim,
                         conv_filt_x_step, conv_filt_y_step,
                         conv_num_filters, conv_vectorization), file=f)

            maxp_input_x_dim = (1 + (conv_input_x_dim - conv_filt_x_dim) / conv_filt_x_step);
            maxp_input_y_dim = (1 + (conv_input_y_dim - conv_filt_y_dim) / conv_filt_y_step);
            maxp_input_z_dim = conv_num_filters;
            maxp_pool_x_size = cnn_array[cl][5];
            maxp_pool_y_size = cnn_array[cl][6];
            maxp_pool_z_size = cnn_array[cl][7];
            maxp_pool_x_step = cnn_array[cl][8];
            maxp_pool_y_step = cnn_array[cl][9];
            maxp_pool_z_step = cnn_array[cl][10];

            if maxp_input_x_dim < 1 or maxp_input_y_dim < 1 or maxp_input_z_dim < 1:
                sys.exit("non-positive maxpooling input size ({0}, {1}, {2})".
                         format(maxp_input_x_dim, maxp_input_y_dim, maxp_input_z_dim))
            if maxp_pool_x_size > maxp_input_x_dim or maxp_pool_y_size > maxp_input_y_dim or maxp_pool_z_size > maxp_input_z_dim:
                sys.exit("invalid maxpooling pool size vs. input size")
            if maxp_pool_x_step > maxp_pool_x_size or maxp_pool_y_step > maxp_pool_y_size or maxp_pool_z_step > maxp_pool_z_size:
                sys.exit("invalid maxpooling pool step vs. pool size")

            print('component name=L{0}_maxp type=MaxpoolingComponent '
                  'input-x-dim={1} input-y-dim={2} input-z-dim={3} '
                  'pool-x-size={4} pool-y-size={5} pool-z-size={6} '
                  'pool-x-step={7} pool-y-step={8} pool-z-step={9}'.
                  format(cl, maxp_input_x_dim, maxp_input_y_dim, maxp_input_z_dim,
                         maxp_pool_x_size, maxp_pool_y_size, maxp_pool_z_size,
                         maxp_pool_x_step, maxp_pool_y_step, maxp_pool_z_step), file=f)

            maxp_num_pools_x = 1 + (maxp_input_x_dim - maxp_pool_x_size) / maxp_pool_x_step;
            maxp_num_pools_y = 1 + (maxp_input_y_dim - maxp_pool_y_size) / maxp_pool_y_step;
            maxp_num_pools_z = 1 + (maxp_input_z_dim - maxp_pool_z_size) / maxp_pool_z_step;
            maxp_output = maxp_num_pools_x * maxp_num_pools_y * maxp_num_pools_z;

        cnn_reduced_ouput = args.cnn_reduced_dim;
        print('component name=dim-reduce-affine type=NaturalGradientAffineComponent '
              'input-dim={0} output-dim={1}'.format(maxp_output, cnn_reduced_ouput), file=f)

    if not skip_lda:
        cur_dim = (nonlin_output_dims[l-2] * len(splice_array[l-1]) if l > 1 else input_dim)
    else:
        cur_dim = (nonlin_output_dims[l-2] * len(splice_array[l-1]) if l > 1 else cnn_reduced_ouput + args.ivector_dim)

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

        if args.use_mfcc == "true":
            print('component-node name=idct component=idct input={0}'.format("input"),
                  file=f)
            splices = [ ('Offset(idct, {0})'.format(n) if n != 0 else 'idct') for n in splice_array[l-1] ]
            cur_input='Append({0})'.format(', '.join(splices))

        if args.add_delta == "true":
            print('component-node name={0}_conv component={0}_conv input={1}'.
                  format("add-delta", cur_input), file=f)
            cur_input="add-delta_conv"

        for cl in range(0, num_cnn_layers):
            print('component-node name=L{0}_conv component=L{0}_conv input={1}'.
                  format(cl, cur_input), file=f)
            print('component-node name=L{0}_maxp component=L{0}_maxp input=L{0}_conv'.
                  format(cl), file=f)
            cur_input = "L{0}_maxp".format(cl)
        print('component-node name=dim-reduce-affine component=dim-reduce-affine input={0}'.
              format(cur_input), file=f)
        cur_input='dim-reduce-affine'

        if args.ivector_dim > 0:
            print('component-node name=ivector-affine component=ivector-affine input=ReplaceIndex(ivector, t, 0)',
                  file=f)
            cur_input='Append({0}, {1})'.format(cur_input, 'ivector-affine')

    else:
        # e.g. cur_input = 'Append(Offset(renorm1, -2), renorm1, Offset(renorm1, 2))'
        splices = [ ('Offset(renorm{0}, {1})'.format(l-1, n) if n !=0 else 'renorm{0}'.format(l-1))
                    for n in splice_array[l-1] ]
        cur_input='Append({0})'.format(', '.join(splices))
    print('component-node name=affine{0} component=affine{0} input={1} '.
          format(l, cur_input), file=f)
    print('component-node name=nonlin{0} component=nonlin{0} input=affine{0}'.
          format(l), file=f)
    print('component-node name=renorm{0} component=renorm{0} input=nonlin{0}'.
          format(l), file=f)

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


