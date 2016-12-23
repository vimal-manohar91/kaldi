#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, 'steps/')
# the following is in case we weren't running this from the normal directory.
sys.path.insert(0, os.path.realpath(os.path.dirname(sys.argv[0])) + '/')

import libs.nnet3.xconfig.parser as xparser
import libs.common as common_lib


def get_args():
    # we add compulsary arguments as named arguments for readability
    parser = argparse.ArgumentParser(
        description="Reads an xconfig file and creates config files "
                    "for neural net creation and training",
        epilog='Search egs/*/*/local/{nnet3,chain}/*sh for examples')
    parser.add_argument('--xconfig-file', required=True,
                        help='Filename of input xconfig file')
    parser.add_argument('--config-dir', required=True,
                        help='Directory to write config files and variables')
    parser.add_argument('--nnet-edits', type=str, default=None,
                        action=common_lib.NullstrToNoneAction,
                        help="Edit network before getting nnet3-info")

    print(' '.join(sys.argv))

    args = parser.parse_args()
    args = check_args(args)

    return args


def check_args(args):
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)
    return args


def backup_xconfig_file(xconfig_file, config_dir):
    """we write a copy of the xconfig file just to have a record of the
    original input.
    """
    try:
        xconfig_file_out = open(config_dir + '/xconfig', 'w')
    except:
        raise Exception('{0}: error opening file '
                        '{1}/xconfig for output'.format(
                            sys.argv[0], config_dir))
    try:
        xconfig_file_in = open(xconfig_file)
    except:
        raise Exception('{0}: error opening file {1} for input'.format(
                            sys.argv[0], config_dir))

    print("# This file was created by the command:\n"
          "# {0}\n"
          "# It is a copy of the source from which the config files in "
          "# this directory were generated.\n".format(' '.join(sys.argv)),
          file=xconfig_file_out)

    while True:
        line = xconfig_file_in.readline()
        if line == '':
            break
        print(line.strip(), file=xconfig_file_out)
    xconfig_file_out.close()
    xconfig_file_in.close()


def write_expanded_xconfig_files(config_dir, all_layers):
    """ This functions writes config_dir/xconfig.expanded.1 and
    config_dir/xconfig.expanded.2, showing some of the internal stages of
    processing the xconfig file before turning it into config files.
    """
    try:
        xconfig_file_out = open(config_dir + '/xconfig.expanded.1', 'w')
    except:
        raise Exception('{0}: error opening file '
                        '{1}/xconfig.expanded.1 for output'.format(
                            sys.argv[0], config_dir))

    print('# This file was created by the command:\n'
          '# ' + ' '.join(sys.argv) + '\n'
          '#It contains the same content as ./xconfig but it was parsed and\n'
          '#default config values were set.\n'
          '# See also ./xconfig.expanded.2\n', file=xconfig_file_out)

    for layer in all_layers:
        print(str(layer), file=xconfig_file_out)
    xconfig_file_out.close()

    try:
        xconfig_file_out = open(config_dir + '/xconfig.expanded.2', 'w')
    except:
        raise Exception('{0}: error opening file '
                        '{1}/xconfig.expanded.2 for output'.format(
                            sys.argv[0], config_dir))

    print('# This file was created by the command:\n'
          '# ' + ' '.join(sys.argv) + '\n'
          '# It contains the same content as ./xconfig but it was parsed,\n'
          '# default config values were set, \n'
          '# and Descriptors (input=xxx) were normalized.\n'
          '# See also ./xconfig.expanded.1\n\n',
          file=xconfig_file_out)

    for layer in all_layers:
        layer.normalize_descriptors()
        print(str(layer), file=xconfig_file_out)
    xconfig_file_out.close()


def get_config_headers():
    """ This function returns a map from config-file basename
    e.g. 'init', 'ref', 'layer1' to a documentation string that goes
    at the top of the file.
    """
    # resulting dict will default to the empty string for any config files not
    # explicitly listed here.
    ans = defaultdict(str)

    ans['init'] = (
        '# This file was created by the command:\n'
        '# ' + ' '.join(sys.argv) + '\n'
        '# It contains the input of the network and is used in\n'
        '# accumulating stats for an LDA-like transform of the\n'
        '# input features.\n')
    ans['ref'] = (
        '# This file was created by the command:\n'
        '# ' + ' '.join(sys.argv) + '\n'
        '# It contains the entire neural network, but with those\n'
        '# components that would normally require fixed vectors/matrices\n'
        '# read from disk, replaced with random initialization\n'
        '# (this applies to the LDA-like transform and the\n'
        '# presoftmax-prior-scale, if applicable).  This file\n'
        '# is used only to work out the left-context and right-context\n'
        '# of the network.\n')
    ans['final'] = (
        '# This file was created by the command:\n'
        '# ' + ' '.join(sys.argv) + '\n'
        '# It contains the entire neural network.\n')

    return ans


# This is where most of the work of this program happens.
def write_config_files(config_dir, all_layers):
    # config_basename_to_lines is map from the basename of the
    # config, as a string (i.e. 'ref', 'all', 'init') to a list of
    # strings representing lines to put in the config file.
    config_basename_to_lines = defaultdict(list)

    config_basename_to_header = get_config_headers()

    for layer in all_layers:
        try:
            pairs = layer.get_full_config()
            for config_basename, line in pairs:
                config_basename_to_lines[config_basename].append(line)
        except Exception as e:
            print("{0}: error producing config lines from xconfig "
                  "line '{1}': error was: {2}".format(sys.argv[0],
                                                      str(layer), repr(e)),
                  file=sys.stderr)
            # we use raise rather than raise(e) as using a blank raise
            # preserves the backtrace
            raise

    for basename, lines in config_basename_to_lines.items():
        header = config_basename_to_header[basename]
        filename = '{0}/{1}.config'.format(config_dir, basename)
        try:
            f = open(filename, 'w')
            print(header, file=f)
            for line in lines:
                print(line, file=f)
            f.close()
        except Exception as e:
            print('{0}: error writing to config file {1}: error is {2}'.format(
                    sys.argv[0], filename, repr(e)), file=sys.stderr)
            # we use raise rather than raise(e) as using a blank raise
            # preserves the backtrace
            raise


def add_back_compatibility_info(config_dir, nnet_edits=None):
    """This will be removed when python script refactoring is done."""

    common_lib.run_kaldi_command("nnet3-init {0}/ref.config "
                                 "{0}/ref.raw".format(config_dir))
    model = "{0}/ref.raw".format(config_dir)
    if nnet_edits is not None:
        model = """nnet3-copy --edits='{0}' {1} - |""".format(nnet_edits,
                                                              model)

    print("""nnet3-info "{0}" | head -4""".format(model), file=sys.stderr)
    out, err = common_lib.run_kaldi_command("""nnet3-info "{0}" | """
                                            """head -4""".format(model))
    # out looks like this
    # left-context: 7
    # right-context: 0
    # num-parameters: 90543902
    # modulus: 1
    info = {}
    for line in out.split("\n"):
        parts = line.split(":")
        if len(parts) != 2:
            continue
        info[parts[0].strip()] = int(parts[1].strip())

    # Writing the back-compatible vars file
    #   model_left_context=0
    #   model_right_context=7
    #   num_hidden_layers=3
    vf = open('{0}/vars'.format(config_dir), 'w')
    vf.write('model_left_context={0}\n'.format(info['left-context']))
    vf.write('model_right_context={0}\n'.format(info['right-context']))
    vf.write('num_hidden_layers=1\n')
    vf.close()

    common_lib.force_symlink("final.config".format(config_dir),
                             "{0}/layer1.config".format(config_dir))


def main():
    args = get_args()
    backup_xconfig_file(args.xconfig_file, args.config_dir)
    all_layers = xparser.read_xconfig_file(args.xconfig_file)
    write_expanded_xconfig_files(args.config_dir, all_layers)
    write_config_files(args.config_dir, all_layers)
    add_back_compatibility_info(args.config_dir, args.nnet_edits)


if __name__ == '__main__':
    main()


# test:
# mkdir -p foo; (echo 'input dim=40 name=input'; echo 'output name=output input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo
#  mkdir -p foo; (echo 'input dim=40 name=input'; echo 'output-layer name=output dim=1924 input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo

# mkdir -p foo; (echo 'input dim=40 name=input'; echo 'relu-renorm-layer name=affine1 dim=1024'; echo 'output-layer name=output dim=1924 input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo

# mkdir -p foo; (echo 'input dim=100 name=ivector'; echo 'input dim=40 name=input'; echo 'fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=foo/bar/lda.mat'; echo 'output-layer name=output dim=1924 input=Append(-1,0,1)')  >xconfig; ./xconfig_to_configs.py xconfig foo
