import torch
import json

from Models.lottery_vgg import vgg16_bn
from Models.lottery_vgg import path_vgg16_bn
from Models.lottery_resnet import wide_resnet32
from Models.lottery_resnet import path_wide_resnet32

### Changes to files outside this one ###

# lottery_resnet.py, lottery_vgg.py: added "path_vgg16_bn" and "path_wide_resnet32" which still call the same
# VGG16 and WRN classes but use sum-pooling (LPPool2d(1.0)) wherever any pooling happens. This is necessary because
# the path count at a neuron after pooling should be the sum of the path counts at the neurons that were
# included in the pooling operation.

# Utils/path_instructions: weight_layers.txt contains the names of layers (either in VGG16 or WRN) that are essentially
# linear, so fully-connected and convolutional layers. (Residual layers aren't included since they shouldn't need
# to be modified for path-counting to work)

# bn_layers.txt: same thing but for layers that have batch-norm
# instructions.json: just puts together what's in those two individual files

### Information on this file ###

# the pipeline is to take a trained model, construct the corresponding "path model" from it, then run a forward pass
# through the path model with all 1's as input

# a "path model" is constructed from the original model via three changes (other than the pooling change):
# 1. all weights in linear layers are replaced by their masks
# 2. all biases are set to 0
# 3. all batch-norm is effectively removed

# get_path_count is the only function needed from outside for counting paths, and should accept either a loaded model
# or a path to the file with the model's state_dict as input

# takes a trained model (basic_mdl) and path_mdl as input
# and overwrites path_mdl with the path model constructed from the trained model
def path_clone(basic_mdl, path_mdl, weight_layers, bn_layers, debug=False):
    basic_sd = basic_mdl.state_dict()
    path_sd = path_mdl.state_dict()
    for name in weight_layers:
        # masking the weights
        if name + '.weight' in path_sd:
            path_sd[name + '.weight'] = basic_sd[name + '.weight_mask']
        elif debug:
            print("No weight for layer", name)
        # killing the biases, if they exist
        if name + '.bias' in path_sd:
            path_sd[name + '.bias'].zero_()
        elif debug:
            print("No bias for layer", name)
    for name in bn_layers:
        # killing the batchnorm
        # batch norm maps x to gamma * (x - running_mean)/sqrt(running_var) + beta
        if name + '.weight' in path_sd:
            path_sd[name + '.weight'].fill_(1) # gamma
            path_sd[name + '.bias'].fill_(0) # beta
            path_sd[name + '.running_mean'].fill_(0)
            path_sd[name + '.running_var'].fill_(1) # may be necessary to make this 1 - 1e-5 or something to get exact integer values (batch norm adds a little to running_var for numerical reasons)
        elif debug:
            print("No bn for layer", name)
    path_mdl.load_state_dict(path_sd)

# given a path model, computes how many paths go through it
def count_paths(path_mdl):
    path_mdl.eval()
    path_mdl.double()
    input = torch.ones((1, 3, 32, 32), dtype=torch.float64)
    output = path_mdl(input)
    return torch.sum(output).item()

# loading layer names
with open('Utils/path_instructions/instructions.json', 'r') as f:
    instructions = json.load(f)

weight_layers = instructions['weight_layers']
bn_layers = instructions['bn_layers']

# arch should be 'vgg16-bn' or 'wide-resnet32' and match the model/file being passed in
def get_path_count(mdl=None, file=None, arch=None):
    if arch == 'vgg16-bn':
        if not mdl:
            mdl = vgg16_bn((3, 32, 32), 10, False, False)
            mdl.load_state_dict(torch.load(file))
        path_mdl = path_vgg16_bn((3, 32, 32), 10, False, False)
    elif arch == 'wide-resnet32':
        if not mdl:
            mdl = wide_resnet32((3, 32, 32), 10, False, False)
            mdl.load_state_dict(torch.load(file))
        path_mdl = path_wide_resnet32((3, 32, 32), 10, False, False)
    else:
        raise(Exception("Invalid arch"))

    path_clone(mdl, path_mdl, weight_layers, bn_layers)
    return count_paths(path_mdl)