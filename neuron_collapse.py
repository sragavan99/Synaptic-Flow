# Functions that in some way measure the degree of neuron collapse that occurs

from Models.lottery_vgg import vgg16_bn
import torch

# mode: 'in' (incoming weights), 'out' (outoing weights), 'all' (all filters)
def number_active_filters(mdl=None, file=None, arch=None, mode='in'):

    surviving = 0
    total = 0
    # just compute 'in' for now
    
    if not mdl:
        mdl = vgg16_bn((3, 32, 32), 10, False, False)
        mdl.load_state_dict(torch.load(file, map_location=torch.device('cpu')))

    for layer in mdl.state_dict():
        if 'conv' not in layer or 'mask' not in layer or 'bias' in layer:
            continue

        mask = mdl.state_dict()[layer]
        input_sums = torch.sum(mask, dim=(1,2,3))
        surviving += torch.nonzero(input_sums).shape[0]
        total += mask.shape[0]

    return surviving

# This code is non-functional due to differences between open-lth and Synaptic-Flow
# mdl_path = '/n/fs/visualai-scr/arjuns/prune/open_lth/slurm/output/open_lth_data/lottery_69db99a739267a8705b0419258a0b823/replicate_1/level_1/main/mask.pth'
# surviving, total = number_active_filters(file=mdl_path)
# print(surviving)
# print(total)
