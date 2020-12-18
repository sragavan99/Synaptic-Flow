import torch
import json
import matplotlib.pyplot as plt
import copy

from Models.lottery_vgg import vgg16_bn
from Models.lottery_resnet import wide_resnet32

# wgt_model: typically the model at init, this is the model where the weights come from
# mask_model: the model from which we get the mask to decide which weights survived, if not provided assume all weights survive
def visualise_surviving_weights(wgt_mdl, mask_mdl=None, rescale=False, bins=50):
    wgt_sd = wgt_mdl.state_dict()
    if mask_mdl:
        mask_sd = mask_mdl.state_dict()

    all_wgts = []
    for name, param in wgt_sd.items():
        # latter conditions remove unprunable parameters i.e. residual connections and batchnorm layers
        if name.endswith('weight') and 'shortcut' not in name and 'bn' not in name:
            flat_wgt = param.reshape(-1)

            if rescale:
                flat_wgt = flat_wgt/torch.std(flat_wgt)

            if mask_mdl:
                flat_mask = mask_sd[name + '_mask'].reshape(-1).bool()
                flat_wgt = flat_wgt[flat_mask]
            all_wgts.append(copy.deepcopy(flat_wgt))

    all_wgts = torch.cat(all_wgts)
    print("Number of weights", all_wgts.shape)
    plt.hist(all_wgts.detach().cpu().numpy(), bins=bins)
    plt.show()

# rescale: whether to standardise the weights in each layer before visualising
def visualise(wgt_mdl=None, wgt_file=None, mask_mdl=None, mask_file=None, arch=None, rescale=False, bins=50):
    if arch == 'vgg16-bn':
        if not wgt_mdl:
            wgt_mdl = vgg16_bn((3, 32, 32), 10, False, False)
            wgt_mdl.load_state_dict(torch.load(wgt_file, map_location=torch.device('cpu')))
        if not mask_mdl:
            mask_mdl = vgg16_bn((3, 32, 32), 10, False, False)
            mask_mdl.load_state_dict(torch.load(mask_file, map_location=torch.device('cpu')))
    elif arch == 'wide-resnet32':
        if not wgt_mdl:
            wgt_mdl = wide_resnet32((3, 32, 32), 10, False, False)
            wgt_mdl.load_state_dict(torch.load(wgt_file, map_location=torch.device('cpu')))
        if not mask_mdl:
            mask_mdl = wide_resnet32((3, 32, 32), 10, False, False)
            mask_mdl.load_state_dict(torch.load(mask_file, map_location=torch.device('cpu')))
    else:
        raise(Exception("Invalid arch"))

    visualise_surviving_weights(wgt_mdl, mask_mdl, rescale, bins)