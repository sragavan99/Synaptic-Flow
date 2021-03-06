import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from Utils import plot
from train import *
from prune import *
from path_counting import get_path_count
from neuron_collapse import number_active_filters

# This code is currently very specific to the setting where you
# 1. load a model from a file
# 2. shuffle the surviving weights in each layer
# 3. then post-train for 155 epochs

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu, args.seed)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset)

    if args.validation:
        trainset = 'train'
        evalset = 'val'
    else:
        trainset = 'trainval'
        evalset = 'test'

    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, trainset, args.workers, corrupt_prob=args.prune_corrupt)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, trainset, args.workers, corrupt_prob=args.train_corrupt)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, evalset, args.workers)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)

    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)
    torch.save(model.state_dict(),"{}/init-model.pt".format(args.result_dir))

    ## Pre-Train ##
    assert(args.pre_epochs == 0)
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose)

    torch.save(model.state_dict(),"{}/pre-trained.pt".format(args.result_dir))

    ## Load in the model ##
    # maskref_dict = torch.load(args.mask_file, map_location=device)
    # model_dict = model.state_dict()

    # mask_dict = dict(filter(lambda v: 'mask' in v[0], maskref_dict.items()))
    # print("Keys being loaded\n", '\n'.join(mask_dict.keys()))
    # model_dict.update(mask_dict)

    model.load_state_dict(torch.load(args.model_file, map_location=device))

    # sanity check part 1
    dict_init = {}
    for name, param in model.state_dict().items():
        print(name)
        if name.endswith('weight') and 'shortcut' not in name and 'bn' not in name:
            mask = model.state_dict()[name + '_mask']
            dict_init[name] = (param.sum().item(), mask.sum().item(), (param * mask).sum().item())

    ## This uses a pruner but only for the purpose of shuffling weights ##
    assert(args.prune_epochs == 0 and args.weightshuffle)
    if args.pruner in ["synflow", "altsynflow", "synflowmag", "rsfgrad"]:
        model.double() # to address exploding/vanishing gradients in SynFlow for deep models
    print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
    pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    sparsity = args.sparsity # 10**(-float(args.compression))
    prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert, args.weightshuffle)
    if args.pruner in ["synflow", "altsynflow", "synflowmag", "rsfgrad"]:
        model.float() # to address exploding/vanishing gradients in SynFlow for deep models
    torch.save(model.state_dict(),"{}/post-prune-model.pt".format(args.result_dir))

    # sanity check part 2
    for name, param in model.state_dict().items():
        print(name)
        if name.endswith('weight') and 'shortcut' not in name and 'bn' not in name:
            mask = model.state_dict()[name + '_mask']
            print(name, dict_init[name], param.sum().item(), mask.sum().item(), (param * mask).sum().item())

    ## Compute Path Count ##
    print("Number of paths", get_path_count(mdl=model, arch=args.model))
    print("Number of active filters", number_active_filters(mdl=model, arch=args.model))

    ## Post-Train ##
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                  test_loader, device, args.post_epochs, args.verbose) 

    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
    prune_result = metrics.summary(model, 
                                   pruner.scores,
                                   metrics.flop(model, input_shape, device),
                                   lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        plot.get_plots(post_result, save_path="{}/plots.png".format(args.result_dir))
        prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/post-train-model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))


