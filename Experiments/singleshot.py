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

    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, trainset, args.workers, corrupt_prob=args.prune_corrupt, length=args.prune_dataset_ratio * num_classes)
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
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    if args.rewind_epochs > 0:
        pre_result = train_eval_loop_midsave(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose, args.rewind_epochs)
    else:
        pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose)

    torch.save(model.state_dict(),"{}/pre-trained.pt".format(args.result_dir))

    ## Prune ##
    if args.pruner in ["synflow", "altsynflow", "synflowmag", "rsfgrad"]:
        model.double() # to address exploding/vanishing gradients in SynFlow for deep models
    print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
    pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    sparsity = args.sparsity # 10**(-float(args.compression))
    prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
    if args.pruner in ["synflow", "altsynflow", "synflowmag", "rsfgrad"]:
        model.float() # to address exploding/vanishing gradients in SynFlow for deep models
    torch.save(model.state_dict(),"{}/post-prune-model.pt".format(args.result_dir))

    ## Compute Path Count ##
    print("Number of paths", get_path_count(mdl=model, arch=args.model))

    # Load weights from rewinded model
    if args.rewind_epochs > 0:
        original_dict = torch.load("model_pretrain_midway.pt", map_location=device)
        original_weights = dict(filter(lambda v: 'mask' not in v[0], original_dict.items()))
        model_dict = model.state_dict()
        model_dict.update(original_weights)
        model.load_state_dict(model_dict)
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


