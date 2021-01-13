import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *

def run(args):
    if not args.save:
        print("This experiment requires an expid.")
        quit()

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

    prune_loader = load.dataloader(args.dataset, args.train_batch_size, trainset, args.workers, corrupt_prob=args.prune_corrupt)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, trainset, args.workers, corrupt_prob=args.train_corrupt)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, evalset, args.workers)

    ## Model ##
    print('Creating {} model.'.format(args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier,
                                                     args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Save Original ##
    torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    ## Train-Prune Loop ##
    sparsity = args.sparsity
    try:
        level = args.level_list[0]
    except IndexError:
        raise ValueError("'--level-list' must have size >= 1.")
    
    print('{} compression ratio, {} train-prune levels'.format(sparsity, level))
    
    # Reset Model, Optimizer, and Scheduler
    model.load_state_dict(torch.load("{}/model.pt".format(args.result_dir), map_location=device))
    optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
    scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))
    
    for l in range(level):

        # Pre Train Model
        level_train_result = train_eval_loop(model, loss, optimizer, scheduler, prune_loader, 
                        test_loader, device, args.pre_epochs, args.verbose)
        level_train_result.to_pickle(f'{args.result_dir}/train-level-{l}-metrics.pkl')
        
        torch.save(model.state_dict(),"{}/train-level-{}.pt".format(args.result_dir, l))
        
        # Prune Model
        pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
        #sparsity = (10**(-float(compression)))**((l + 1) / level)
        prune_loop(model, loss, pruner, prune_loader, device, sparsity,
                    args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)
        
        # Prune Result
        prune_result = metrics.summary(model, 
                                    pruner.scores,
                                    metrics.flop(model, input_shape, device),
                                    lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
        prune_result.to_pickle("{}/sparsity-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(sparsity), str(l + 1)))


        # Reset Model's Weights
        original_dict = torch.load("{}/model.pt".format(args.result_dir), map_location=device)
        original_weights = dict(filter(lambda v: (v[1].requires_grad == True), original_dict.items()))
        model_dict = model.state_dict()
        model_dict.update(original_weights)
        model.load_state_dict(model_dict)
        
        # Reset Optimizer and Scheduler
        optimizer.load_state_dict(torch.load("{}/optimizer.pt".format(args.result_dir), map_location=device))
        scheduler.load_state_dict(torch.load("{}/scheduler.pt".format(args.result_dir), map_location=device))


    # Train Model
    post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                    test_loader, device, args.post_epochs, args.verbose)
    
    # Save Data
    post_result.to_pickle("{}/post-train-{}-{}-{}.pkl".format(args.result_dir, args.pruner, str(sparsity),  str(level)))

    # Save final model
    torch.save(model.state_dict(), f'{args.result_dir}/post-final-train.pt')
    torch.save(optimizer.state_dict(), f'{args.result_dir}/optimizer.pt')
    torch.save(scheduler.state_dict(), f'{args.result_dir}/scheduler.pt')

