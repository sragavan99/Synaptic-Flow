#!/usr/bin/env python3

BASE = "python3 main.py --dataset cifar10 --model-class lottery --optimizer momentum --train-batch-size 128 --post-epochs 160 --lr-drops 80 120 --weight-decay 1e-4 --expid results --verbose"


# With pruning
for compression in range(1, 4):
    for levels in [1, 3]:
        for model in ['wide-resnet32', 'vgg16-bn']:
            for pruner in ['synflow', 'mag']:
                    for random_prune in [True, False]:
                        for random_train in [True, False]:                            
                            lr = 0.1
                            result_dir = f'levels-{levels}-{model}-{pruner}-lr-{lr}-{random_prune}-{random_train}-sparsity-{compression}'
                            cmd = ' '.join([BASE, f'--model={model}', f'--lr={lr}', f'--pruner={pruner}', f'--compression={compression}', f'--result-dir={result_dir}', f'--level-list {levels}'])
                            if pruner == 'mag':
                                cmd += ' --pre-epochs 160'
                                cmd += ' --experiment multishot'
                            if random_prune:
                                cmd += ' --prune-corrupt=1.0'
                            if random_train:
                                cmd += ' --train-corrupt=1.0'
                            print(cmd + '\n')

print('=' * 20 + ' train only ' + '=' * 20)

for model in ['wide-resnet32', 'vgg16-bn']:
    for random_train in [True, False]:
        lr = 0.1
        result_dir = f'no-prune-{model}-{lr}-{random_train}'
        cmd = ' '.join([BASE, f'--model={model}', f'--lr={lr}', '--compression=0', f'--result-dir={result_dir}', '--prune-epochs=0'])
        if random_train:
            cmd += ' --train-corrupt=1.0'
        print(cmd + '\n')

