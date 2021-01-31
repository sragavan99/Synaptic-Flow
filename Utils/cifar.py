import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, ConcatDataset, DataLoader

# dset: "train"/"val"/"trainval"/"test"
# seed is passed to construct a random train/test split of CIFAR-10
# this currently doesn't support accepting a seed for a 45k/5k split, though this would be a quick change
def get_cifar_dataset(dset, transform, seed=None):
    if seed is None:
        if dset == "test":
            return torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)
        else:
            # get the official training set
            trainvalset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
            
            # retrieve indices
            if dset == "trainval":
                return trainvalset
            elif dset == "train":
                idxs = torch.load('CIFAR10/train_idxs.pt')
                return Subset(trainvalset, idxs)
            else:
                assert(dset == "val")
                idxs = torch.load('CIFAR10/val_idxs.pt')
                return Subset(trainvalset, idxs)
    else:
        assert(dset in ['trainval', 'test']) # this is for creating shuffled CIFAR-10 with 50k/10k split as of now
        trainset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=transform)

        # getting all 60000 data points of CIFAR-10 together
        fullset = ConcatDataset([trainset, testset])
        fullloader = DataLoader(fullset, batch_size=100000, shuffle=False)

        for batch in fullloader:
            images, labels = batch

        # now finding a subset
        idxs = torch.LongTensor([])
        g = torch.Generator()
        g.manual_seed(seed)
        for y in range(10):
            target_idxs = (labels == y).nonzero().squeeze()
            total = target_idxs.shape[0]
            assert(total == 6000)
            split = 1000

            # random shuffle
            target_idxs = target_idxs[torch.randperm(total, generator=g)]
            if dset == 'trainval':
                idxs = torch.cat((idxs, target_idxs[split:]))
            else:
                idxs = torch.cat((idxs, target_idxs[:split]))

        torch.save(idxs, dset + '_' + str(seed) + '.pt')

        return Subset(fullset, idxs)
        
        