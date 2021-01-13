import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset

# dset: "train"/"val"/"trainval"/"test"
def get_cifar_dataset(dset, transform):
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
            
        
        