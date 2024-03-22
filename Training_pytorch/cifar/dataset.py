import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# original get10()
# def get10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
#     data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
#     num_workers = kwargs.setdefault('num_workers', 1)
#     kwargs.pop('input_size', None)
#     print("Building CIFAR-10 data loader with {} workers".format(num_workers))
#     ds = []
#     if train:
#         train_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(
#                 root=data_root, train=True, download=True,
#                 transform=transforms.Compose([
#                     transforms.Pad(4),
#                     transforms.RandomCrop(32),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                 ])),
#             batch_size=batch_size, shuffle=True, **kwargs)

#         ds.append(train_loader)
#     if val:
#         test_loader = torch.utils.data.DataLoader(
#             datasets.CIFAR10(
#                 root=data_root, train=False, download=True,
#                 transform=transforms.Compose([
#                     transforms.ToTensor(),
#                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                 ])),
#             batch_size=batch_size, shuffle=False, **kwargs)
#         ds.append(test_loader)
#     ds = ds[0] if len(ds) == 1 else ds
#     return ds

def get10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, subset_size=None, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    if train:
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform)

        if subset_size is not None and subset_size < len(train_dataset):
            train_dataset = torch.utils.data.Subset(train_dataset, range(subset_size))
            
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        
        ds.append(train_loader)
        
    if val:
        test_dataset = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform)

        if subset_size is not None and subset_size < len(test_dataset):
            test_dataset = torch.utils.data.Subset(test_dataset, range(subset_size))
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        
        ds.append(test_loader)
        
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def get100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

