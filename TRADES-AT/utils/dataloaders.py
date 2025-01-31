import numpy as np
import torch
from torchvision import datasets, transforms

from config import dir_dataset, DataName
from utils.datasets import CIFAR10WithID, CIFAR100WithID, TinyImageNet, TinyImageNetWithID


def get_cifar10_loader(train_id=False, test_id=False):
    bs_train = 128
    bs_eval = 256

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if train_id:
        trainset = CIFAR10WithID(dir_dataset, train=True, transform=transform_train)
    else:
        trainset = datasets.CIFAR10(root=dir_dataset, train=True, download=False, transform=transform_train)

    targets = np.asarray(trainset.targets)

    if test_id:
        testset = CIFAR10WithID(dir_dataset, train=False, transform=transform_test)
    else:
        testset = datasets.CIFAR10(root=dir_dataset, train=False, download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs_train, shuffle=True,
                                               num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs_eval, shuffle=False,
                                              num_workers=0, pin_memory=True)

    return train_loader, test_loader, targets


def get_cifar100_loader(train_id=False, test_id=False):
    bs_train = 128
    bs_eval = 256

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if train_id:
        trainset = CIFAR100WithID(dir_dataset, train=True, transform=transform_train)
    else:
        trainset = datasets.CIFAR100(root=dir_dataset, train=True, download=False, transform=transform_train)

    targets = np.asarray(trainset.targets)

    if test_id:
        testset = CIFAR100WithID(dir_dataset, train=False, transform=transform_test)
    else:
        testset = datasets.CIFAR100(root=dir_dataset, train=False, download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs_train, shuffle=True,
                                               num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs_eval, shuffle=False,
                                              num_workers=6, pin_memory=True)

    return train_loader, test_loader, targets


def get_tiny_imagenet_200_loader(train_id=False, test_id=False):
    bs_train = 64
    bs_eval = 64

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if train_id:
        trainset = TinyImageNetWithID(dir_dataset, train=True, transform=transform_train)
    else:
        trainset = TinyImageNet(dir_dataset, train=True, transform=transform_train)

    targets = np.asarray(trainset.data.targets)

    if test_id:
        testset = TinyImageNetWithID(dir_dataset, train=False, transform=transform_test)
    else:
        testset = TinyImageNet(root=dir_dataset, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs_train, shuffle=True,
                                               num_workers=6, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs_eval, shuffle=False,
                                              num_workers=6, pin_memory=True)

    return train_loader, test_loader, targets


def get_test_loader(data: DataName):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    if data is DataName.CIFAR10:
        test_set = datasets.CIFAR10
        num_classes = 10
    elif data is DataName.CIFAR100:
        test_set = datasets.CIFAR100
        num_classes = 100
    elif data is DataName.TinyImageNet:
        test_set = TinyImageNet
        num_classes = 200
    else:
        raise NotImplementedError

    test_set = test_set(root=dir_dataset, train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False,
                                              num_workers=6, pin_memory=True)

    return test_loader, num_classes
