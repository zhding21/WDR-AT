from models.wide_resnet_for_tinyimagenet import Wide_ResNet as WRN_imagenet
from models.wide_resnet_for_cifar import Wide_ResNet as WRN_cifar
from models.preact_resnet_cifar import PreActResNet18
from models.resnet import ResNet18
from models.LeNet import LeNet

__all__ = [
    'PreActResNet18',
    'WRN_cifar',
    'WRN_imagenet',
    'ResNet18',
    'LeNet'
]