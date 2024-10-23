import os
import torch

from model.LeNet import LeNet
from model.AlexNet import AlexNet
from model.ResNet import ResNet18, ResNet34, ResNet50, ResNet101
from model.PreActResNet import PreActResNet18, PreActResNet34, PreActResNet50, PreActResNet101
from model.WideResNet import WideResNet
from model.DenseNet import DenseNet121, DenseNet161, DenseNet169, DenseNet201


MODEL = {
    'DenseNet121': DenseNet121,
    'PreActResNet18': PreActResNet18,
    'ResNet18': ResNet18,
    'WideResNet34-10': WideResNet,
    'LeNet': LeNet,
    'AlexNet': AlexNet,

}

from model.models.lenet import LeNet1
from model.models.preact_resnet import PreActResNet181
from model.models.resnet import ResNet181
MODEL_cspgi = {
    'PreActResNet18': PreActResNet181,
    'ResNet18': ResNet181,
    'LeNet': LeNet1,
}

from model.models1.LeNet import LeNet2
from model.models1.preact_resnet_cifar import PreActResNet182
from model.models1.resnet import ResNet182
MODEL_eacc = {
    'PreActResNet18': PreActResNet182,
    'ResNet18': ResNet182,
    'LeNet': LeNet2,
}


def save_model(model, model_name, model_path):
    model_path = os.path.join(model_path, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, 'checkpoint.pth'))


def load_model(model, model_path, model_name='checkpoint.pth'):
    if not os.path.exists(model_path):
        raise Exception("Model doesn't exists! Train first!")
    model_state_dict = torch.load(os.path.join(model_path, model_name))
    model.load_state_dict(model_state_dict['model_state_dict'])
    print("Model Loaded Success: {}".format(model_path))
