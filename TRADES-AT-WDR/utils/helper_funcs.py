import random
import warnings

from torch.nn import Module

from config import *
from utils.dataloaders import *
from models import *
from trainers import *
from torch.optim.lr_scheduler import CyclicLR, MultiStepLR


def choose_model(dataset_name: DataName, model_name: ModelName):
    num_classes_dict = {
        DataName.CIFAR10: 10,
        DataName.CIFAR100: 100,
        DataName.TinyImageNet: 200,
    }

    num_classes = num_classes_dict[dataset_name]

    if model_name is ModelName.PreActResNet_18:
        assert dataset_name is not DataName.TinyImageNet
        model = PreActResNet18(num_classes=num_classes)
    elif model_name is ModelName.WideResNet_34_8:
        assert dataset_name is DataName.TinyImageNet
        model = WRN_imagenet(depth=34, widen_factor=8, num_classes=num_classes)
    elif model_name is ModelName.WideResNet_28_10:
        model = WRN_cifar(depth=28, widen_factor=10, num_classes=num_classes)
    elif model_name is ModelName.ResNet_18:
        model = ResNet18(num_classes=num_classes)
    elif model_name is ModelName.LeNet:
        model = LeNet(num_classes=num_classes)
    else:
        raise ValueError

    return model


def prepare_data(dataset_name: DataName, train_id, test_id):
    if dataset_name is DataName.CIFAR10:
        get_loader = get_cifar10_loader
        num_classes = 10
    elif dataset_name is DataName.CIFAR100:
        get_loader = get_cifar100_loader
        num_classes = 100
    elif dataset_name is DataName.TinyImageNet:
        get_loader = get_tiny_imagenet_200_loader
        num_classes = 200
    else:
        raise NotImplementedError

    train_loader, test_loader, targets = get_loader(train_id=train_id, test_id=test_id)
    return train_loader, test_loader, targets, num_classes


def get_trainer(name_method: MethodName):
    method_trainer_dict = {
        # ------ multi-step AT ------
        MethodName.Trades: Trades,
        MethodName.TradesUpdated: TradesUpdated,
        MethodName.Teat: Teat,
        MethodName.TeatUpdated: TeatUpdated,
        # ------ single-step AT ------
        MethodName.Fast: Fast,
        MethodName.FastUpdated: FastUpdated,
        MethodName.GradAlign: GradAlign,
        MethodName.GradAlignUpdated: GradAlignUpdated,

    }

    trainer = method_trainer_dict[name_method]
    print('=========================')
    print(f'Use {trainer.__name__}!')
    print('=========================')

    return trainer


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')
        warnings.warn("You are using CPU!")

    return device


def get_lr_scheduler(total_epoch: int, len_loader: int, lr_schedule_type: LrSchedulerType,
                     optimizer, lr_min=None, lr_max=None):
    lr_steps = total_epoch * len_loader
    if lr_schedule_type is LrSchedulerType.Step:
        # total 100 epoch, decay in 50/75 epoch
        lr_scheduler = MultiStepLR(optimizer,
                                   milestones=[int(lr_steps / 2 - len_loader), int(lr_steps * 3 / 4 - len_loader)],
                                   gamma=(lr_min / lr_max) ** (1 / 2))
    elif lr_schedule_type is LrSchedulerType.Cyclic:
        lr_scheduler = CyclicLR(optimizer, base_lr=lr_min, max_lr=lr_max,
                                step_size_up=int(lr_steps * 2 / 5 - len_loader),
                                step_size_down=int(lr_steps * 3 / 5 + len_loader))
    else:
        raise ValueError

    return lr_scheduler


def save_epoch_weights(save_dir: str, i_epoch: int, model: Module):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=False, exist_ok=True)
    weights_path = save_dir / f'weights_{i_epoch}.pth'
    torch.save(model.state_dict(), weights_path)


def wait_gpu(temperature: int = 60):
    import subprocess
    import time

    def is_gpu_free():
        query_threshold = {'temperature.gpu': temperature,
                           'memory.used': 800,
                           'utilization.gpu': 30,
                           'utilization.memory': 30}

        for t in query_threshold:
            result = subprocess.run(['nvidia-smi',
                                     f'--query-gpu={t}',
                                     '--format=csv,noheader'],
                                    stdout=subprocess.PIPE)

            result = int(result.stdout[:-1].decode('utf-8').split(' ')[0])

            if result > query_threshold[t]:
                print(f'{t}: {result} > {query_threshold[t]}, not ready')
                return False

        return True

    while not is_gpu_free():
        time.sleep(5)
        pass
