from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

dir_dataset = Path('../data/cifar-data')
dir_wandb = Path('./wandb')  # for local wandb run files
dir_wandb_saved_files = Path('./wandb_saved_files')  # for personal saved files in wandb process

assert dir_dataset != '../data/cifar-data', 'change dir_dataset to your own path!'


class LrSchedulerType(Enum):
    Step = auto()
    Cyclic = auto()


class MethodName(Enum):
    Trades = auto()
    TradesUpdated = auto()
    Teat = auto()
    TeatUpdated = auto()

    Fast = auto()
    FastUpdated = auto()
    GradAlign = auto()
    GradAlignUpdated = auto()


methods_lr_schedule = {
    MethodName.Trades: 'step-wise',
    MethodName.TradesUpdated: 'step-wise',
    MethodName.Teat: 'step-wise',
    MethodName.TeatUpdated: 'step-wise',

    MethodName.Fast: 'cyclic-wise',
    MethodName.FastUpdated: 'cyclic-wise',  #   step
    MethodName.GradAlign: 'cyclic-wise',
    MethodName.GradAlignUpdated: 'cyclic-wise',
}


class DataName(Enum):
    CIFAR10 = auto()
    CIFAR100 = auto()
    TinyImageNet = auto()


class ModelName(Enum):
    LeNet = auto()
    ResNet_18 = auto()
    PreActResNet_18 = auto()  # for CIFAR data
    WideResNet_28_10 = auto()  # for CIFAR data
    WideResNet_34_8 = auto()  # for TinyImageNet data


class Seed(Enum):
    SEED_1 = 32516
    output_path = auto()

class OB(Enum):
    alpha = auto()
    beta = auto()

class RobEvalAttack(Enum):
    PGD_my = auto()
    PGD_common = auto()
    PGD_50_10 = auto()  # 50 iterations with 10 restarts
    CW = auto()  # C&W attack
    Auto = auto()  # auto attack


@dataclass
class ConfigLinfAttack:
    epsilon: float = 8 / 255
    perturb_steps: int = None
    step_size: float = None


@dataclass
class ParamsLamRange:  # for multi-step AT
    # lam for rob
    lam_min: int = 3
    lam_max: int = 9


@dataclass
class ParamsStepSizeRange:  # for single-step AT
    # lam for rob
    step_size_min: float = 2 / 255
    step_size_max: float = 12 / 255


@dataclass
class Config:
    lr_init: float = 0.1
    lr_max: float = 0.1
    lr_min: float = 0.0
    momentum: float = 0.9
    weight_decay: float = 2e-4
    total_epoch: int = 110

    lr_schedule_type: LrSchedulerType = None

    method: MethodName = None
    data: DataName = None
    model: ModelName = None
    seed: Seed = None
    output_path: Seed = None
    alpha: OB = None
    beta: OB = None

    param_atk_train: ConfigLinfAttack = None
    param_atk_eval: ConfigLinfAttack = None

    param_lam_range: ParamsLamRange = None
    param_step_size_range: ParamsStepSizeRange = None
    param_fixed_lam: int = None
