import pprint
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.optim as optim
import copy
import logging
import time
import torch
from config import *
from utils.helper_funcs import choose_model, get_trainer, set_seed, get_device, prepare_data, get_lr_scheduler
# from utils.helper_funcs_wandb import *
# from utils.log import Logger
from utils.tester import AdvTester
from bayes_opt import BayesianOptimization


logger = logging.getLogger(__name__)


def main(config: Config):

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(config.output_path, 'output_our.log'))
    logger.info(config)
    start_time = time.time()
    logger.info('start_time: %s; alp: %s; bet: %s', str(start_time), str(config.alpha), str(config.beta))

    set_seed(config.seed.value)
    device = get_device()

    # set_wandb_env(is_online=False)  # if offline, can be later synced with the `wandb sync` command.
    # param_config = make_wandb_config(config)
    # disable_debug_internal_log()  # need set internal log path in wandb init
    # run = wandb.init(project="ExampleExploitationAT", reinit=True, dir=dir_wandb,
    #                  group=f'{config.data.name}',
    #                  job_type=f'{config.method.name}',
    #                  config=param_config)

    # wandb.run.name = f'{wandb.run.id}'
    # wandb.run.log_code(".")  # walks the current directory and save files that end with .py.
    # config_wandb = from_wandb_config(wandb.config)  # for parameter sweep
    # print(f'Param config = \n {pprint.pformat(config_wandb, indent=4)}')
    # wb_metric_epoch_step, wb_metric_test_acc, wb_metric_test_rob = define_wandb_epoch_metrics()

    model = choose_model(config.data, config.model)
    model = model.to(device)
    model.train()

    train_loader, test_loader, targets, num_classes = prepare_data(config.data, train_id=True, test_id=False)
    optimizer = optim.SGD(model.parameters(), lr=config.lr_init, momentum=config.momentum,
                          weight_decay=config.weight_decay)
    lr_scheduler = get_lr_scheduler(config.total_epoch, len(train_loader), config.lr_schedule_type,
                                    optimizer, config.lr_min, config.lr_max)
    trainer = get_trainer(config.method)
    trainer = trainer(config=config, device=device,
                      model=model, train_loader=train_loader, optimizer=optimizer, num_classes=num_classes,
                      targets=targets, lr_scheduler=lr_scheduler)
    tester = AdvTester(config.param_atk_eval, device, model, test_loader)

    # log = Logger(is_use_wandb=True)
    # log.log_to_file(f'Param config = \n {pprint.pformat(config_wandb, indent=4)}')

    acc_all = []
    rob_all = []
    best_acc = 0.
    logger.info('Epoch \t Seconds \t LR \t \t \t PGD Acc \t Real Acc')
    for i_epoch in range(1, config.total_epoch + 1):
        # log.log_to_file(f'epoch: {i_epoch}')
        start_epoch_time = time.time()
        trainer.train_epoch(i_epoch)
        epoch_time = time.time()
        acc, rob = tester.eval()

        adv_acc = tester.evaluate_pgd(10, 1)
        real_acc = tester.evaluate_standard()

        acc_all.append((acc.item(), real_acc))
        rob_all.append((rob.item(), adv_acc))

        lr = lr_scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
                    i_epoch, epoch_time - start_epoch_time, lr, adv_acc, real_acc)

        if best_acc < adv_acc:
            best_acc = adv_acc
            best_state_dict = copy.deepcopy(model.state_dict())
    torch.save(best_state_dict, os.path.join(config.output_path, f'model_best_our_{start_time}.pth'))
    torch.save(model.state_dict(), os.path.join(config.output_path, f'model_final_our_{start_time}.pth'))
    logger.info(acc_all)
    logger.info(rob_all)
    logger.info('\n')
    return best_acc


def main_main(alp, bet):
    def set_lr_param(cfg: Config, lr_type: str):
        if lr_type == 'step-wise':
            cfg.lr_init = 0.05
            cfg.lr_max = 0.1
            cfg.lr_min = 0.001
            cfg.lr_schedule_type = LrSchedulerType.Step
        elif lr_type == 'cyclic-wise':
            cfg.lr_init = 0.
            cfg.lr_max = 0.2
            cfg.lr_min = 0.0
            cfg.lr_schedule_type = LrSchedulerType.Cyclic
        else:
            raise ValueError


    config = Config(
        lr_init=0.1,
        lr_max=0.1,
        lr_min=0.001,
        momentum=0.9,
        weight_decay=2e-4,
        total_epoch=110,

        param_atk_train=ConfigLinfAttack(
            epsilon=8 / 255,
            perturb_steps=1,
            step_size=10 / 255
        ),
        param_atk_eval=ConfigLinfAttack(
            epsilon=8 / 255,
            perturb_steps=10,   # 50
            step_size=2 / 255),

        lr_schedule_type=LrSchedulerType.Cyclic,
        param_fixed_lam=6,
        param_step_size_range=ParamsStepSizeRange(
            step_size_min=2 / 255,
            step_size_max=20 / 255,
        ),
        param_lam_range=ParamsLamRange(
            lam_min=4,
            lam_max=12,
        ),
    )

    # config.model = ModelName.PreActResNet_18
    config.model = ModelName.ResNet_18
    config.seed = Seed.SEED_1

    config.data = DataName.CIFAR10
    method = MethodName.TradesUpdated
    config.method = method
    set_lr_param(config, methods_lr_schedule[method])
    config.output_path = f'results/cifar10/ResNet18_{methods_lr_schedule[method].split("-")[0]}/{str(method).split(".")[1]}/EACC5'
    config.alpha = alp
    config.beta = bet
    best_acc = main(config)
    return best_acc


if __name__ == '__main__':
    # 定义贝叶斯优化实例
    optimizer = BayesianOptimization(
        f=main_main,
        pbounds={
            'alp': (1.0, 2.0),
            'bet': (0.5, 1),
        },
        verbose=2,
        # verbose参数在optimize函数中控制着输出的优化信息的详细程度。当verbose=2时，输出每次迭代的详细信息，包括目标函数值、超参数取值、期望改进值、期望目标值、标准偏差等。当verbose=0时，优化过程中不输出任何信息0；当verbose=1时，输出每次迭代的目标函数值和超参数取值。
        random_state=1234,  # 控制随机数种子
    )

    # 运行优化器
    # 默认情况下，initial_points参数的默认值为 min(20, num_dims * 5)，
    # 其中num_dims表示超参数空间的维度。如果超参数空间的维度较小，则默认采样的初始点数为20，否则会根据维度进行缩放。
    optimizer.maximize(n_iter=15)

    # 输出最佳超参数组合和性能
    print('输出最佳超参数组合和性能: ', optimizer.max)
