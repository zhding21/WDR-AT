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


logger = logging.getLogger(__name__)


def main(config: Config):

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(config.output_path, 'output.log'))
    logger.info(config)

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
    optimizer = optim.SGD(model.parameters(), lr=config.lr_init,
                          momentum=config.momentum, weight_decay=config.weight_decay)
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
    torch.save(best_state_dict, os.path.join(config.output_path, 'model_best.pth'))
    torch.save(model.state_dict(), os.path.join(config.output_path, 'model_final.pth'))
    logger.info(acc_all)
    logger.info(rob_all)


if __name__ == '__main__':
    def set_lr_param(cfg: Config, lr_type: str):
        if lr_type == 'step-wise':
            cfg.lr_init = 0.1
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
            perturb_steps=10,
            step_size=2 / 255
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
    # method = MethodName.FastUpdated
    method = MethodName.TradesUpdated
    config.method = method
    set_lr_param(config, methods_lr_schedule[method])
    config.output_path = f'results/cifar10/ResNet18_{methods_lr_schedule[method].split("-")[0]}/{str(method).split(".")[1]}/EACC'
    main(config)
