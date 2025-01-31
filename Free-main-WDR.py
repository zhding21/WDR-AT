import argparse
import logging
import os
import time
import copy
# import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bayes_opt import BayesianOptimization

from utils.model import ResNet18
from utils.util import (upper_limit, lower_limit, std, clamp, get_loaders, evaluate_pgd,
                        evaluate_standard, WeightedCrossEntropyLoss)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.04, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='FAT-FREE-ResNet18', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output_FAT_Our.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std

    def eval_model(alp, bet):
        model = ResNet18().cuda()
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)


        if args.opt_level == 'O2':
            amp_args['master_weights'] = args.master_weights
        # model, opt = amp.initialize(model, opt, **amp_args)

        # criterion = nn.CrossEntropyLoss()
        criterion = WeightedCrossEntropyLoss(bet)
        none_criterion = nn.CrossEntropyLoss(reduction='none')
        mean_criterion = nn.CrossEntropyLoss()

        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
        delta.requires_grad = True

        lr_steps = args.epochs * len(train_loader) * args.minibatch_replays
        if args.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif args.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

        # Training
        best_acc = 0.
        start_train_time = time.time()

        model_path = os.path.join(args.out_dir, 'model_FAT_Our_' + str(start_train_time))
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t PGD Acc \t Real Acc')

        for epoch in range(args.epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                for _ in range(args.minibatch_replays):
                    output = model(X + delta[:X.size(0)])
                    keywords = torch.zeros(X.size(0))
                    # our method
                    adv_loss_mean = mean_criterion(output, y)
                    adv_loss_none = none_criterion(output, y)

                    model.eval()
                    with torch.no_grad():
                        real_output = model(X)
                        real_loss_mean = mean_criterion(real_output, y)
                        real_loss_none = none_criterion(real_output, y)
                    model.train()

                    loss_diff_mean = (adv_loss_mean - real_loss_mean).abs() * alp
                    k = 0
                    for adv_loss_val, real_loss_val in zip(adv_loss_none.tolist(), real_loss_none.tolist()):
                        loss_diff_val = adv_loss_val - real_loss_val
                        if loss_diff_mean < loss_diff_val:
                            keywords[k] = 1
                        k += 1

                    loss = criterion(output, F.one_hot(y, num_classes=10), keywords)
                    opt.zero_grad()
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                    delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)

                    opt.step()
                    delta.grad.zero_()
                    scheduler.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            epoch_time = time.time()
            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
            test_loss, test_acc = evaluate_standard(test_loader, model)

            model_name = '{}-pgd_{:.4f}-real_{:.4f}.pth'.format(epoch, pgd_acc, test_acc)
            torch.save(model.state_dict(), os.path.join(model_path, model_name))

            lr = scheduler.get_lr()[0]
            logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                        epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n,
                        pgd_acc, test_acc)

            if best_acc < pgd_acc:
                best_acc = pgd_acc
                best_state_dict = copy.deepcopy(model.state_dict())

        train_time = time.time()
        torch.save(best_state_dict, os.path.join(args.out_dir, 'model_FAT_Our_Best_' + str(start_train_time) + '.pth'))
        torch.save(model.state_dict(),
                   os.path.join(args.out_dir, 'model_FAT_Our_Final_' + str(start_train_time) + '.pth'))

        logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

        # Evaluation
        model_test = ResNet18().cuda()
        model_test.load_state_dict(best_state_dict)
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t time \t alp \t bet')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %d \t %.8f \t %.8f \n\n',
                    test_loss, test_acc, pgd_loss, pgd_acc, start_train_time, alp, bet)
        return pgd_acc

    # 定义贝叶斯优化实例
    optimizer = BayesianOptimization(
        f=eval_model,
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


if __name__ == "__main__":
    main()
