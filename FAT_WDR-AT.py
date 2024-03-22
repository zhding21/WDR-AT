import argparse
import copy
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bayes_opt import BayesianOptimization

from model import ResNet18
from util import upper_limit, lower_limit, std, clamp, get_loaders, evaluate_pgd, evaluate_standard, WeightedCrossEntropyLoss, get_weight

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='./cifar-data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='FAT-FGSM-ResNet18', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
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
    alpha = (args.alpha / 255.) / std

    def BO_train(alp, bet):
        model = ResNet18().cuda()
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                              weight_decay=args.weight_decay)
        # criterion = nn.CrossEntropyLoss()
        criterion = WeightedCrossEntropyLoss(bet)
        none_criterion = nn.CrossEntropyLoss(reduction='none')
        mean_criterion = nn.CrossEntropyLoss()

        if args.delta_init == 'previous':
            delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()

        lr_steps = args.epochs * len(train_loader)
        if args.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                          step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif args.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4],
                                                             gamma=0.1)

        best_acc = 0.
        start_train_time = time.time()

        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t PGD Acc \t Real Acc')
        for epoch in range(args.epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                # if i == 0:
                #     first_batch = (X, y)
                if args.delta_init != 'previous':
                    delta = torch.zeros_like(X).cuda()
                if args.delta_init == 'random':
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta = delta.detach()

                keywords = torch.zeros(X.size(0))
                output = model(X + delta[:X.size(0)])
                adv_loss_mean = mean_criterion(output, y)
                adv_loss_none = none_criterion(output, y)
                weight = get_weight(keywords, model, X, X + delta[:X.size(0)], y)
                loss = criterion(output, F.one_hot(y, num_classes=10), weight)
                opt.zero_grad()

                loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                scheduler.step()
            epoch_time = time.time()

            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
            test_loss, test_acc = evaluate_standard(test_loader, model)

            lr = scheduler.get_lr()[0]
            logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f', epoch, epoch_time - start_epoch_time,
                        lr, train_loss / train_n, train_acc / train_n, pgd_acc, test_acc)

            if best_acc < pgd_acc:
                best_acc = pgd_acc
                best_state_dict = copy.deepcopy(model.state_dict())
        train_time = time.time()
        torch.save(best_state_dict, f'{args.out_dir}/model_FAT_Best_our_{start_train_time}.pth')
        torch.save(model.state_dict(), f'{args.out_dir}/model_FAT_Final_our_{start_train_time}.pth')

        logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

        model_test = ResNet18().cuda()
        model_test.load_state_dict(best_state_dict)
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t time \t alp \t bet')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %d \t %.8f \t %.8f \n\n',
                    test_loss, test_acc, pgd_loss, pgd_acc, start_train_time, alp, bet)
        return pgd_acc

    optimizer = BayesianOptimization(
        f=BO_train,
        pbounds={
            'alp': (1.0, 2.0),
            'bet': (0.5, 1),
        },
        verbose=2,
        random_state=1234,
    )
    optimizer.maximize(n_iter=15)
    print('输出最佳超参数组合和性能: ', optimizer.max)


if __name__ == "__main__":
    main()
