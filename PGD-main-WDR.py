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
from tqdm import tqdm

from bayes_opt import BayesianOptimization

from utils.model_util import MODEL
from utils.util import (upper_limit, lower_limit, std, clamp, get_loaders, evaluate_pgd,
                        evaluate_standard, WeightedCrossEntropyLoss)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='results', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    return parser.parse_args()


def main():
    args = get_args()

    output_path = os.path.join(args.out_dir, 'CIFAR-10', 'PGD')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, 'output_our.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    def eval_model(alp, bet):
        start_time = time.time()
        logger.info('start_time: %s; alp: %s; bet: %s', str(start_time), str(alp), str(bet))
        # model = PreActResNet18().cuda()
        model = MODEL['ResNet18'](10).cuda()
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
        amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
        if args.opt_level == 'O2':
            amp_args['master_weights'] = args.master_weights
        # model, opt = amp.initialize(model, opt, **amp_args)

        delta_criterion = nn.CrossEntropyLoss()
        criterion = WeightedCrossEntropyLoss(bet)
        none_criterion = nn.CrossEntropyLoss(reduction='none')
        mean_criterion = nn.CrossEntropyLoss()

        lr_steps = args.epochs * len(train_loader)
        if args.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif args.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

        # Training
        best_acc = 0.
        start_train_time = time.time()

        # model_path = os.path.join(args.out_dir, 'model_AT-PGD_Our_' + str(start_train_time))
        # if not os.path.exists(model_path):
        #     os.mkdir(model_path)

        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t PGD Acc \t Real Acc')
        for epoch in range(args.epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            with tqdm(train_loader, desc=f'{epoch+1}/{args.epochs}: train models') as data_iterator:
                for i, (X, y) in enumerate(data_iterator):
                    X, y = X.cuda(), y.cuda()
                    delta = torch.zeros_like(X).cuda()
                    if args.delta_init == 'random':
                        for i in range(len(epsilon)):
                            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.requires_grad = True
                    for _ in range(args.attack_iters):
                        output = model(X + delta)
                        loss = delta_criterion(output, y)
                        # with amp.scale_loss(loss, opt) as scaled_loss:
                        #     scaled_loss.backward()
                        loss.backward()
                        grad = delta.grad.detach()
                        delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                        delta.grad.zero_()
                    delta = delta.detach()

                    keywords = torch.zeros(X.size(0))
                    output = model(X + delta)

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
                    # loss = criterion(output, y)
                    loss = criterion(output, F.one_hot(y, num_classes=10), keywords)
                    opt.zero_grad()
                    # with amp.scale_loss(loss, opt) as scaled_loss:
                    #     scaled_loss.backward()
                    loss.backward()
                    opt.step()
                    train_loss += loss.item() * y.size(0)
                    train_acc += (output.max(1)[1] == y).sum().item()
                    train_n += y.size(0)
                    scheduler.step()
            epoch_time = time.time()

            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
            test_loss, test_acc = evaluate_standard(test_loader, model)

            # model_name = '{}-pgd_{:.4f}-real_{:.4f}.pth'.format(epoch, pgd_acc, test_acc)
            # torch.save(model.state_dict(), os.path.join(model_path, model_name))

            lr = scheduler.get_lr()[0]
            logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                        epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n,
                        pgd_acc, test_acc)

            if best_acc < pgd_acc:
                best_acc = pgd_acc
                best_state_dict = copy.deepcopy(model.state_dict())
        train_time = time.time()

        torch.save(best_state_dict, os.path.join(output_path, f'model_best_our_{start_time}.pth'))
        torch.save(model.state_dict(), os.path.join(output_path, f'model_final_our_{start_time}.pth'))
        logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

        # Evaluation
        model_test = MODEL['ResNet18'](10).cuda()
        model_test.load_state_dict(best_state_dict)
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        logger.info('--Test Loss \t Test Acc \t PGD Loss \t PGD Acc \t time \t alp \t bet')
        logger.info('%.4f \t \t %.4f \t %.4f \t %.4f \t %d \t %.8f \t %.8f \n\n',
                    test_loss, test_acc, pgd_loss, pgd_acc, start_train_time, alp, bet)
        return best_acc

    # 定义贝叶斯优化实例
    optimizer = BayesianOptimization(
        f=eval_model,
        pbounds={
            'alp': (1.0, 2.0),
            'bet': (0.5, 1),
        },
        verbose=2,
        random_state=1234,
    )

    # 运行优化器
    optimizer.maximize(n_iter=15)

    # 输出最佳超参数组合和性能
    print('输出最佳超参数组合和性能: ', optimizer.max)


if __name__ == "__main__":
    main()
