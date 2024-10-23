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

from utils.model_util import MODEL
from utils.util import (upper_limit, lower_limit, std, clamp, get_loaders, evaluate_pgd, evaluate_standard)

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
        filename=os.path.join(output_path, 'output_PGD.log'))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    def eval_model(alp, bet):
        model = MODEL['ResNet18'](10).cuda()
        model.train()

        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
        amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
        if args.opt_level == 'O2':
            amp_args['master_weights'] = args.master_weights
        # model, opt = amp.initialize(model, opt, **amp_args)
        criterion = nn.CrossEntropyLoss()

        lr_steps = args.epochs * len(train_loader)
        if args.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif args.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

        # Training
        best_acc = 0.
        start_train_time = time.time()

        # model_path = os.path.join(args.out_dir, 'model_AT-PGD')
        # if not os.path.exists(model_path):
        #     os.mkdir(model_path)

        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc \t PGD Acc \t Real Acc')
        for epoch in range(args.epochs):
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_loader):
                X, y = X.cuda(), y.cuda()
                delta = torch.zeros_like(X).cuda()
                if args.delta_init == 'random':
                    for i in range(len(epsilon)):
                        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                for _ in range(args.attack_iters):
                    output = model(X + delta)
                    loss = criterion(output, y)
                    # with amp.scale_loss(loss, opt) as scaled_loss:
                    #     scaled_loss.backward()
                    loss.backward()
                    grad = delta.grad.detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                    delta.grad.zero_()
                delta = delta.detach()
                output = model(X + delta)
                loss = criterion(output, y)
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
        torch.save(best_state_dict, os.path.join(output_path, 'model_best.pth'))
        torch.save(model.state_dict(), os.path.join(output_path, 'model_final.pth'))
        logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

        # Evaluation
        model_test = MODEL['ResNet18'](10).cuda()
        model_test.load_state_dict(model.state_dict())
        model_test.float()
        model_test.eval()

        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
        test_loss, test_acc = evaluate_standard(test_loader, model_test)

        logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
        logger.info('%.4f \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

    eval_model(1, 2)


if __name__ == "__main__":
    main()
