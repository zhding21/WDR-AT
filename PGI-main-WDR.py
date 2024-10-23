import argparse
import copy
import logging
import os
import time
import numpy as np
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization

from utils.model_util import MODEL
from utils.util import upper_limit, lower_limit, std, clamp, get_all_loaders, evaluate_pgd, evaluate_standard

from torch.nn import functional as F
from torch.autograd import Variable


logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='data', type=str)
    parser.add_argument('--epochs', default=110, type=int)
    parser.add_argument('--epochs_reset', default=40, type=int)
    parser.add_argument('--lr_schedule', default='cyclic', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--epsilon_y', default=0.1, type=float)
    parser.add_argument('--alpha_y', default=0.1, type=float, help='Step size')
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous', 'normal'],
                        help='Perturbation initialization method')
    parser.add_argument('--normal_mean', default=0, type=float, help='normal_mean')
    parser.add_argument('--normal_std', default=1, type=float, help='normal_std')
    parser.add_argument('--out-dir', default='results', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    parser.add_argument('--factor', default=0.6, type=float, help='Label Smoothing')
    parser.add_argument('--lamda', default=10, type=float, help='Label Smoothing')
    parser.add_argument('--momentum_decay', default=0.3, type=float, help='momentum_decay')

    return parser.parse_args()


def _label_smoothing(label, factor):
    one_hot = np.eye(10)[label.cuda().data.cpu().numpy()]

    result = one_hot * factor + (one_hot - 1.) * ((factor - 1) / float(10 - 1))

    return result


class WeightedLabelSmoothLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, targets, keywords):
        log_probs = F.log_softmax(inputs, dim=1)
        loss = (-targets * log_probs).sum(dim=-1)
        if self.weights is not None:
            weights = torch.ones_like(loss)
            for i, keyword in enumerate(keywords):
                if keyword:
                    weights[i] = self.weights
            loss *= weights
        return torch.mean(loss)

def LabelSmoothLoss(input, target):
    log_prob = F.log_softmax(input, dim=-1)
    loss = (-target * log_prob).sum(dim=-1).mean()
    return loss


# upper_limit_y = 1
# lower_limit_y = 0


def main():
    args = get_args()

    output_path = os.path.join(args.out_dir, 'CIFAR-10', 'PGI')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(output_path, 'output_our.log'))
    logger.info(args)
    logger.info('Model Architecture: %s', model_val)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_all_loaders(args.data_dir, args.batch_size)
    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std

    def eval_model(alp, bet):
        start_time = time.time()
        logger.info('start_time: %s; alp: %s; bet: %s', str(start_time), str(alp), str(bet))
        model = MODEL['ResNet18'](10).cuda()
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                              weight_decay=args.weight_decay)

        criterion = WeightedLabelSmoothLoss(bet)
        none_criterion = nn.CrossEntropyLoss(reduction='none')
        mean_criterion = nn.CrossEntropyLoss()

        num_of_example = 50000
        batch_size = args.batch_size

        iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)

        lr_steps = args.epochs * iter_num
        if args.lr_schedule == 'cyclic':
            scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                          step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
        elif args.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                             milestones=[lr_steps * 100 / 110, lr_steps * 105 / 110],
                                                             gamma=0.1)
        # Training
        logger.info('Epoch \t Seconds \t LR \t Train Loss \t Train Acc \t PGD Acc \t Real Acc')
        best_acc = 0.
        start_train_time = time.time()

        # model_path = os.path.join(args.out_dir, 'model_AT-PGI_Our_' + str(start_train_time))
        # if not os.path.exists(model_path):
        #     os.mkdir(model_path)

        for i, (X, y) in enumerate(train_loader):
            cifar_x, cifar_y = X.cuda(), y.cuda()

        import random
        def atta_aug(input_tensor, rst):
            batch_size = input_tensor.shape[0]
            x = torch.zeros(batch_size)
            y = torch.zeros(batch_size)
            flip = [False] * batch_size

            for i in range(batch_size):
                flip_t = bool(random.getrandbits(1))
                x_t = random.randint(0, 8)
                y_t = random.randint(0, 8)
                rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + 32, y_t:y_t + 32]
                if flip_t:
                    rst[i] = torch.flip(rst[i], [2])
                flip[i] = flip_t
                x[i] = x_t
                y[i] = y_t
            return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}

        for epoch in range(args.epochs):
            print(f'{epoch + 1}/{args.epochs}, alp: {alp}; bet: {bet}')
            batch_size = args.batch_size
            cur_order = np.random.permutation(num_of_example)
            iter_num = num_of_example // batch_size + (0 if num_of_example % batch_size == 0 else 1)
            batch_idx = -batch_size
            start_epoch_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            if epoch % args.epochs_reset == 0:
                temp = torch.rand(50000, 3, 32, 32)
                if args.delta_init != 'previous':
                    all_delta = torch.zeros_like(temp).cuda()
                    all_momentum = torch.zeros_like(temp).cuda()
                if args.delta_init == 'random':
                    for j in range(len(epsilon)):
                        all_delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    all_delta.data = clamp(alpha * torch.sign(all_delta), -epsilon, epsilon)
            idx = torch.randperm(cifar_x.shape[0])

            cifar_x = cifar_x[idx, :, :, :].view(cifar_x.size())

            cifar_y = cifar_y[idx].view(cifar_y.size())
            all_delta = all_delta[idx, :, :, :].view(all_delta.size())
            all_momentum = all_momentum[idx, :, :, :].view(all_delta.size())

            for i in range(iter_num):
                batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < num_of_example else 0
                X = cifar_x[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
                y = cifar_y[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
                delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()
                next_delta = all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()

                momentum = all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]].clone().detach()

                X, Y = X.cuda(), y.cuda()
                batch_size = X.shape[0]
                rst = torch.zeros(batch_size, 3, 32, 32).cuda()
                X, transform_info = atta_aug(X, rst)

                label_smoothing = Variable(torch.tensor(_label_smoothing(y, args.factor)).cuda()).float()

                delta.requires_grad = True

                ori_output = model(X + delta[:X.size(0)])

                ori_loss = LabelSmoothLoss(ori_output, label_smoothing.float())

                decay = args.momentum_decay

                ori_loss.backward(retain_graph=True)
                x_grad = delta.grad.detach()

                grad_norm = torch.norm(x_grad, p=1)
                momentum = x_grad / grad_norm + momentum * decay

                next_delta.data = clamp(delta + alpha * torch.sign(momentum), -epsilon, epsilon)
                next_delta.data[:X.size(0)] = clamp(next_delta[:X.size(0)], lower_limit - X, upper_limit - X)

                delta.data = clamp(delta + alpha * torch.sign(x_grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta = delta.detach()

                keywords = torch.zeros(X.size(0))
                output = model(X + delta[:X.size(0)])

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

                loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)
                # loss = LabelSmoothLoss(output, (label_smoothing).float()) +
                # args.lamda * loss_fn(output.float(), ori_output.float())
                loss = criterion(output, (label_smoothing).float(), keywords) + \
                       args.lamda * loss_fn(output.float(), ori_output.float())

                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
                scheduler.step()

                all_momentum[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = momentum
                all_delta[cur_order[batch_idx:min(num_of_example, batch_idx + batch_size)]] = next_delta

            epoch_time = time.time()
            lr = scheduler.get_lr()[0]

            pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
            test_loss, test_acc = evaluate_standard(test_loader, model)

            # model_name = '{}-pgd_{:.4f}-real_{:.4f}.pth'.format(epoch, pgd_acc, test_acc)
            # torch.save(model.state_dict(), os.path.join(model_path, model_name))

            logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                        epoch, epoch_time - start_epoch_time, lr, train_loss / train_n, train_acc / train_n,
                        pgd_acc, test_acc)

            if best_acc < pgd_acc:
                best_acc = pgd_acc
                best_state_dict = copy.deepcopy(model.state_dict())
        train_time = time.time()

        torch.save(best_state_dict, os.path.join(output_path, f'model_best_our_{start_time}.pth'))
        torch.save(model.state_dict(), os.path.join(output_path, f'model_final_our_{start_time}.pth'))
        logger.info('Total train time: %.4f minutes', (train_time - start_train_time) / 60)

        # Evaluation
        # model_test = ResNet18().cuda()
        model_test = MODEL['ResNet18'](10).cuda()
        # model_test = torch.nn.DataParallel(model_test)
        # model_test = model_test.cuda()
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
