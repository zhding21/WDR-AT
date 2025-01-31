import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
from config import *
import torch.nn.functional as F


def eval_pgd_batch(model, X, y,
                   epsilon=8 / 255, perturb_steps=20, step_size=2 / 255, random_start=True):
    out = model(X)
    corr_nat = (out.data.max(1)[1] == y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random_start:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(perturb_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    corr_pgd = (model(X_pgd).data.max(1)[1] == y.data).float().sum()
    return corr_nat.cpu(), corr_pgd.cpu()


def eval_pgd_batch_split(model, X, y,
                         epsilon=0.031, perturb_steps=20, step_size=0.003, random_start=True):
    out = model(X)
    _, predicted = torch.max(out.data, 1)
    corr_nat = (predicted == y.data).cpu().float().sum()
    is_correct_nat = (predicted == y).cpu()
    X_pgd = Variable(X.data, requires_grad=True)
    if random_start:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda()
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(perturb_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    out_adv = model(X_pgd)
    _, predicted_adv = torch.max(out_adv.data, 1)
    corr_pgd = (predicted_adv == y.data).cpu().float().sum()

    is_correct_adv = (predicted_adv == y).cpu()
    return is_correct_nat, is_correct_adv, corr_nat, corr_pgd

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


def normalize(X):
    return (X - mu) / std


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for zz in range(restarts):
        delta = torch.zeros_like(X).cuda()
        for i in range(len(epsilon)):
            delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
        delta.data = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)
            index = torch.where(output.max(1)[1] == y)
            if len(index[0]) == 0:
                break
            loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index[0], :, :, :]
            g = grad[index[0], :, :, :]
            d = clamp(d + alpha * torch.sign(g), -epsilon, epsilon)
            d = clamp(d, lower_limit - X[index[0], :, :, :], upper_limit - X[index[0], :, :, :])
            delta.data[index[0], :, :, :] = d
            delta.grad.zero_()
        all_loss = F.cross_entropy(model(X + delta), y, reduction='none').detach()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


class AdvTester:
    def __init__(self, adv_config: ConfigLinfAttack, device: str, model: torch.nn.Module, test_loader):
        self.adv_config = adv_config
        self.device = device
        self.model = model
        self.test_loader = test_loader

    def eval(self):
        robust_corr_total, natural_corr_total = 0, 0
        acc, rob = 0, 0
        count = 0

        self.model.eval()

        iterator_tqdm = tqdm(self.test_loader, file=sys.stdout, position=0, ncols=120)

        for data, target in iterator_tqdm:
            data, target = data.to(self.device), target.to(self.device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            corr_natural, corr_robust = eval_pgd_batch(self.model, X, y,
                                                       self.adv_config.epsilon,
                                                       self.adv_config.perturb_steps,
                                                       self.adv_config.step_size,
                                                       random_start=True)
            robust_corr_total += corr_robust
            natural_corr_total += corr_natural
            count += len(target)

            acc = natural_corr_total / count
            rob = robust_corr_total / count

            iterator_tqdm.set_description_str(f'On {count} examples. '
                                              f'Acc:{acc:.2%} | '
                                              f'Rob:{rob:.2%}')
        iterator_tqdm.close()

        return acc, rob

    def evaluate_standard(self):
        test_acc = 0
        n = 0
        self.model.eval()
        with torch.no_grad():
            for i, (X, y) in enumerate(self.test_loader):
                X, y = X.cuda(), y.cuda()
                output = self.model(X)
                test_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
        return test_acc / n

    def evaluate_pgd(self, attack_iters, restarts, epsilon=(8 / 255.) / std):
        alpha = (2 / 255.) / std
        pgd_acc = 0
        n = 0
        self.model.eval()
        for i, (X, y) in enumerate(self.test_loader):
            X, y = X.cuda(), y.cuda()
            pgd_delta = attack_pgd(self.model, X, y, epsilon, alpha, attack_iters, restarts)
            with torch.no_grad():
                output = self.model(normalize(X + pgd_delta))
                pgd_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
        return pgd_acc / n

    def eval_split(self, ids_AC, ids_RC):
        def get_split_acc(ids_all, correct_all):
            ids_all = np.concatenate(ids_all)
            correct_all = np.concatenate(correct_all)
            mask_rob = np.in1d(ids_all, ids_RC)
            mask_non_rob = np.in1d(ids_all, ids_AC)
            acc_RC = np.sum(correct_all[mask_rob]) / len(ids_RC)
            acc_AC = np.sum(correct_all[mask_non_rob]) / len(ids_AC)

            return acc_RC, acc_AC

        natural_corr_total, robust_corr_total = 0, 0
        total_acc, total_rob = 0, 0
        count = 0

        self.model.eval()

        ids_all = []
        correct_all_nat = []
        correct_all_adv = []

        iterator_tqdm = tqdm(self.test_loader, file=sys.stdout, position=0, ncols=120)

        for ids, data, target in iterator_tqdm:
            ids_all.append(ids)

            data, target = data.to(self.device), target.to(self.device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            is_correct_nat, is_correct_adv, corr_natural, corr_robust \
                = eval_pgd_batch_split(self.model, X, y,
                                       self.adv_config.epsilon,
                                       self.adv_config.perturb_steps,
                                       self.adv_config.step_size,
                                       random_start=True)
            natural_corr_total += corr_natural
            robust_corr_total += corr_robust
            correct_all_nat.append(is_correct_nat)
            correct_all_adv.append(is_correct_adv)
            count += len(target)

            total_acc = natural_corr_total / count
            total_rob = robust_corr_total / count

            iterator_tqdm.set_description_str(f'On {count} examples. '
                                              f'Acc:{total_acc:.2%} | '
                                              f'Rob:{total_rob:.2%}')
        acc_on_RC, acc_on_AC = get_split_acc(ids_all, correct_all_nat)
        rob_on_RC, rob_on_AC = get_split_acc(ids_all, correct_all_adv)
        iterator_tqdm.write(f'RC test data: Acc:{acc_on_RC:.2%} Rob:{rob_on_RC:.2%} | '
                            f'AC test data: Acc:{acc_on_AC:.2%} Rob:{rob_on_AC:.2%}')
        iterator_tqdm.close()

        return total_acc, total_rob, acc_on_RC, acc_on_AC, rob_on_RC, rob_on_AC


class NatTester:
    def __init__(self, model: torch.nn.Module, device: str, test_loader):
        self.device = device
        self.model = model
        self.test_loader = test_loader

    def eval(self):
        correct = 0
        total = 0

        iterator_tqdm = tqdm(self.test_loader, file=sys.stdout, position=0, ncols=120)

        with torch.no_grad():
            for i, test_batch in enumerate(iterator_tqdm):
                inputs = test_batch[0].to(self.device)
                labels = test_batch[1].to(self.device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                iterator_tqdm.set_description_str(f'Test on {total} examples. '
                                                  f'Natural acc-{correct / total:.2%}')
        iterator_tqdm.close()

        return correct / total
