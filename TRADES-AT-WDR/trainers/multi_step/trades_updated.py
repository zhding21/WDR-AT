import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.autograd import Variable
from tqdm import tqdm
from utils.helper_funcs_wandb import define_wandb_batch_metrics

from config import *


class WeightedKLDivLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, inputs, targets, keywords, lams):
        # 计算输入的对数概率
        log_probs = F.log_softmax(inputs, dim=1)
        # 计算KL散度，注意targets应为log概率
        targets_log_probs = torch.log(targets)
        kl_div = targets * (targets_log_probs - log_probs)
        # loss = torch.sum(kl_div, dim=1)
        loss = lams.cuda() * torch.sum(kl_div, dim=-1)

        if self.weights is not None:
            weights = torch.ones_like(loss)
            for i, keyword in enumerate(keywords):
                if keyword:
                    weights[i] = self.weights
            loss *= weights
        return torch.mean(loss)


class TradesUpdated:
    def __init__(self, config: Config, device: str, lr_scheduler,
                 model: torch.nn.Module, train_loader, optimizer, num_classes: int, targets, use_wandb=False, **kwargs):
        self.alpha = config.alpha
        self.criterion_our = WeightedKLDivLoss(config.beta)
        self.none_criterion = nn.CrossEntropyLoss(reduction='none')
        self.mean_criterion = nn.CrossEntropyLoss()

        self.model = model
        self.adv_config = config.param_atk_train
        self.optimizer = optimizer
        self.device = device
        self.total_epoch = config.total_epoch
        self.train_loader = train_loader
        self.num_samples = len(targets)
        self.num_cls = num_classes
        self.lr_scheduler = lr_scheduler

        self.epoch_start_collect = 3
        self.mometum_factor = 0.9

        self.probs_global = torch.zeros((self.num_samples,), dtype=torch.float)
        self.logits_current_epoch = torch.zeros((self.num_samples, num_classes))
        self.targets = torch.from_numpy(targets).type(torch.long)

        self.lam_min = config.param_lam_range.lam_min
        self.lam_max = config.param_lam_range.lam_max

        self.bs_print = 200
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch,
                             file=sys.stdout, position=0, ncols=120)

        self.use_wandb = use_wandb
        if self.use_wandb:  # use wandb to record
            self.wb_metric_batch, self.wb_metric_lr, self.wb_metric_epoch, self.wb_metric_loss = define_wandb_batch_metrics()

    def _assign_lam_to_samples(self, ):
        ids_sorted = torch.argsort(self.probs_global)
        lams_for_samples = torch.empty((self.num_samples,), dtype=torch.float)
        lams = torch.linspace(self.lam_min, self.lam_max, steps=self.num_samples)
        lams_for_samples[ids_sorted] = lams

        return lams_for_samples

    def _collect_global_probs(self):
        probs_all = F.softmax(self.logits_current_epoch, dim=-1)
        probs = probs_all[range(self.num_samples), self.targets]
        self.probs_global = self.mometum_factor * self.probs_global + (1 - self.mometum_factor) * probs

    def train_epoch(self, idx_epoch):
        """
        idx_epoch should start from 1
        """
        if idx_epoch >= self.epoch_start_collect:
            self._collect_global_probs()
            lams_rob_for_sample = self._assign_lam_to_samples()
        else:
            lams_rob_for_sample = torch.full((self.num_samples,), self.lam_min, dtype=torch.float)

        self.model.train()

        for batch_idx, (idx, data, target) in enumerate(self.train_loader, 1):
            self.optimizer.zero_grad()
            lr = self.lr_scheduler.get_last_lr()[0]

            lams = lams_rob_for_sample[idx]

            # calculate robust loss
            loss, logits_adv = self._trades_batch(
                x_natural=data.to(self.device), y=target.to(self.device),
                optimizer=self.optimizer,
                step_size=self.adv_config.step_size,
                epsilon=self.adv_config.epsilon,
                perturb_steps=self.adv_config.perturb_steps,
                lams=lams)

            if idx_epoch >= self.epoch_start_collect - 1:
                self.logits_current_epoch[idx] = logits_adv

            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            if batch_idx % self.bs_print == 0:
                self.tqdm_bar.write(f'[{idx_epoch:<2}, {batch_idx + 1:<5}] '
                                    f'Adv loss: {loss:<6.4f} '
                                    f'lr: {lr:.4f} ')

            self.tqdm_bar.update(1)
            self.tqdm_bar.set_description(f'epoch-{idx_epoch:<3} '
                                          f'batch-{batch_idx + 1:<3} '
                                          f'Adv loss-{loss:<.2f} '
                                          f'lr-{lr:.3f} '
                                          f'b-rob-{torch.mean(lams):.3f}')

            if self.use_wandb:
                idx_batch_total = (idx_epoch - 1) * len(self.train_loader) + batch_idx
                wandb.log({self.wb_metric_lr: lr, self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_loss: loss.item(), self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_epoch: idx_epoch, self.wb_metric_batch: idx_batch_total})

        if idx_epoch >= self.total_epoch:
            self.tqdm_bar.clear()
            self.tqdm_bar.close()

    def _trades_batch(self, x_natural, y, optimizer, step_size=0.007, epsilon=0.031, perturb_steps=10, lams=None):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='sum')
        self.model.eval()
        # generate adversarial example
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()

        with torch.no_grad():
            logits_nat = self.model(x_natural)

        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                       F.softmax(logits_nat, dim=1))

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()
        # calculate robust loss
        logits_nat = self.model(x_natural)
        loss_natural = F.cross_entropy(logits_nat, y)

        logits_adv = self.model(x_adv)
        # criterion_kl = nn.KLDivLoss(reduction='none')
        # loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
        #                            F.softmax(logits_nat, dim=1))
        # loss_robust = lams.cuda() * torch.sum(loss_robust, dim=-1)
        # loss_robust = torch.mean(loss_robust)
        keywords = torch.zeros(x_natural.size(0))
        adv_loss_mean = self.mean_criterion(logits_adv, y)
        adv_loss_none = self.none_criterion(logits_adv, y)
        self.model.eval()
        with torch.no_grad():
            real_output = self.model(x_natural)
            real_loss_mean = self.mean_criterion(real_output, y)
            real_loss_none = self.none_criterion(real_output, y)
        self.model.train()
        loss_diff_mean = (adv_loss_mean - real_loss_mean).abs() * self.alpha
        k = 0
        for adv_loss_val, real_loss_val in zip(adv_loss_none.tolist(), real_loss_none.tolist()):
            loss_diff_val = adv_loss_val - real_loss_val
            if loss_diff_mean < loss_diff_val:
                keywords[k] = 1
            k += 1
        loss_robust = self.criterion_our(F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1), keywords, lams)

        loss = loss_natural + loss_robust

        return loss, logits_adv.detach().cpu()
