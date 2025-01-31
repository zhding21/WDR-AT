import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.autograd import Variable
from tqdm import tqdm

from config import Config
from utils.helper_funcs_wandb import define_wandb_batch_metrics, WeightedCrossEntropyLoss


class FastUpdated:
    def __init__(self, config: Config, device: str, lr_scheduler,
                 model: torch.nn.Module, train_loader, optimizer,
                 num_classes: int, targets, use_wandb=False,
                 **kwargs):
        self.alpha = config.alpha
        self.criterion_our = WeightedCrossEntropyLoss(config.beta)
        self.none_criterion = nn.CrossEntropyLoss(reduction='none')
        self.mean_criterion = nn.CrossEntropyLoss()

        self.model = model
        self.adv_config = config.param_atk_train
        self.optimizer = optimizer
        self.device = device

        self.total_epoch = config.total_epoch
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler

        self.epoch_start_collect = 3
        self.mometum_factor = 0.9

        self.num_samples = len(targets)
        self.num_cls = num_classes
        self.probs_global = torch.zeros((self.num_samples,), dtype=torch.float)
        self.logits_current_epoch = torch.zeros((self.num_samples, num_classes))
        self.targets = torch.from_numpy(targets).type(torch.long)

        self.step_size_min = config.param_step_size_range.step_size_min
        self.step_size_max = config.param_step_size_range.step_size_max

        self.bs_print = 200
        self.tqdm_bar = tqdm(total=len(self.train_loader) * self.total_epoch, file=sys.stdout, position=0, ncols=120)

        self.use_wandb = use_wandb
        if self.use_wandb:  # use wandb to record
            self.wb_metric_batch, self.wb_metric_lr, self.wb_metric_epoch, self.wb_metric_loss = define_wandb_batch_metrics()

    def _assign_step_size_to_samples(self):
        ids_sorted = torch.argsort(self.probs_global)  # probs rise
        step_sizes_for_samples = torch.empty((self.num_samples,), dtype=torch.float)
        step_sizes = torch.linspace(self.step_size_min, self.step_size_max, steps=self.num_samples)
        step_sizes_for_samples[ids_sorted] = step_sizes

        return step_sizes_for_samples

    def _collect_global_probs(self):
        probs_all = F.softmax(self.logits_current_epoch, dim=-1)
        probs = probs_all[range(self.num_samples), self.targets]
        self.probs_global = self.mometum_factor * self.probs_global + (1 - self.mometum_factor) * probs

    def train_epoch(self, idx_epoch):
        " idx_epoch should start from 1"
        if idx_epoch < self.epoch_start_collect:
            step_sizes_for_samples = torch.full((self.num_samples,), fill_value=self.step_size_min)
        else:
            self._collect_global_probs()
            step_sizes_for_samples = self._assign_step_size_to_samples()

        self.model.train()

        for batch_idx, (idx, data, target) in enumerate(self.train_loader, 1):
            data, target = data.to(self.device), target.to(self.device)

            lr = self.lr_scheduler.get_last_lr()[0]

            self.optimizer.zero_grad()
            step_sizes_batch = step_sizes_for_samples[idx]

            # calculate robust loss
            loss, logits_adv = self._train_batch(x_natural=data, y=target, step_sizes_batch=step_sizes_batch)

            if self.epoch_start_collect - 1 <= idx_epoch:
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
                                          f'step-size-mean-{step_sizes_batch.mean().item() * 255:.2f}')

            if self.use_wandb:
                idx_batch_total = (idx_epoch - 1) * len(self.train_loader) + batch_idx
                wandb.log({self.wb_metric_lr: lr, self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_loss: loss.item(), self.wb_metric_batch: idx_batch_total})
                wandb.log({self.wb_metric_epoch: idx_epoch, self.wb_metric_batch: idx_batch_total})

        if idx_epoch >= self.total_epoch:
            self.tqdm_bar.clear()
            self.tqdm_bar.close()

    def _train_batch(self, x_natural, y, step_sizes_batch):

        # define CE-loss
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.model.eval()
        batch_size = len(x_natural)
        step_sizes_batch = step_sizes_batch.view(batch_size, 1, 1, 1).cuda()
        # generate adversarial example
        x_adv = x_natural.detach() + torch.randn_like(x_natural).uniform_(-self.adv_config.epsilon,
                                                                          self.adv_config.epsilon)

        x_adv.requires_grad_()
        with torch.enable_grad():
            loss = criterion(self.model(x_adv), y)
        grad = torch.autograd.grad(loss, [x_adv])[0]
        x_adv = x_adv.detach() + step_sizes_batch * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural
                                    - self.adv_config.epsilon), x_natural + self.adv_config.epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

        self.model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        self.optimizer.zero_grad()
        # calculate robust loss
        logits_adv = self.model(x_adv)

        # our method
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
        loss = self.criterion_our(logits_adv, F.one_hot(y, num_classes=10), keywords)
        # loss = F.cross_entropy(logits_adv, y)

        return loss, logits_adv.detach().cpu()
