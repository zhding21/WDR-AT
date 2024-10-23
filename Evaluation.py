import os
import argparse
import logging
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from autoattack.autoattack import AutoAttack

from utils.model_util import MODEL, MODEL_cspgi
from utils.util import attack_pgd, attack_fgsm, attack_cw, attack_bim, get_loaders, get_loaders_100, std


def test_model(class_model, test_data, attack_dict):
    real_acc, adv_acc, total = 0, 0, 0
    real_total_loss, adv_total_loss = 0, 0

    epsilon = (attack_dict['epsilon'] / 255.) / std
    alpha = (attack_dict['alpha'] / 255.) / std

    if attack_dict['attack_type'] == 'Auto-Attack':
        adversary = AutoAttack(class_model, norm='Linf', eps=epsilon, version='standard',
                               log_path="results/eval_auto_attack_comp.log")
        adversary.seed = 0

    class_model.eval()
    for i, (img, label) in enumerate(test_data):
        img, label = img.cuda(), label.cuda()
        if attack_dict['attack_type'] == 'PGD':
            delta = attack_pgd(class_model, img, label, epsilon, alpha, attack_dict['attack_iters'],
                               attack_dict['restarts'])
        elif attack_dict['attack_type'] == 'FGSM':
            delta = attack_fgsm(class_model, img, label, epsilon, alpha,  attack_dict['attack_iters'],
                                attack_dict['restarts'])
        elif attack_dict['attack_type'] == 'CW':
            delta = attack_cw(class_model, img, label, epsilon, alpha, attack_dict['attack_iters'],
                              attack_dict['restarts'])
        elif attack_dict['attack_type'] == 'BIM':
            delta = attack_bim(class_model, img, label, epsilon, alpha, attack_dict['attack_iters'],
                               attack_dict['restarts'])
        elif attack_dict['attack_type'] == 'Auto-Attack':
            img, label = Variable(img, requires_grad=True), Variable(label)
            X_pgd = adversary.run_standard_evaluation(img, label, bs=img.size(0))

        with torch.no_grad():
            real_output = class_model(img)
            real_loss = F.cross_entropy(real_output, label)
            real_total_loss += real_loss.item() * label.size(0)

            if attack_dict['attack_type'] == 'Auto-Attack':
                adv_output = class_model(X_pgd)
            else:
                adv_output = class_model(img + delta)
            adv_loss = F.cross_entropy(adv_output, label)
            adv_total_loss += adv_loss.item() * label.size(0)

            real_acc += (real_output.max(1)[1] == label).sum().item()
            adv_acc += (adv_output.max(1)[1] == label).sum().item()

            total += label.size(0)
    return adv_acc/total, adv_total_loss/total, real_acc/total, real_total_loss/total


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename='./results/CIFAR-10/Evaluation.log')
    _, test_loader = get_loaders('./data', 128)

    for at_model in ['FAT', 'PGD', 'FGSM', 'PGI']:
        logger.info(f'Adversary Train: {at_model}-ResNet18')    # ,
        for attack_val in [
            {'attack_type': 'PGD', 'epsilon': 8, 'alpha': 2, 'attack_iters': 10, 'restarts': 1},
            {'attack_type': 'PGD', 'epsilon': 8, 'alpha': 2, 'attack_iters': 20, 'restarts': 1},
            {'attack_type': 'PGD', 'epsilon': 8, 'alpha': 2, 'attack_iters': 50, 'restarts': 1},
            {'attack_type': 'FGSM', 'epsilon': 8, 'alpha': 8, 'attack_iters': 1, 'restarts': 1},
            {'attack_type': 'BIM', 'epsilon': 8, 'alpha': 2, 'attack_iters': 20, 'restarts': 1},
            {'attack_type': 'CW', 'epsilon': 8, 'alpha': 2, 'attack_iters': 20, 'restarts': 1},
            {'attack_type': 'Auto-Attack', 'epsilon': 8, 'alpha': 2, 'attack_iters': 0, 'restarts': 0}]:
            logger.info('\t\t\t\t\t Eval Attack Method')
            logger.info('\t\t attack_type: %s \t attack_iters: %d \t restarts: %d', attack_val['attack_type'],
                        attack_val['attack_iters'], attack_val['restarts'])
            for bf_val in ['best', 'final']:
                if at_model in ['PGI']:
                    comp_eval_model = MODEL_cspgi['ResNet18'](10).cuda()
                    comp_eval_model = torch.nn.DataParallel(comp_eval_model)
                    our_eval_model = MODEL_cspgi['ResNet18'](10).cuda()
                    our_eval_model = torch.nn.DataParallel(our_eval_model)
                else:
                    comp_eval_model = MODEL['ResNet18'](10).cuda()
                    our_eval_model = MODEL['ResNet18'](10).cuda()


                comp_model_path = f'./results_new/CIFAR-10/{at_model}/model_{bf_val}.pth'
                comp_eval_model.load_state_dict(torch.load(comp_model_path))
                comp_eval_model.float()
                comp_eval_model.eval()
                if at_model in ['Sub-Fast', 'Sub-PGD']:
                    our_model_path = f'./results_new/CIFAR-10/{at_model}/model_{bf_val}_our.pt'
                else:
                    our_model_path = f'./results_new/CIFAR-10/{at_model}/model_{bf_val}_our.pth'
                our_eval_model.load_state_dict(torch.load(our_model_path))
                our_eval_model.float()
                our_eval_model.eval()

                com_adv_acc, _, com_real_acc, _ = test_model(comp_eval_model, test_loader, attack_val)
                logger.info('Comp: \t %s \t\tTest Acc: %.4f \t PGD Acc: %.4f', bf_val, com_real_acc, com_adv_acc)

                our_adv_acc, _, our_real_acc, _ = test_model(our_eval_model, test_loader, attack_val)
                logger.info('Our: \t %s \t\tTest Acc: %.4f \t PGD Acc: %.4f', bf_val, our_real_acc, our_adv_acc)
            logger.info('\n')
        logger.info('\n\n')


if __name__ == "__main__":
    main()
