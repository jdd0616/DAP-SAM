import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, sigmoid_rampup
from torchvision import transforms
from icecream import ic
from val import test_single_PROMPT, val_single_PROMPT
from datasets.dataset_RETOUCH import RETOUCH_SSL_dataset, RandomGenerator

from PIL import Image
from skimage.measure import label


def sharpening(P):
    T = 1/ 0.1
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen


def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    # for the first branch
    low_res_logits1 = outputs['low_res_logits1']
    loss_ce1 = ce_loss(low_res_logits1, low_res_label_batch[:].long())
    loss_dice1 = dice_loss(low_res_logits1, low_res_label_batch, softmax=True)
    loss1 = (1 - dice_weight) * loss_ce1 + dice_weight * loss_dice1

    # for the second branch
    low_res_logits2 = outputs['low_res_logits2']
    loss_ce2 = ce_loss(low_res_logits2, low_res_label_batch[:].long())
    loss_dice2 = dice_loss(low_res_logits2, low_res_label_batch, softmax=True)
    loss2 = (1 - dice_weight) * loss_ce2 + dice_weight * loss_dice2

    loss = loss1 + loss2
    return loss, loss1, loss_ce1, loss_dice1, loss2, loss_ce2, loss_dice2


def calc_loss_labeled(low_res_logits, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8):
    # low_res_logits_labeled = low_res_logits[:labeled_bs]
    # low_res_label_batch_labeled = low_res_label_batch[:labeled_bs]
    low_res_logits_labeled = low_res_logits
    low_res_label_batch_labeled = low_res_label_batch

    loss_ce = ce_loss(low_res_logits_labeled, low_res_label_batch_labeled[:].long())
    loss_dice = dice_loss(low_res_logits_labeled, low_res_label_batch_labeled, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def get_current_consistency_weight(epoch, consistency, max_iter):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    consistency_rampup = int(max_iter * 200.0 / 30000.0)
    return consistency * sigmoid_rampup(epoch, consistency_rampup)



def trainer_dualbrunch_onlylabl_normalassisted(args, model, snapshot_path, multimask_output, low_res, device):

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes

    if args.dataset != 'RETOUCH':
        raise ValueError('This release supports the RETOUCH dataset only.')
    db_train = RETOUCH_SSL_dataset(mode='labeled', split="train", num_classes=args.num_classes, ratio=args.ratio,
                                       base_dir=args.root_path,
                                       transform=transforms.Compose(
                                           [RandomGenerator(output_size=[args.img_size, args.img_size],
                                                            low_res=[low_res, low_res])]))
    db_val = RETOUCH_SSL_dataset(mode='labeled', split="test", num_classes=args.num_classes,
                                 ratio=args.ratio, base_dir=args.root_path)


    train_labeled_loader = DataLoader(db_train, batch_size= args.batch_size, shuffle=True)
    # train_unlabeled_loader = DataLoader(db_unlabeled, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(db_val, batch_size=args.batch_size, shuffle=False)

    print("The length of train set is: {}".format(len(db_train)))

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999),
                                weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9,
                              weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(train_labeled_loader)  # max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for labeled_batch in (tqdm(train_labeled_loader, desc=f'Epoch {epoch_num + 1}/{max_epoch}', unit='batch')):

            image_batch, label_batch = labeled_batch['image'], labeled_batch['label']  # [b, c, h, w], [b, h, w]
            # image_batch_unlabel, label_batch_unlabel = unlabeled_batch['image'], unlabeled_batch['label']  # [b, c, h, w], [b, h, w]
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            # image_batch_unlabel, label_batch_unlabel = image_batch_unlabel.to(device), label_batch_unlabel.to(device)
            PSEUDO_MASK = labeled_batch['PSEUDO_MASK'].to(device)

            outputs = model(image_batch, multimask_output, args.img_size, 'train', PSEUDO_MASK, 'cross', args.promptmode)

            outputs1 = outputs['low_res_logits1']
            outputs2 = outputs['low_res_logits2']

            supervised_loss1, loss_ce1, loss_dice1 = calc_loss_labeled(outputs1, label_batch, ce_loss, dice_loss,args.dice_param)
            supervised_loss2, loss_ce2, loss_dice2 = calc_loss_labeled(outputs2, label_batch, ce_loss, dice_loss, args.dice_param)
            loss1 = supervised_loss1 + supervised_loss2
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()

            consistency_weight = args.coe
            T = args.T

            # the second round
            if iter_num < args.warm_iter:
                supervised_round2_loss1, loss_round2_ce1, loss_round2_dice1 = 0.0, 0.0, 0.0
                supervised_round2_loss1_r, loss_round2_ce1_r, loss_round2_dice1_r = 0.0, 0.0, 0.0
            else:
                outputs_round2 = model(image_batch, multimask_output, args.img_size, 'train', PSEUDO_MASK, 'cross', args.promptmode)
                outputs_round2_1 = outputs_round2['low_res_logits1']
                outputs_round2_2 = outputs_round2['low_res_logits2']

                supervised_round2_loss1, loss_round2_ce1, loss_round2_dice1 = calc_loss_labeled(outputs_round2_1,label_batch, ce_loss,dice_loss, args.dice_param)

                supervised_round2_loss2, loss_round2_ce1_r, loss_round2_dice1_r = calc_loss_labeled(
                    outputs_round2_2, label_batch, ce_loss, dice_loss, args.dice_param)

            if iter_num < args.warm_iter:
                consistency_loss2 = 0.0
                consistency_loss1_r = 0.0
                loss2 = 0.0
            else:
                outputs_round2_label = model(image_batch, multimask_output, args.img_size, 'train', PSEUDO_MASK, 'cross', args.promptmode)
                outputs_round2_label_1 = outputs_round2_label['low_res_logits1']
                outputs_round2_label_2 = outputs_round2_label['low_res_logits2']
                outputs_round2_label_soft1 = torch.softmax(outputs_round2_label_1, dim=1)
                outputs_round2_label_soft2 = torch.softmax(outputs_round2_label_2, dim=1)

                sharpen_2 = sharpening(outputs_round2_label_soft2)
                sharpen_1 = sharpening(outputs_round2_label_soft1)
                consistency_loss2 = 0.5 * mse_loss(sharpen_2, outputs_round2_label_soft1)
                consistency_loss1 = 0.5 * mse_loss(sharpen_1, outputs_round2_label_soft2)


                # loss2 = supervised_round2_loss1 + supervised_round2_loss2 + consistency_weight * consistency_loss2 + args.coe2 * consistency_loss1
                loss2 = supervised_round2_loss1 + supervised_round2_loss2 + consistency_weight * consistency_loss2 + consistency_weight * consistency_loss1
                optimizer.zero_grad()
                loss2.backward()
                optimizer.step()


            loss = loss1 + loss2

            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (
                            1.0 - shift_iter/max_iterations ) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            # the first round
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss1', loss1, iter_num)
            writer.add_scalar('info/loss_ce1', loss_ce1, iter_num)
            writer.add_scalar('info/loss_dice1', loss_dice1, iter_num)
            writer.add_scalar('info/loss_ce2', loss_ce2, iter_num)
            writer.add_scalar('info/loss_dice2', loss_dice2, iter_num)

            # the second round
            writer.add_scalar('info/loss2', loss2, iter_num)
            # loss_round2_ce1, loss_round2_dice1
            writer.add_scalar('info/loss_round2_ce1', loss_round2_ce1, iter_num)
            writer.add_scalar('info/loss_round2_dice1', loss_round2_dice1, iter_num)
            writer.add_scalar('info/loss_round2_ce1_r', loss_round2_ce1_r, iter_num)
            writer.add_scalar('info/loss_round2_dice1_r', loss_round2_dice1_r, iter_num)
            writer.add_scalar('info/consistency_loss2',
                              consistency_loss2, iter_num)
            writer.add_scalar('info/consistency_loss1_r',
                              consistency_loss1_r, iter_num)



            if iter_num < args.warm_iter + 1:
                logging.info('iteration %d : loss : %f, loss1: %f' % (iter_num, loss.item(), loss1.item()))
            else:
                logging.info('iteration %d : loss : %f, loss1: %f, loss2: %f' % (
                iter_num, loss.item(), loss1.item(), loss2.item()))

            if iter_num > 0 and iter_num % 200 == 0:  # evaluation 200
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(validation_loader):

                    metric_i = val_single_PROMPT(
                        sampled_batch["image"], sampled_batch["label"], model, iter_num, args.num_classes,
                        sampled_batch["PSEUDO_MASK"], case_batch=sampled_batch["case_name"],
                        save_path=snapshot_path, device=device, prompt_mode=args.promptmode)

                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                print(metric_list)
                for class_i in range(num_classes):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_precision'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_precision = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_precision', mean_precision, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             'best_model.pth')
                    try:
                        model.save_lora_parameters(save_best)
                        model.save_lora_parameters(save_mode_path)
                    except:
                        model.module.save_lora_parameters(save_best)
                        model.module.save_lora_parameters(save_mode_path)

                logging.info(
                    'iteration %d : mean_dice : %f mean_precision : %f' % (iter_num, performance, mean_precision))
                model.train()

        save_interval = 200  # int(max_epoch/6)   # 20
        if (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_lora_parameters(save_mode_path)
            except:
                model.module.save_lora_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
