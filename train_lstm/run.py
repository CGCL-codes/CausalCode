# -*- coding: utf-8 -*-


import autoroot
import wandb
import argparse
import gzip
import pickle
import random
import sys
import os
import time
from math import inf

import numpy

from dataset import OJ104, CodeChef
from lstm_classifier import LSTMClassifier, LSTMEncoder
from train_lstm.attacker import Attacker, InsAttacker
from lstm_eval import evaluate
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from attacker4advtraining import AdversarialTrainingAttacker, AdversarialTrainingInsAttacker
import attacker4simple as a4s


def gettensor(batch, device, batchfirst=False):
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).to(device), \
        torch.tensor(labels, dtype=torch.long).to(device)
    if batchfirst:
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels


def contrastive_loss_function(domain_list, batch_size, outputs, labels):
    # Convert outputs to tensor and send to GPU
    outputs_0 = outputs[domain_list[0]].to(device)
    outputs_1 = outputs[domain_list[1]].to(device)

    # Calculate the L2 norm (normalized) of outputs_0 and outputs_1
    norm_outputs_0 = outputs_0 / outputs_0.norm(dim=1, keepdim=True)
    norm_outputs_1 = outputs_1 / outputs_1.norm(dim=1, keepdim=True)

    # Calculate cosine similarity between two outputs
    sim_matrix = torch.mm(norm_outputs_0, norm_outputs_1.t())

    # The diagonal elements are the similarities of the positive sample pairs
    pos_sim = torch.diag(sim_matrix)

    # Calculate losses
    numerator = torch.exp(pos_sim)
    denominator = torch.sum(torch.exp(sim_matrix), dim=1) - numerator
    loss = -torch.log(numerator / denominator).mean()

    return loss


def contrastive_loss_function_backup(domain_list, batch_size, outputs, labels, alpha=1.0):
    # Stack all outputs and labels together
    all_outputs = torch.stack([outputs[d] for d in domain_list])
    all_labels = torch.stack([labels[d] for d in domain_list])

    # Extend all_outputs for vectorized calculations
    expanded_outputs = all_outputs.unsqueeze(2).expand(-1, -1, batch_size, -1)
    transposed_outputs = expanded_outputs.transpose(1, 2)

    # Calculate difference
    diff = expanded_outputs - transposed_outputs
    diff_matrix = torch.sum(diff ** 2, dim=-1)

    # Create two masks, one for marking locations where the labels are the same and another for marking locations where the labels are different
    same_label_mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
    diff_label_mask = 1.0 - same_label_mask

    # extended mask
    same_label_mask = same_label_mask.unsqueeze(-1).expand(-1, -1, batch_size, batch_size)
    diff_label_mask = diff_label_mask.unsqueeze(-1).expand(-1, -1, batch_size, batch_size)

    # Apply mask
    same_label_diffs = diff_matrix * same_label_mask
    diff_label_diffs = diff_matrix * diff_label_mask

    # Clear self-difference (case of i==j, k==l)
    eye = torch.eye(len(domain_list), device=all_outputs.device).unsqueeze(-1)
    eye = eye.expand(-1, -1, batch_size).bool()
    same_label_diffs[eye] = 0
    diff_label_diffs[eye] = 0

    # Calculate total difference
    total_same_label_diff = torch.sum(same_label_diffs)
    total_diff_label_diff = torch.sum(diff_label_diffs)

    # Calculate the average difference
    average_same_label_diff = total_same_label_diff / (len(domain_list) * len(domain_list) * batch_size * batch_size)
    average_diff_label_diff = total_diff_label_diff / (len(domain_list) * len(domain_list) * batch_size * batch_size)

    # Combining the two parts of the loss, the weight between the two can be adjusted through alpha.
    print('average_same_label_diff: ', average_same_label_diff)
    print('average_diff_label_diff: ', average_diff_label_diff)
    combined_loss = average_same_label_diff - alpha * average_diff_label_diff

    return combined_loss


def ws_loss_function_backup(domain_list, batch_size, outputs, labels):
    # Stack all outputs and labels together
    all_outputs = torch.stack([outputs[d] for d in domain_list])
    all_labels = torch.stack([labels[d] for d in domain_list])

    # Extend all_outputs for vectorized calculations
    expanded_outputs = all_outputs.unsqueeze(2).expand(-1, -1, batch_size, -1)
    transposed_outputs = expanded_outputs.transpose(1, 2)

    # Calculate difference
    diff = expanded_outputs - transposed_outputs
    diff_matrix = torch.sum(diff ** 2, dim=-1)

    # Create a mask that marks locations where the labels are the same
    mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
    mask = mask.unsqueeze(-1).expand(-1, -1, batch_size, batch_size)

    # Apply mask
    masked_diffs = diff_matrix * mask

    # Clear self-difference (case of i==j, k==l)
    eye = torch.eye(len(domain_list), device=all_outputs.device).unsqueeze(-1)
    eye = eye.expand(-1, -1, batch_size).bool()
    masked_diffs[eye] = 0

    # Calculate the total difference and average difference
    total_diff = torch.sum(masked_diffs)
    average_diff = total_diff / (len(domain_list) * len(domain_list) * batch_size * batch_size)

    return average_diff


def ws_loss_function(domain_list, batch_size, outputs, labels):
    # Stack all outputs and labels together
    all_outputs = torch.stack([outputs[d] for d in domain_list])
    all_labels = torch.stack([labels[d] for d in domain_list])

    # Initialize a difference matrix
    diff_matrix = torch.zeros(len(domain_list), len(domain_list), batch_size, device=all_outputs.device)

    # Compute difference matrix
    for i in range(len(domain_list)):
        for j in range(i + 1, len(domain_list)):
            diff = all_outputs[i] - all_outputs[j]
            diff = torch.sum(diff ** 2, dim=-1)
            diff_matrix[i, j] = diff
            diff_matrix[j, i] = diff  # The difference is symmetric, so this can be done to reduce computation
    # Create a mask that marks locations where the labels are the same
    mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
    # Apply mask
    masked_diffs = diff_matrix * mask
    # Calculate total difference
    total_diff = torch.sum(masked_diffs)
    # Calculate the average difference
    average_diff = total_diff / (len(domain_list) * batch_size)
    return average_diff


def train_alert(classifier, optimizer, opt, training_set, valid_set, domain_list):
    """
    Training model
        Contains two training methods: alert and augment.
    """
    classifier.train()
    epoch = 0
    best_loss = [inf, inf]
    early_stop_counter = 0
    print('start training epoch ' + str(epoch + 1) + '....')
    erm_loss_total = torch.tensor(0.0).to(device)
    loss_total = torch.tensor(0.0).to(device)
    criterion = nn.CrossEntropyLoss()
    while True:
        inputs = {}
        labels = {}
        batch = {}
        for d in domain_list:
            batch[d] = training_set[d].next_batch(opt.batch_size)
        if batch['origin']['new_epoch']:
            from lstm_eval import evaluate
            evaluate(classifier, valid_set, device)
            classifier.train()
            epoch += 1
            torch.save(classifier.state_dict(), opt.model_save_dir + '/' + str(epoch) + '.pt')
            print("model saved at " + opt.model_save_dir + '/' + str(epoch) + '.pt')
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            with open(opt.model_save_dir + '/eval.txt', 'a') as f:
                f.write('epoch-- %d erm_loss_total  %.4f\n' % (epoch, erm_loss_total))
                f.write('epoch-- %d loss_total  %.4f\n' % (epoch, loss_total))
            print('epoch-- %d erm_loss_total  %.4f' % (epoch, erm_loss_total))
            print('epoch-- %d loss_total  %.4f' % (epoch, loss_total))
            if erm_loss_total < best_loss[0]:
                best_loss[0] = erm_loss_total
                early_stop_counter = 0
                torch.save(classifier.state_dict(), opt.model_save_dir + '/best.pt')
            else:
                early_stop_counter += 1
                if early_stop_counter > 2:
                    break
            if epoch >= opt.epochs:
                break
            print('start training epoch ' + str(epoch + 1) + '....')
            erm_loss_total = 0
            loss_total = 0
        for d in domain_list:
            inputs[d], labels[d] = gettensor(training_set[d].next_batch(opt.batch_size), device=device,
                                             batchfirst=False)

        optimizer.zero_grad()
        outputs = {}
        erm_loss = torch.tensor(0.0).to(device)
        loss_e = torch.tensor(0.0).to(device)
        for d in domain_list:
            outputs[d] = classifier(inputs[d])[0]
            erm_loss += criterion(outputs[d], labels[d])
        loss_e = erm_loss.clone()
        loss_e.backward()
        optimizer.step()
        erm_loss_total += erm_loss.item()
        loss_total += loss_e.item()
        del erm_loss
        del loss_e


def train_original(classifier, cfg, training_set, valid_set, wandb):
    classifier.train()
    epoch = 0
    best_loss = 100000
    loss_total = 0
    step = 0
    early_stop = 0
    criterion = nn.CrossEntropyLoss()
    os.makedirs(cfg.model_save_dir, exist_ok=True)
    print('start training epoch ' + str(epoch + 1) + '....')
    while True:
        batch = training_set.next_batch(cfg.batch_size)
        if batch['new_epoch']:
            epoch += 1
            wandb.log({"train/epoch": epoch})
            acc = evaluate(classifier, valid_set, device)
            wandb.log({"eval/acc": acc, "train/epoch": epoch})
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            classifier.train()
            torch.save(classifier.state_dict(), cfg.model_save_dir + "/" + str(epoch) + '.pt')
            if cfg.lrdecay:
                adjust_learning_rate(optimizer)
            if epoch == cfg.epochs:
                break
            if best_loss > loss_total:
                best_loss = loss_total
                early_stop = 0
                torch.save(classifier.state_dict(), cfg.model_save_dir + "/" + 'best.pt')
            else:
                print('early stop at epoch ' + str(epoch) + '....')
                early_stop += 1
            loss_total = 0
            step = 0
            if cfg.early_stop == early_stop:
                break
            print('start training epoch ' + str(epoch + 1) + '....')
        inputs, labels = gettensor(batch, device, batchfirst=False)

        optimizer.zero_grad()
        outputs = classifier(inputs)[0]
        loss = criterion(outputs, labels)
        step += 1
        if step % 50 == 0:
            wandb.log({"train/loss": loss.item()})
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    return classifier


def adjust_learning_rate(optimizer, decay_rate=0.95):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def train_carrot(classifier, optimizer, opt, training_set, valid_set):
    """
        carrot training
        Every 14 epochs, 2000 adversarial samples are updated and expanded until convergence.
    """
    classifier.train()
    epoch = 0
    best_loss = inf
    loss_total = 0
    early_stop = 0
    criterion = nn.CrossEntropyLoss()
    os.makedirs(opt.model_save_dir, exist_ok=True)
    print('start training epoch ' + str(epoch + 1) + '....')
    training_set, valid_set = gen_carrot(classifier, opt, training_s)
    while True:
        batch = training_set.next_batch(opt.batch_size)
        if batch['new_epoch']:
            epoch += 1
            from lstm_eval import evaluate
            evaluate(classifier, valid_set, device)
            classifier.train()
            torch.save(classifier.state_dict(), opt.model_save_dir + "/" + str(epoch) + '.pt')
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            if epoch == opt.epochs:
                break
            if best_loss > loss_total:
                best_loss = loss_total
                early_stop = 0
                torch.save(classifier.state_dict(), opt.model_save_dir + "/" + 'best.pt')
            else:
                early_stop += 1
                print('early stop : ' + str(early_stop) + '....')
            loss_total = 0
            if opt.early_stop == early_stop and epoch > 15:
                break
            if epoch % 12 == 0:
                early_stop = 0
                training_set, valid_set = gen_carrot(classifier, opt, training_s)

            print('start training epoch ' + str(epoch + 1) + '....')
        inputs, labels = gettensor(batch, device, batchfirst=False)

        optimizer.zero_grad()
        outputs = classifier(inputs)[0]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_total += loss.item()

    return classifier


def gen_carrot(classifier, cfg, original_dataset):
    adv_train_p = renew_adv(classifier, cfg, original_dataset)
    if cfg.task == 'code_function':
        data_set = OJ104(path=cfg.data,

                         vocab_size=vocab_size,
                         adv_train_path=adv_train_p,
                         adv_train_size=cfg.adv_train_size)  # if you have adversarial samples
        training_set = data_set.train
        valid_set = data_set.dev
    else:
        data_set = CodeChef(path=cfg.data,
                            vocab_size=vocab_size,
                            adv_train_path=adv_train_p,
                            adv_train_size=cfg.adv_train_size)  # if you have adversarial samples
        training_set = data_set.train
        valid_set = data_set.dev
    return training_set, valid_set


def get_train_batch(cfg, model, renew_num=0):
    training_set = {}
    batch = {}
    # Reload the data set to keep the order consistent
    rand_seed = 666
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    domain_l = [None] * len(cfg.domain_list)
    for i, d in enumerate(cfg.domain_list):
        if d == 'origin':
            domain_name = d
            data_path = os.path.join(root, cfg.task, "dataset", d) + "/data.pkl.gz"
        elif d != 'origin' and not opt.do_renew:  # Read generated data
            data_path = os.path.join(root, cfg.task, "dataset", opt.model_name, d) + "/data.pkl.gz"
        else:  # do_renew
            data_index = renew_num * opt.aug_num + i - 1 + opt.begin_num
            data_path = os.path.join(root, cfg.task, "dataset", opt.model_name,
                                     d + str(data_index)) + "/data.pkl.gz"
            domain_name = d + str(data_index)

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('loading data_path: ', data_path)
        print('domain_name: ', domain_name)
        if cfg.task == 'code_defect':
            domain_dataset = CodeChef(path=data_path, seed=rand_seed)
        else:
            domain_dataset = OJ104(path=data_path, seed=rand_seed)
        training_set[domain_name] = domain_dataset.train
        batch[domain_name] = training_set[domain_name].next_batch(cfg.batch_size)
        domain_l[i] = domain_name
    print('domain_list', domain_l)
    return training_set, domain_l, batch


def get_train_batch_backup(cfg, model, renew_num=0):
    training_set = {}
    batch = {}
    rand_seed = random.randint(1, 1000)
    # Reload the data set to keep the order consistent
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    domain_l = [None] * len(cfg.domain_list)
    for i, d in enumerate(cfg.domain_list):
        print('domain_list', cfg.domain_list)
        if d == 'origin' or not cfg.do_renew:
            domain_name = d
            data_path = os.path.join(root, cfg.task, "dataset", d) + "/data.pkl.gz"
        elif renew_num == 0:  # Read generated data
            data_path = os.path.join(root, cfg.task, "dataset", d + '-' + str(renew_num)) + "/data.pkl.gz"
            domain_name = d + '-' + str(renew_num)
        else:  # Generate new data based on the latest model trained, code defects
            data_path = os.path.join(root, cfg.task, "dataset",
                                     'aug_num' + str(cfg.aug_num) + '-' + d + '-' + str(renew_num)) + "/data.pkl.gz"
            domain_name = 'aug_num' + str(cfg.aug_num) + '-' + d + '-' + str(renew_num)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('loading data_path: ', data_path)
        print('domain_name: ', domain_name)
        if cfg.task == 'code_defect':
            domain_dataset = CodeChef(path=data_path)
        else:
            domain_dataset = OJ104(path=data_path)
        training_set[domain_name] = domain_dataset.train
        batch[domain_name] = training_set[domain_name].next_batch(cfg.batch_size)
        domain_l[i] = domain_name
    return training_set, domain_l, batch


def train_CausalCode(cf, cfg, valid_set, wandb):
    cf.train()
    epoch = 0
    penalty_ws = 1
    criterion = nn.CrossEntropyLoss()
    ws_proportion = 0.1
    begin_wasserstein = 1
    best_loss = [inf, inf]
    increase_flag = True
    early_stop_counter = 0
    renew_num = 0
    rand_seed = random.randint(1, 1000)
    erm_loss_total = torch.tensor(0.0).to(device)
    wasserstein_loss_total = torch.tensor(0.0).to(device)
    loss_total = torch.tensor(0.0).to(device)
    if opt.continue_train:
        epoch = opt.begin_epoch
        renew_num = int(opt.begin_epoch / 7)
        print('continue train from epoch ' + str(epoch) + '....' + 'renew_num: ' + str(renew_num))

    training_set, d_list, batch = get_train_batch(cfg, cf, renew_num)
    print('start training epoch ' + str(epoch + 1) + '....')
    while True:
        inputs = {}
        labels = {}
        batch = {}
        for d in d_list:
            torch.manual_seed(rand_seed)
            random.seed(rand_seed)
            numpy.random.seed(rand_seed)
            batch[d] = training_set[d].next_batch(cfg.batch_size)
        assert numpy.array_equal(batch['origin']['id'], batch[d_list[1]]['id']), "The two arrays are not equal."
        if batch['origin']['new_epoch']:
            from lstm_eval import evaluate
            epoch += 1
            wandb.log({"train/epoch": epoch})
            acc = evaluate(cf, valid_set, device)
            wandb.log({"eval/acc": acc, "train/epoch": epoch})
            cf.train()
            torch.save(cf.state_dict(), cfg.model_save_dir + '/' + str(epoch) + '.pt')
            print("model saved at " + cfg.model_save_dir + '/' + str(epoch) + '.pt')
            if cfg.lrdecay:
                adjust_learning_rate(optimizer)
            wandb.log({
                'epoch': epoch,
                'train/erm_loss_total': erm_loss_total,
                'train/wasserstein_loss_total': wasserstein_loss_total,
                'train/loss_total': loss_total
            })
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print('epoch-- %d erm_loss_total  %.4f' % (epoch, erm_loss_total))
            print('epoch-- %d wasserstein_loss_total  %.4f' % (epoch, wasserstein_loss_total))
            print('epoch-- %d loss_total  %.4f' % (epoch, loss_total))
            print('ws_proportion: ', ws_proportion)
            if wasserstein_loss_total > 0:
                if erm_loss_total < best_loss[0] or wasserstein_loss_total < best_loss[1]:
                    best_loss[0] = erm_loss_total
                    best_loss[1] = wasserstein_loss_total
                    increase_flag = True
                    early_stop_counter = 0
                    torch.save(cf.state_dict(), cfg.model_save_dir + '/best.pt')
                else:
                    ws_proportion = 0.5
                    increase_flag = False
                    early_stop_counter += 1
            if epoch >= cfg.epochs and epoch > 15:
                break
            if wasserstein_loss_total > 10000:
                print('wasserstein_loss_total>10000')  # gradient explosion
                break
            if early_stop_counter > cfg.early_stop and epoch > 15:
                print('early_stop')
                break
            print('early_stop_counter: ', early_stop_counter)
            rand_seed = random.randint(1, 1000)
            if epoch % 7 == 0:
                renew_num += 1
                ws_proportion = 0.3
                early_stop_counter = 0
                best_loss = [inf, inf]
                increase_flag = True
                training_set, d_list, batch = get_train_batch(cfg, cf, renew_num)
            if begin_wasserstein == epoch:
                increase_flag = True
            print('start training epoch ' + str(epoch + 1) + '....')
            erm_loss_total = 0
            wasserstein_loss_total = 0
            loss_total = 0
        for d in d_list:
            inputs[d], labels[d] = gettensor(batch[d], device, batchfirst=False)
        optimizer.zero_grad()
        outputs = {}
        erm_loss = torch.tensor(0.0).to(device)
        loss_e = torch.tensor(0.0).to(device)
        wasserstein_loss = torch.tensor(0.0).to(device)
        for d in d_list:
            outputs[d] = cf(inputs[d])[0]
            erm_loss += criterion(outputs[d], labels[d])
        loss_e = erm_loss.clone()
        if begin_wasserstein < epoch:
            if opt.enhance_method == 'CausalCode':
                wasserstein_loss = ws_loss_function(d_list, cfg.batch_size, outputs, labels)
            elif opt.enhance_method == 'contrastive':
                wasserstein_loss = contrastive_loss_function(d_list, cfg.batch_size, outputs, labels)
            # Set the loss in stages. In the first stage, there is only erm_loss. In the second stage, erm_loss and wasserstein_loss exist at the same time. The weight of wasserstein_loss gradually increases by 2 times. In the third stage, the weight of wasserstein_loss remains unchanged at 1, and training starts with both losses. rising or converging
            if increase_flag and ws_proportion <= penalty_ws:
                increase_flag = False  # A new round begins, reset the flag
                ws_proportion += 0.1
            loss_e += ws_proportion * wasserstein_loss
        loss_e.backward()
        optimizer.step()
        erm_loss_total += erm_loss.item()
        wasserstein_loss_total += wasserstein_loss.item()
        loss_total += loss_e.item()
        del erm_loss
        del wasserstein_loss
        del loss_e


def renew_adv(classifier, opt, original_dataset):
    adv_train_path = os.path.join(root, opt.task, "dataset", 'lstm_carrot_' + opt.attack_type)
    print('start renew adv ....', adv_train_path)
    if opt.attack_type == 'token':
        with gzip.open(os.path.join(root, opt.task, "dataset", 'origin', 'data_uid.pkl.gz'), "rb") as f:  # 'all'
            symtab = pickle.load(f)
        adv_train_path = adv_save_path + "/token.pkl.gz"
        atk = AdversarialTrainingAttacker(original_dataset, symtab, classifier)
        atk.attack_all(40, 50,
                       res_save=adv_train_path,
                       adv_sample_size=2000)
    elif opt.attack_type == 'deadcode':
        with gzip.open(os.path.join(root, opt.task, "dataset", 'origin', 'data_inspos.pkl.gz'), "rb") as f:
            instab = pickle.load(f)
        adv_train_path = adv_save_path + "/deadcode.pkl.gz"
        atk = AdversarialTrainingInsAttacker(original_dataset, instab, classifier)
        atk.attack_all(40, 50,
                       res_save=adv_train_path,
                       adv_sample_size=2000)
    return adv_train_path


def main():
    global root, optimizer, training_s, opt, device, embedding_size, max_len, vocab_size, adv_save_path, original_dataset, instab, symtab
    root = '../data/'
    parser = argparse.ArgumentParser()
    # Fixed parameters
    parser.add_argument('--gpu', default='0', type=str, required=False)
    parser.add_argument('--lr', default=1e-3, type=float, required=False)
    parser.add_argument('--early_stop', default=3, type=int,
                        required=False)  # carrot early_stop -4  epochs 30  # CausalCode early_stop -4  epochs 30
    parser.add_argument('--model_name', default='LSTM', required=False)
    parser.add_argument('--epochs', type=int, default=28)
    parser.add_argument('--attn', action='store_true')
    parser.add_argument('--l2p', type=float, default=1e-8, required=False)
    parser.add_argument('--lrdecay', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_attack', action='store_true', default=False)
    parser.add_argument('--do_renew', action='store_true', default=False)
    parser.add_argument('--do_noise', action='store_true', default=False)

    parser.add_argument('--enhance_method', type=str, default='CausalCode',
                        help='origin,CausalCode,carrot,alert,augment,contrastive')
    parser.add_argument('--task', type=str, default='code_defect', help='code_function, code_defect')

    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--adv_train_path', type=str, default=None)
    parser.add_argument('--adv_train_size', type=int, default=2000)

    parser.add_argument('--model_save_dir', type=str, default=root + '/model/LSTM/origin/')

    parser.add_argument('--attack_type', default='token', type=str, required=False, help='token, dead_code ,mhm')
    parser.add_argument('--attack_model_name', type=str, default='best', help='The name of the attack target model')

    parser.add_argument('--domain_list', type=str, default='origin,adversarial_token')

    parser.add_argument('--wandb_name', type=str, default='default', help='The name of wandb, naming rules: experimental purpose, QA few')

    # Temporary experimental parameters
    parser.add_argument('--aug_num', type=int, default=3)
    parser.add_argument('--begin_num', type=int, default=1)
    parser.add_argument('--iter_select', type=str, default='30', help='1,5,15,20,35,50,65')
    parser.add_argument('--continue_train', action='store_true', default=False)
    parser.add_argument('--begin_epoch', type=int, default=0)

    opt = parser.parse_args()

    os.environ["WANDB_MODE"] = "offline"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    device = torch.device("cuda:" + opt.gpu)
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    opt.data = root + opt.task + '/dataset/origin/data.pkl.gz'
    if opt.task == 'code_function':
        max_len = 500
        n_class = 104
        vocab_size = 5000
        original_dataset = OJ104(path=opt.data)
    elif opt.task == 'code_defect':
        n_class = 4
        vocab_size = 3000
        max_len = 300
        original_dataset = CodeChef(path=opt.data)
    training_s = original_dataset.train
    valid_s = original_dataset.dev
    test_s = original_dataset.test

    opt.model_save_dir = root + opt.task + '/model/' + opt.model_name + '/' + opt.enhance_method + '/' + opt.attack_type

    # wandb initialization
    if opt.enhance_method == 'CausalCode' or opt.enhance_method == 'contrastive':
        pass
    else:
        opt.wandb_name = opt.model_name + '-' + opt.enhance_method + '-' + opt.task + '-' + opt.attack_type
    wandb.init(project='CausalCode-exp', name=opt.wandb_name)
    wandb.config.update(vars(opt))
    wandb.run.log_code('./', include_fn=lambda path: path.endswith(".py"))

    # LSTM model
    enc = LSTMEncoder(embedding_size, hidden_size, n_layers)
    classifier = LSTMClassifier(vocab_size, embedding_size, enc, hidden_size, n_class, max_len,
                                attn=opt.attn).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr)

    # Download Data
    with gzip.open(os.path.join(root, opt.task, 'dataset', 'origin', 'data_uid.pkl.gz'), "rb") as f:
        symtab = pickle.load(f)
    with gzip.open(os.path.join(root, opt.task, 'dataset', 'origin', 'data_inspos.pkl.gz'), "rb") as f:
        instab = pickle.load(f)
    if opt.do_train:
        if opt.enhance_method == 'carrot':
            print('Begin carrot -------------')
            adv_save_path = root + opt.task + '/dataset/carrot/LSTM/'
            os.makedirs(adv_save_path, exist_ok=True)
            classifier.load_state_dict(
                torch.load(os.path.join(root, opt.task, 'model', opt.model_name, 'origin', '15.pt')))
            os.makedirs(opt.model_save_dir, exist_ok=True)
            train_carrot(classifier, optimizer, opt, training_s, valid_s)
        elif opt.enhance_method == 'alert' or opt.enhance_method == 'augment':
            opt.domain_list = opt.domain_list.split(',')
            training_list = {}
            for d in opt.domain_list[1:]:
                if opt.task == 'code_function':
                    domain_dataset = OJ104(path=os.path.join(root, opt.task, "dataset", d, 'data.pkl.gz'))
                else:
                    domain_dataset = CodeChef(path=os.path.join(root, opt.task, "dataset", d, 'data.pkl.gz'))
                training_list[d] = domain_dataset.train
            training_list['origin'] = training_s
            opt.model_save_dir = os.path.join(root, opt.task, 'model', opt.model_name, opt.enhance_method,
                                              opt.wandb_name)
            os.makedirs(opt.model_save_dir, exist_ok=True)
            train_alert(classifier, optimizer, opt, training_list, valid_s, opt.domain_list)
        elif opt.enhance_method == 'CausalCode' or opt.enhance_method == 'contrastive':
            print('Begin CausalCode -------------')
            if opt.do_renew:
                opt.domain_list = ['origin']
                if opt.attack_type == 'token':
                    for i in range(opt.aug_num):
                        dataset_first_name = 'CausalCode' + '-' + opt.attack_type + '-' + opt.wandb_name + '-'
                        opt.domain_list.append(dataset_first_name)
                elif opt.attack_type == 'dead_code':
                    for i in range(opt.aug_num):
                        dataset_first_name = opt.model_name + '-' + 'CausalCode' + '-' + opt.attack_type + '-index-'
                        opt.domain_list.append(dataset_first_name)
            elif not opt.do_renew:
                opt.domain_list = opt.domain_list.split(',')
                print('opt.domain_list: ', opt.domain_list)
            opt.model_save_dir = os.path.join(opt.model_save_dir, opt.wandb_name)

            if opt.continue_train:
                if opt.continue_train:
                    classifier.load_state_dict(torch.load(opt.model_save_dir + '/best.pt'))
            else:
                classifier.load_state_dict(
                    torch.load(os.path.join(root, opt.task, 'model', opt.model_name, 'origin', 'best.pt')))

            wandb.config.update({"model_save_dir": opt.model_save_dir}, allow_val_change=True)
            print('opt.model_save_dir: ', opt.model_save_dir)
            os.makedirs(opt.model_save_dir, exist_ok=True)
            train_CausalCode(classifier, opt, valid_s, wandb)
        else:
            train_original(classifier, opt, training_s, valid_s, wandb)
    else:
        classifier.load_state_dict(torch.load(opt.model_save_dir + '/' + opt.attack_model_name + '/best.pt'))

    if opt.do_eval:
        print('eval on test set...')
        classifier.eval()
        test_acc = evaluate(classifier, test_s, device)
        wandb.log({"test/acc": test_acc})

    if opt.do_attack:
        classifier.train()
        if opt.attack_type == 'token':
            atk = Attacker(original_dataset, symtab, classifier, device)
            atk.attack_all(40, 50, wandb=wandb)
        elif opt.attack_type == 'dead_code':
            atk = InsAttacker(original_dataset, instab, classifier, device)
            atk.attack_all(40, 20, wandb=wandb)
        elif opt.attack_type == 'all':
            atk = Attacker(original_dataset, symtab, classifier, device)
            atk.attack_all(40, 50, wandb=wandb)
            atk = InsAttacker(original_dataset, instab, classifier, device)
            atk.attack_all(40, 20, wandb=wandb)
    print('opt.model_save_dir: ', opt.model_save_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
