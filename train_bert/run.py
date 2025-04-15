# -*- coding: utf-8 -*-


import argparse
import gzip
import os
import pickle
import shutil
from math import inf

import wandb
import time

from dataset import OJ104, CodeChef

import torch
import torch.nn as nn
from torch import optim
import random
import numpy
from bert_eval import evaluate
from attacker import Attacker, InsAttacker
from graphcodebert import GraphCodeBERTClassifier
from attacker4simple import CarrotToken, CarrotDeadCode, CausalCodeToken, CausalCodeDeadCode





def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate




def get_train_batch(renew_num=0):
    training_set = {}
    batch = {}
    domain_list = opt.domain_list
    rand_seed = random.randint(1, 1000)
    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    domain_list = [None] * len(opt.domain_list)
    for i, d in enumerate(opt.domain_list):
        # Reload the data set to keep the order consistent
        domain_name = d
        if d == 'origin' or not opt.do_renew:  # enter： token-origin,random_token0,random_token1; unknown- origin,random_token0,random_deadcode0
            data_path = os.path.join(root, opt.task, 'dataset', d) + "/data.pkl.gz"
        else:  # code_defect -input
            domain_name = d + str(renew_num * opt.aug_num + i - 1 + opt.begin_num)
            data_path = os.path.join(root, opt.task, 'dataset', domain_name) + "/data.pkl.gz"
        print('data_path: ', data_path)
        if opt.task == 'code_defect':
            domain_dataset = CodeChef(path=data_path)
        else:
            domain_dataset = OJ104(path=data_path)
        training_set[domain_name] = domain_dataset.train
        batch[domain_name] = training_set[domain_name].next_batch(opt.batch_size)
        domain_list[i] = domain_name
    return training_set, domain_list, batch




def train_CausalCode(cf, optimizer, opt, valid_set, wandb):
    cf.train()
    epoch = 0
    renew_num = 0
    penalty_ws = 1.0  # Compare the weight of the loss, compare the sizes of erm_loss and contrast_loss through debugging, and adjust this parameter so that they are of the same order of magnitude.
    training_set = {}
    ws_proportion = opt.begin_proportion  # Initially 0.1, but the loss of causal learning has not decreased.
    begin_wasserstein = 2
    if opt.continue_train:
        epoch = opt.begin_epoch
        renew_num = int(opt.begin_epoch / 7)
        print('continue train from epoch ' + str(epoch) + '....' + 'renew_num: ' + str(renew_num))
    best_loss = [inf, inf, inf]
    best_acc = 0.0
    increase_flag = True
    early_stop_counter = 0
    rand_seed = random.randint(1, 1000)
    print('start get data....')
    erm_loss_total = torch.tensor(0.0).to(device)
    wasserstein_loss_total = torch.tensor(0.0).to(device)
    loss_total = torch.tensor(0.0).to(device)
    training_set, domain_list, input_batch = get_train_batch(renew_num, cf)
    print('train epoch ' + str(epoch + 1) + '....')
    while True:
        input_batch = {}
        for d in domain_list:
            torch.manual_seed(rand_seed)
            random.seed(rand_seed)
            numpy.random.seed(rand_seed)
            batch = training_set[d].next_batch(opt.batch_size)
            input_batch[d] = batch
        assert numpy.array_equal(input_batch['origin']['id'],
                                 input_batch[domain_list[1]]['id']), "The two arrays are not equal."
        if batch['new_epoch']:
            from bert_eval import evaluate
            epoch += 1
            wandb.log({"train/epoch": epoch})
            if epoch > 7:
                with torch.no_grad():
                    acc = evaluate(cf, valid_set, device)
                    wandb.log({"eval/acc": acc, "train/epoch": epoch})
                if acc > best_acc:
                    best_acc = acc
                    early_stop_counter = 0
            cf.train()
            if epoch > 8:
                if acc < best_acc - 3:
                    print("model overfitting, early stop at epoch " + str(epoch) + "....")
                classifier.model.save_pretrained(os.path.join(opt.model_save_dir, str(epoch) + ".pt"))
                classifier.tokenizer.save_pretrained(os.path.join(opt.model_save_dir, str(epoch) + ".pt"))
                print("model saved at " + opt.model_save_dir + '/' + str(epoch) + '.pt')
            if opt.lr_decay:
                adjust_learning_rate(optimizer)
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            wandb.log({
                'epoch': epoch,
                'train/erm_loss_total': erm_loss_total,
                'train/wasserstein_loss_total': wasserstein_loss_total,
                'train/loss_total': loss_total
            })
            print('epoch-- %d erm_loss_total  %.4f' % (epoch, erm_loss_total))
            print('epoch-- %d wasserstein_loss_total  %.4f' % (epoch, wasserstein_loss_total))
            print('epoch-- %d loss_total  %.4f' % (epoch, loss_total))
            print('ws_proportion: ', ws_proportion)
            if wasserstein_loss_total > 0:

                # if erm_loss_total < best_loss[0] or wasserstein_loss_total < best_loss[1]:
                #     best_loss[0] = erm_loss_total
                #     best_loss[1] = wasserstein_loss_total
                if wasserstein_loss_total < best_loss[1] or loss_total < best_loss[2]:
                    best_loss[2] = loss_total
                    best_loss[1] = wasserstein_loss_total
                    increase_flag = True
                    early_stop_counter = 0
                    classifier.model.save_pretrained(os.path.join(opt.model_save_dir, "best.pt"))
                    classifier.tokenizer.save_pretrained(os.path.join(opt.model_save_dir, "best.pt"))
                    print("model saved at " + opt.model_save_dir + '/best.pt')
                else:
                    ws_proportion = opt.begin_proportion + 0.3
                    increase_flag = False
                    if epoch > 8:
                        early_stop_counter += 1

            if epoch >= opt.epochs:
                break
            if epoch % 7 == 0:  # Random samples are regenerated every 4 rounds
                renew_num += 1
                renew_num = renew_num % 4
                ws_proportion = opt.begin_proportion + 0.3
                best_loss = [inf, inf, inf]
                training_set, domain_list, input_batch = get_train_batch(renew_num, classifier)

            print('early_stop_counter: ', early_stop_counter)
            if early_stop_counter >= opt.early_stop and epoch >= 22:
                break
            rand_seed = random.randint(1, 1000)
            print('start training epoch ' + str(epoch + 1) + '....')
            erm_loss_total = 0
            wasserstein_loss_total = 0
            loss_total = 0
        optimizer.zero_grad()
        outputs = {}
        erm_loss = torch.tensor(0.0).to(device)
        wasserstein_loss = torch.tensor(0.0).to(device)
        labels = {d: input_batch[d]['y'] for d in domain_list}
        for d in domain_list:
            outputs[d], e_loss = classifier(input_batch[d]['x'], input_batch[d]['y'])
            erm_loss += e_loss
            del e_loss
        loss_e = erm_loss.clone()
        if epoch > begin_wasserstein:
            if opt.enhance_method == 'CausalCode':
                wasserstein_loss = ws_loss_function(domain_list, opt.batch_size, outputs, labels)
            elif opt.enhance_method == 'contrastive':
                wasserstein_loss = contrastive_loss_function(domain_list, outputs)
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


def ws_loss_function(domain_list, batch_size, outputs, labels):
    # Stack all outputs and labels together
    all_outputs = torch.stack([outputs[d] for d in domain_list])
    all_labels = torch.stack([torch.from_numpy(labels[d]) for d in domain_list])
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
    mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float().to(device)
    # Apply mask
    masked_diffs = diff_matrix * mask
    # Calculate total difference
    total_diff = torch.sum(masked_diffs)
    # Calculate the average difference
    return total_diff / (len(domain_list) * batch_size)



if __name__ == "__main__":
    root = '../data/'
    rand_seed = 1726
   parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--task', type=str, default='code_function', help='code_function, code_defect')
    parser.add_argument('--l2p', type=float, default=0)
    parser.add_argument('--enhance_method', type=str, default='origin',
                        help='origin,carrot,CausalCode,alert,augment')
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--data', type=str, default='dataset/origin/data.pkl.gz')
    parser.add_argument('--model_name', type=str, default="graphcodebert",
                        help="unixcoder, graphcodebert,cotrabert_c,contrabert_g")
    parser.add_argument('--model_save_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=28)
    parser.add_argument('--early_stop', default=3, type=int, required=False)
    parser.add_argument('--adv_train_path', type=str, default=None)
    parser.add_argument('--adv_train_size', type=int, default=2000)
    parser.add_argument('--attack_type', type=str, default='token', help='token, dead_code,token,unknown')
    parser.add_argument('--domain_list', type=str, default='origin,graphcodebert-CausalCode-dead_code')
    parser.add_argument('--model_dir', type=str, default='')  # for carrot
    parser.add_argument('--project_name', type=str, default='ICSE2025')
    parser.add_argument('--do_eval', action='store_true', default=False)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_carrot_attack', action='store_true', default=False)
    parser.add_argument('--do_mhm_attack', action='store_true', default=False)
    parser.add_argument('--do_renew', action='store_true', default=False)
    parser.add_argument('--attack_model_name', type=str, default='origin,graphcodebert-CausalCode-dead_code',
                        help='The name of the attack target model')

    # CausalCode Sweep
    parser.add_argument('--aug_num', type=int, default=1)
    parser.add_argument('--begin_num', type=int, default=1)
    parser.add_argument('--begin_proportion', type=float, default='0.1', help='0.1,0.3,0.5,0.7,0.9')
    parser.add_argument('--wandb_name', type=str, default='default',
                        help='The name of wandb, naming rules: experimental purpose, QA few')
    parser.add_argument('--continue_train', action='store_true', default=False)
    parser.add_argument('--begin_epoch', type=int, default=None)

    # n_candi
    parser.add_argument('--n_candi', type=int, default=10)

    opt = parser.parse_args()
    if opt.attack_type is None:
        # Determine whether the string 'dead_code' is in opt.wandb_name
        if 'token' in opt.wandb_name:
            opt.attack_type = 'token'
        else:
            opt.attack_type = 'dead_code'

    if opt.wandb_name == 'default':
        opt.wandb_name = opt.model_name + '-' + opt.enhance_method + '-' + opt.task + '-' + opt.attack_type + time.strftime(
            "%Y-%m-%d", time.localtime())
    # elif opt.wandb_name.startswith('1-all-asr-graphcodebert-code_function-CausalCode-token') and opt.do_train:
    #     opt.wandb_name = opt.wandb_name + '-' + str(opt.begin_proportion) + '-epoch' + str(opt.begin_epoch)
    # wandb initialization
    print('wandb_name: ', opt.wandb_name)
    wandb.init(project='CausalCode-bert-test', name=opt.wandb_name)
    wandb.config.update(vars(opt))
    os.environ["WANDB_MODE"] = "offline"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + opt.gpu)
    print(device)

    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)
    opt.data = root + opt.task + '/dataset/origin/data.pkl.gz'

    if opt.task == 'code_function':
        n_class = 104
        original_dataset = OJ104(path=opt.data)
    elif opt.task == 'code_defect':
        n_class = 4
        original_dataset = CodeChef(path=opt.data)
    training_s = original_dataset.train
    valid_s = original_dataset.dev
    test_s = original_dataset.test

    model_origin = "../transformers/microsoft/graphcodebert-base"

    opt.model_save_dir = root + opt.task + '/model/' + opt.model_name + '/' + opt.enhance_method + '/' + opt.attack_type

    if opt.attack_type == 'token':
        with gzip.open(os.path.join(root, opt.task, 'dataset', 'origin', 'data_uid.pkl.gz'), "rb") as f:  # 'all'
            symtab = pickle.load(f)
    elif opt.attack_type == 'dead_code':
        with gzip.open(os.path.join(root, opt.task, 'dataset', 'origin', 'data_inspos.pkl.gz'), "rb") as f:
            instab = pickle.load(f)

    if opt.do_train:
        classifier = GraphCodeBERTClassifier(model_path=model_origin,
                                             num_labels=n_class,
                                             device=device).to(device)
        optimizer = optim.AdamW(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        if opt.enhance_method == 'CausalCode' or opt.enhance_method == 'contrastive':
            if opt.continue_train:
                model_origin = os.path.join(root, opt.task, 'model', opt.model_name, opt.enhance_method,
                                            opt.attack_type, opt.wandb_name, 'best.pt')

                classifier = GraphCodeBERTClassifier(model_path=model_origin,
                                                     num_labels=n_class,
                                                     device=device).to(device)

            opt.model_save_dir = os.path.join(root, opt.task, 'model', opt.model_name, opt.enhance_method,
                                              opt.attack_type, opt.wandb_name)

            if opt.do_renew:
                opt.domain_list = ['origin']
                if opt.attack_type == 'token':
                    for i in range(opt.aug_num):
                        dataset_first_name = opt.model_name + '-' + 'CausalCode' + '-' + opt.attack_type + '-index-'
                        opt.domain_list.append(dataset_first_name)
                elif opt.attack_type == 'dead_code':
                    for i in range(opt.aug_num):
                        dataset_first_name = opt.model_name + '-' + 'CausalCode' + '-' + opt.attack_type + '-index-'
                        opt.domain_list.append(dataset_first_name)
            elif not opt.do_renew:
                os.makedirs(opt.model_save_dir, exist_ok=True)
                opt.domain_list = opt.domain_list.split(',')
            print(opt.model_save_dir)
            train_CausalCode(classifier, optimizer, opt, valid_s, wandb)
       
    else:
        model_attack = opt.model_save_dir + '/' + opt.wandb_name + '/best.pt'
        print('load model from ' + model_attack)
        classifier = GraphCodeBERTClassifier(model_path=model_attack,
                                             num_labels=n_class,
                                             device=device).to(device)
    if opt.do_eval:
        print('eval on test set...')
        classifier.eval()
        test_acc = evaluate(classifier, test_s, device)
        wandb.log({"test/acc": test_acc})
        del test_acc
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    if opt.do_attack:
        classifier.train()

        if opt.attack_type == 'token':
            atk = Attacker(original_dataset, symtab, classifier)
            atk.attack_all(5, 40, wandb=wandb)
        elif opt.attack_type == 'dead_code':
            atk = InsAttacker(original_dataset, instab, classifier)
            atk.attack_all(5, 40, wandb=wandb)
    wandb.finish()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
