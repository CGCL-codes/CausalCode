# -*- coding: utf-8 -*-


import argparse
import os
from dataset import OJ104
from bert_eval import evaluate
import torch
import torch.nn as nn
from torch import optim
import random
import numpy


def trainEpochs(epochs, training_set, valid_set, device,
                batch_size=32, print_each=100, plot_each=100, saving_path='./'):
    classifier.train()
    epoch = 0
    i = 0
    penalty_s = -1
    print_loss_total = 0
    w_loss_total = 0
    n_batch = int(training_set['origin'].get_size() / batch_size)
    print(training_set['origin'].get_size())
    print('start training epoch ' + str(epoch + 1) + '....')
    erm_loss_total = 0
    loss_total = 0
    early_stop = 0
    best_w_loss = 1000000
    batch = {}
    while True:
        input_batch = {}
        for d in domain_list:
            rand_seed = 666
            torch.manual_seed(rand_seed)
            random.seed(rand_seed)
            numpy.random.seed(rand_seed)
            batch = training_set[d].next_batch(batch_size)
            input_batch[d] = batch

        assert numpy.array_equal(input_batch['origin']['id'],
                                 input_batch[domain_list[1]]['id']), "The two arrays are not equal."

        if batch['new_epoch']:
            epoch += 1
            with torch.no_grad():
                os.makedirs(os.path.join(saving_path, str(epoch) + ".pt"), exist_ok=True)
                evaluate(classifier, valid_set, device, os.path.join(saving_path, str(epoch) + ".pt"), batch_size)
            classifier.train()
            classifier.model.save_pretrained(os.path.join(saving_path, str(epoch) + ".pt"))
            classifier.tokenizer.save_pretrained(os.path.join(saving_path, str(epoch) + ".pt"))
            if opt.lrdecay:
                adjust_learning_rate(optimizer)
            if epoch == epochs:
                break
            i = 0
            print_loss_total = 0
            with open(saving_path + 'train_log.txt', 'a') as f:
                f.write('epoch-- %d erm_loss_total  %.4f\n' % (epoch, erm_loss_total))
                f.write('epoch-- %d wasserstein_loss_total  %.4f\n' % (epoch, w_loss_total))
                f.write('epoch-- %d loss_total  %.4f\n' % (epoch, loss_total))
            print('epoch-- %d erm_loss_total  %.4f' % (epoch, erm_loss_total))
            print('epoch-- %d wasserstein_loss_total  %.4f' % (epoch, w_loss_total))
            print('epoch-- %d loss_total  %.4f' % (epoch, loss_total))
            print('start training epoch ' + str(epoch + 1) + '....')
            if loss_total < best_w_loss:
                best_w_loss = loss_total
                early_stop = 0
            else:
                early_stop += 1
                if early_stop >= 4:
                    break
            erm_loss_total = 0
            w_loss_total = 0
            loss_total = 0

        optimizer.zero_grad()
        outputs = {}
        pos_match_counter = 0
        erm_loss = torch.tensor(0.0).to(device)
        wasserstein_loss = torch.tensor(0.0).to(device)
        labels = {d: input_batch[d]['y'] for d in domain_list}
        for d in domain_list:
            outputs[d], e_loss = classifier(input_batch[d]['x'], input_batch[d]['y'])
            erm_loss += e_loss
            del e_loss
        loss_e = erm_loss.clone()
        del input_batch
        if opt.enhance_method == 'CausalCode':  # and epoch % 2 == 0
            wasserstein_loss = ws_loss_function(domain_list, batch_size, outputs, labels)
            # print((penalty_ws * (epoch - penalty_s) / (epochs - penalty_s)) * wasserstein_loss)
            loss_e += (penalty_ws * (epoch - penalty_s) / (epochs - penalty_s)) * wasserstein_loss

        loss_e.backward()
        optimizer.step()
        erm_loss_total += erm_loss.item()
        print_loss_total += loss_e.item()
        w_loss_total += wasserstein_loss.item()
        loss_total += loss_e.item()
        del erm_loss
        del wasserstein_loss
        del loss_e

        if (i + 1) % print_each == 0:
            print_loss_avg = print_loss_total / print_each
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (epoch + 1, (i + 1) / n_batch * 100, print_loss_avg))
            print('erm_loss: %.4f' % (erm_loss_total / print_each))
            print('w_loss: %.4f' % (w_loss_total / print_each))
        i += 1


def cosine_similarity(x1, x2):
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
    return 1.0 - cos(x1, x2)


def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def ws_loss_function(domain_list, batch_size, outputs, labels):
    # Stack all outputs and labels together
    all_outputs = torch.stack([outputs[d] for d in domain_list])
    # all_labels = torch.stack([labels[d] for d in domain_list])
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
    # mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float()
    mask = (all_labels.unsqueeze(1) == all_labels.unsqueeze(0)).float().to(device)

    # Apply mask
    masked_diffs = diff_matrix * mask

    # Calculate total difference
    total_diff = torch.sum(masked_diffs)

    # Calculate the average difference
    average_diff = total_diff / (len(domain_list) * batch_size)
    return average_diff


if __name__ == "__main__":
    # /dataset/dg_data/paper01_data/code_function/model/GraphCodeBert/CausalCode/origin_random_token
    root = '/dataset/dg_data/paper01_data/code_function/'

    model_save_path = root + 'model/GraphCodeBert/'
    domain_list = ['origin', 'graphcodebert_alert_token']  # 'random_token_carrot','random_deadcode'
    model_name = 'GraphCodeBert'  # unixcoder, GraphCodeBert
    pos_metric = 'l2'
    penalty_ws = 1
    lr = 0.00003

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--l2p', type=float, default=0)
    parser.add_argument('--lrdecay', action='store_true')
    parser.add_argument('--model', type=str, default="graphcodebert")
    parser.add_argument('--enhance_method', type=str, default='ALERT',
                        help='origin,CausalCode,CARROT,ALERT')  # Invariant feature enhancement-CausalCode or adversarial sample enhancement-ALERT
    parser.add_argument('--save_dir', type=str, default=model_save_path)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    data_path = root + opt.task + '/dataset/origin/data.pkl.gz'

    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + opt.gpu)

    if opt.model == "unixcoder":
        model_path = "microsoft/unixcoder-base"
    elif opt.model == "graphcodebert":
        model_path = "microsoft/graphcodebert-base"

    model_parameter = '+'.join(domain_list)

    # model_save_path = root + 'model/' + model_name + '/' + model_parameter + '/'
    model_save_path = os.path.join(root, 'model', model_name, opt.enhance_method, model_parameter)
    n_class = 104

    training_set = {}
    for domain in domain_list:
        path = root + 'data/' + domain + '/data.pkl.gz'
        print(path)
        poj = OJ104(path=path)
        training_set[domain] = poj.train
        if domain == 'origin':
            valid_set = poj.dev
            test_set = poj.test

    # import transformers after gpu selection
    from graphcodebert import GraphCodeBERTClassifier

    classifier = GraphCodeBERTClassifier(model_path=model_path,
                                         num_labels=n_class,
                                         device=device).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=opt.lr, weight_decay=opt.l2p)

    trainEpochs(opt.epochs, training_set, valid_set, device,
                saving_path=model_save_path, batch_size=opt.bs)
