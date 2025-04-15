# -*- coding: utf-8 -*-


import argparse
import os
import time

from dataset import OJ104, CodeChef

import torch
import torch.nn as nn
from torch import optim
import random
import numpy


def adjust_learning_rate(optimizer, decay_rate=0.8):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate


def evaluate(classifier, dataset, device, batch_size=128):
    classifier = classifier.to(device)
    classifier.eval()
    test_num = 0
    test_correct = 0
    loss_total = 0
    while True:
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        with torch.no_grad():
            outputs, loss = classifier(batch['x'], batch['y'])
            loss_total += loss.item()
            labels = torch.tensor(batch['y'], dtype=torch.long).to(device)
            p_labels = torch.argmax(outputs, dim=1)
            res = p_labels == labels
            test_correct += torch.sum(res)
            test_num += len(labels)
    print('eval_acc:  %.5f' % (float(test_correct) * 100.0 / test_num))
    return float(test_correct) * 100.0 / test_num


if __name__ == "__main__":
    root = '../data/'
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', type=str, default="1")
    parser.add_argument('--data', type=str, default="../dataset/data.pkl.gz")
    parser.add_argument('--model_dir', type=str, default="../model/codebert/model")
    parser.add_argument('--task', type=str, default="code_function")
    parser.add_argument('--bs', type=int, default=16)

    opt = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    if int(opt.gpu) < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + opt.gpu)
    if opt.task == 'code_function':
        n_class = 104
        opt.data = root + opt.task + '/dataset/origin/data.pkl.gz'
        original_dataset = OJ104(path=opt.data)
        training_s = original_dataset.train
        valid_s = original_dataset.dev
        test_s = original_dataset.test
    elif opt.task == 'code_defect':
        n_class = 4
        opt.data = root + opt.task + '/dataset/origin/data.pkl.gz'
        original_dataset = CodeChef(path=opt.data)
        training_s = original_dataset.train
        valid_s = original_dataset.dev
        test_s = original_dataset.test

    n_class = 104

    batch_size = opt.bs
    rand_seed = 1726

    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    from graphcodebert import GraphCodeBERTClassifier

    opt.model_dir = '../project/2023/CausalCode_data/code_function/model/graphcodebert/origin/15.pt/'
    classifier = GraphCodeBERTClassifier(model_path=opt.model_dir,
                                         num_labels=n_class,
                                         device=device).to(device)
    classifier.eval()
    evaluate(classifier, test_s, device)
