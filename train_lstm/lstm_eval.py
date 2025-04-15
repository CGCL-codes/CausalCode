# -*- coding: utf-8 -*-


import argparse
import os
import shutil
import time

from dataset import OJ104
from lstm_classifier import LSTMClassifier, LSTMEncoder, GRUClassifier, GRUEncoder

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np


def write_gen_data_time(res_save, st_time, size):
    log_path = os.path.dirname(res_save)
    with open(os.path.join(log_path, "aug_time_cost.txt"), 'a') as f:
        print("\n Time Cost: %.1f " % (time.time() - st_time))
        f.write("\n Time Cost: %.1f " % (time.time() - st_time))
        f.write("\n data size: %d " % size)
    # Copy attacker4simple.py to the res_save directory
    try:
        shutil.copyfile(os.path.join(os.path.dirname(__file__), "attacker4simple.py"),
                        os.path.join(log_path, "attacker4simple.py"))
    except Exception as e:
        print(e)


def gettensor(batch, device, batchfirst=False):
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).to(device), \
        torch.tensor(labels, dtype=torch.long).to(device)
    if batchfirst:
        #         inputs_pos = [[pos_i + 1 if w_i != 0 else 0 for pos_i, w_i in enumerate(inst)] for inst in inputs]
        #         inputs_pos = torch.tensor(inputs_pos, dtype=torch.long).to(device)
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels


def evaluate(classifier, dataset, device, batch_size=128):
    """
        Evaluate the classifier on the dataset
        epoch=0: stands for individual testing
        dataset: Test set/validation set
        log_path: Log path
        batch_size: Batch size, lstm, astnn can be set larger to 128
    """
    criterion = nn.CrossEntropyLoss()
    classifier = classifier.to(device)  # Move the classifier to the specified device

    classifier.eval()
    test_num = 0
    test_correct = 0
    total_loss = 0
    while True:
        batch = dataset.next_batch(batch_size)
        if batch['new_epoch']:
            break
        inputs, labels = gettensor(batch, device, batchfirst=False)
        inputs = inputs.to(device)  # Move input data to the specified device
        labels = labels.to(device)  # Move label data to specified device
        with torch.no_grad():
            outputs = classifier(inputs)[0]
            loss = criterion(outputs, labels)
            res = torch.argmax(outputs, dim=1) == labels
            test_correct += torch.sum(res)
            test_num += len(labels)
        total_loss += loss.item()
    # logging
    return float(test_correct) * 100.0 / test_num
