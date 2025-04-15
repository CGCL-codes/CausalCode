# -*- coding: utf-8 -*-

import copy

from dataset import OJ104, CodeChef
from modifier import TokenModifier, InsModifier
from modifier import get_batched_data

import numpy
import random
import torch
import torch.nn as nn
import argparse
import pickle, gzip
import os, sys, time
import build_dataset as bd


class AlertToken(object):
    """
        AlertToken
        Attack success and maximum disturbance
    """

    def __init__(self, dataset, symtab, classifier):

        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=self.txt2idx,
                                    idx2txt=self.idx2txt)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab

    def gen_one(self, x_raw, y, uids, n_candidate=5, n_iter=40):
        iter = 0
        n_stop = 0
        batch = get_batched_data([x_raw], [y], self.txt2idx)
        with torch.no_grad():  # Disable gradient calculation
            old_prob = self.cl.prob(batch['x'])[0]
        if torch.argmax(old_prob) != y:
            print("SUCC! Original mistake.")
            return True, x_raw, 0
        old_prob = old_prob[y]
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                # don't waste iteration on the "<unk>"s
                assert not k.startswith('Ġ')
                Gk = 'Ġ' + k
                Gk_idx = self.cl.tokenizer.convert_tokens_to_ids(Gk)
                if Gk_idx == self.cl.tokenizer.unk_token_id:
                    continue
                iter += 1
                new_x_raw, new_x_uid = self.tokenM.rename_uid(x_raw, y, k, n_candidate)
                if new_x_raw is None:
                    n_stop += 1
                    continue
                batch = get_batched_data(new_x_raw, [y] * len(new_x_raw), self.txt2idx)
                with torch.no_grad():  # Disable gradient calculation
                    new_prob = self.cl.prob(batch['x'])
                new_pred = torch.argmax(new_prob, dim=-1)
                for uid, p, pr, _x in zip(new_x_uid, new_pred, new_prob, new_x_raw):
                    if p != y:
                        print("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                              (k, uid, y, old_prob, y, pr[y], p, pr[p]))
                        return True, [_x], 1
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
                    uids[new_x_uid[new_prob_idx]] = uids.pop(k)
                    n_stop = 0
                    print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                          (k, new_x_uid[new_prob_idx], y, old_prob, y, new_prob[new_prob_idx][y]))
                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1
                del new_prob_idx
                del new_x_raw
                del new_prob
                torch.cuda.empty_cache()
        return False, x_raw, 2

    def gen_all(self, n_candidate=5, n_iter=40, res_save=None):

        n_succ = 0
        n_total = 0
        total_time = 0

        st_time = time.time()
        for i in range(self.d.train.get_size()):
            b = self.d.train.next_batch(1)
            tag, x_raw, typ = self.gen_one(b['raw'][0], b['y'][0], self.syms['tr'][b['id'][0]], n_candidate, n_iter)
            if i % 1000 == 0 and i > 0:
                total_time += time.time() - st_time
                print("Avg time cost = %.1f sec" % (total_time / (i + 1)))
                print("Time Cost: %.1f " % (time.time() - st_time))
                # current time
                print("Now time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            while len(x_raw) == 1:
                x_raw = x_raw[0]
            dd['x_tr'][b['id'][0]] = bd.text2index([x_raw], self.txt2idx)[0]
            dd['raw_tr'][b['id'][0]] = x_raw
            n_total += 1
        os.makedirs(os.path.dirname(res_save), exist_ok=True)
        if res_save is not None:
            with gzip.open(res_save, "wb") as f:
                pickle.dump(dd, f)
                print("Save to %s" % res_save)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.train.get_size()))
        write_gen_data_time(res_save, st_time, self.d.train.get_size())


class AlertDeadCode(object):

    def __init__(self, dataset, instab, classifier):

        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=self.txt2idx,
                                idx2txt=self.idx2txt,
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab

    # only support single x: a token-idx list
    def attack(self, x_raw, y, poses, n_candidate=100, n_iter=20):

        self.insM.initInsertDict(poses)
        iter = 0
        n_stop = 0
        batch = get_batched_data([x_raw], [y], self.txt2idx)
        with torch.no_grad():  # Disable gradient calculation
            old_prob = self.cl.prob(batch['x'])[0]
        if torch.argmax(old_prob) != y:
            print("SUCC! Original mistake.")
            return True, x_raw, 0
        old_prob = old_prob[y]

        while iter < n_iter:
            iter += 1
            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_raw_del, new_insertDict_del = self.insM.remove(x_raw, n_candidate_del)
            new_x_raw_add, new_insertDict_add = self.insM.insert(x_raw, n_candidate_ins)
            new_x_raw = new_x_raw_del + new_x_raw_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x_raw == []:  # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            batch = get_batched_data(new_x_raw, [y] * len(new_x_raw), self.txt2idx)
            with torch.no_grad():
                new_prob = self.cl.prob(batch['x'])
            new_pred = torch.argmax(new_prob, dim=-1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x_raw):
                if p != y:
                    print("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                          (self.insM.insertDict["count"], insD["count"],
                           y, old_prob, y, pr[y], p, pr[p]))
                    return True, [_x], 1

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:
                x_raw = new_x_raw[new_prob_idx]  # I just added this sentence, I don’t know if there is any problem
                print("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                      (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"],
                       y, old_prob, y, new_prob[new_prob_idx][y]))
                self.insM.insertDict = new_insertDict[new_prob_idx]  # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1

            if n_stop >= len(new_x_raw):  # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
            del new_prob_idx
            del new_x_raw
            del new_prob
            torch.cuda.empty_cache()

        return False, x_raw, 2

    def attack_all(self, n_candidate=5, n_iter=40, res_save=None):

        n_succ = 0
        total_time = 0
        n_total = 0
        st_time = time.time()
        for i in range(self.d.train.get_size()):
            b = self.d.train.next_batch(1)
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.train.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, x_raw, typ = self.attack(b['raw'][0], b['y'][0], self.inss['stmt_tr'][b['id'][0]], n_candidate, n_iter)
            if i % 1000 == 0 and i > 0:
                total_time += time.time() - st_time
                print("Avg time cost = %.1f sec" % (total_time / (i + 1)))
                print("Time Cost: %.1f " % (time.time() - st_time))
                # current time
                print("Now time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            while len(x_raw) == 1:
                x_raw = x_raw[0]
            dd['x_tr'][b['id'][0]] = bd.text2index([x_raw], self.txt2idx)[0]
            dd['raw_tr'][b['id'][0]] = x_raw
            n_total += 1
        os.makedirs(os.path.dirname(res_save), exist_ok=True)
        write_gen_data_time(res_save, st_time, self.d.train.get_size())
        os.makedirs(os.path.dirname(res_save), exist_ok=True)
        if res_save is not None:
            with gzip.open(res_save, "wb") as f:
                pickle.dump(dd, f)
                print("Save to %s" % res_save)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.train.get_size()))


class CausalCodeToken(object):

    def __init__(self, dataset, symtab, classifier):

        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=self.txt2idx,
                                    idx2txt=self.idx2txt)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab

    def gen_one(self, x_raw, y, uids, n_candidate=5, n_iter=20):
        iter = 0
        n_stop = 0
        stop_iter = random.randint(3, n_iter)
        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue
                # don't waste iteration on the "<unk>"s
                assert not k.startswith('Ġ')
                Gk = 'Ġ' + k
                Gk_idx = self.cl.tokenizer.convert_tokens_to_ids(Gk)
                if Gk_idx == self.cl.tokenizer.unk_token_id:
                    continue
                iter += 1
                new_x_raw, new_x_uid = self.tokenM.rename_uid(x_raw, y, k, n_candidate)
                if new_x_raw is None:
                    n_stop += 1
                    continue
                if iter == stop_iter:
                    new_prob_idx = random.randint(0, len(new_x_raw) - 1)
                    x_raw = new_x_raw[new_prob_idx]
                    return [x_raw], 1

                new_prob_idx = random.randint(0, len(new_x_raw) - 1)
                x_raw = new_x_raw[new_prob_idx]
                uids[new_x_uid[new_prob_idx]] = uids.pop(k)
        return [x_raw], 0

    def gen_all(self, n_candidate=5, n_iter=40, res_save=None):
        n_succ = 0
        st_time = time.time()
        total_time = 0
        with gzip.open(os.path.dirname(os.path.dirname(res_save)) + "/origin/data.pkl.gz", "rb") as f:
            dd = pickle.load(f)
        print("Start generating examples...")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        total_samples = self.d.train.get_size()
        for i in range(total_samples):
            b = self.d.train.next_batch(1)
            x_raw, typ = self.gen_one(b['raw'][0], b['y'][0], self.syms['tr'][b['id'][0]], n_candidate, n_iter)
            if i % 500 == 0 and i > 0:
                total_time += time.time() - st_time
                print('Current progress：%.2f%%' % ((i + 1) / total_samples * 100))
                print("Avg time cost = %.1f sec" % (total_time / (i + 1)))
                print("Time Cost: %.1f " % (time.time() - st_time))
                # current time
                print("Now time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            while len(x_raw) == 1:
                x_raw = x_raw[0]
            dd['x_tr'][b['id'][0]] = bd.text2index([x_raw], self.txt2idx)[0]
            dd['raw_tr'][b['id'][0]] = x_raw
        os.makedirs(os.path.dirname(res_save), exist_ok=True)
        if res_save is not None:
            print("Save to %s" % res_save)
            with gzip.open(res_save, "wb") as f:
                pickle.dump(dd, f)
                print("success save file")
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.train.get_size()))
        write_gen_data_time(res_save, st_time, self.d.train.get_size())


class CausalCodeDeadCode(object):
    def __init__(self, dataset, instab, classifier):
        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=self.txt2idx,
                                idx2txt=self.idx2txt,
                                poses=None)
        self.d = dataset
        self.inss = instab

    # only support single x: a token-idx list
    def attack(self, x_raw, y, poses, n_candidate=5, n_iter=40):
        self.insM.initInsertDict(poses)
        iter = 0
        n_stop = 0

        stop_iter = random.randint(3, n_iter)

        old_raw = copy.deepcopy(x_raw)
        while iter < n_iter:
            iter += 1
            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_raw_del, new_insertDict_del = self.insM.remove(x_raw, n_candidate_del)
            new_x_raw_add, new_insertDict_add = self.insM.insert(x_raw, n_candidate_ins)
            new_x_raw = new_x_raw_del + new_x_raw_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if not new_x_raw:  # no valid candidates
                n_stop += 1
                continue
            if iter == stop_iter:
                # Randomly select one from new_x as the final result
                x_raw = random.choice(new_x_raw)
                return True, [x_raw], 1
            # find if there is any candidate successful wrong classfied
            new_idx = random.randint(0, len(new_x_raw) - 1)
            self.insM.insertDict = new_insertDict[new_idx]  # don't forget this step
        return False, [x_raw], 2

    def gen_all(self, n_candidate=5, n_iter=40, res_save=None):

        n_succ = 0
        total_time = 0
        st_time = time.time()
        original_mistake_num = 0
        attack_success_num = 0
        with gzip.open(os.path.dirname(os.path.dirname(res_save)) + "/origin/data.pkl.gz", "rb") as f:
            dd = pickle.load(f)
        total_samples = self.d.train.get_size()
        for i in range(total_samples):
            b = self.d.train.next_batch(1)
            start_time = time.time()
            right_result, x_raw, asr_flag = self.attack(b['raw'][0], b['y'][0], self.inss['stmt_tr'][b['id'][0]],
                                                        n_candidate,
                                                        n_iter)

            while len(x_raw) == 1:
                x_raw = x_raw[0]
            dd['x_tr'][b['id'][0]] = bd.text2index([x_raw], self.txt2idx)[0]
            dd['raw_tr'][b['id'][0]] = x_raw
            if i % 200 == 0:
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                print('Current progress:%.2f%%' % ((i + 1) / total_samples * 100))
                print("----------%d samples have been generated" % i)
                print('time cost:', time.time() - start_time)
        os.makedirs(os.path.dirname(res_save), exist_ok=True)
        if res_save is not None:
            write_gen_data_time(res_save, st_time, self.d.train.get_size())

            with gzip.open(res_save, "wb") as f:
                pickle.dump(dd, f)
                print("success save file")
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.train.get_size()))


class CarrotToken(object):

    def __init__(self, dataset, symtab, classifier):

        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=self.txt2idx,
                                    idx2txt=self.idx2txt)
        self.cl = classifier
        self.d = dataset
        self.syms = symtab

    def attack(self, x_raw, y, uids, n_candidate=5, n_iter=40):
        ori_x = copy.deepcopy(x_raw)
        iter = 0
        n_stop = 0
        batch = get_batched_data([x_raw], [y], self.txt2idx)
        with torch.no_grad():  # Disable gradient calculation
            old_prob = self.cl.prob(batch['x'])[0]
        if torch.argmax(old_prob) != y:
            return True, x_raw, False
        old_prob = old_prob[y]

        while iter < n_iter:
            keys = list(uids.keys())
            for k in keys:
                if iter >= n_iter:
                    break
                if n_stop >= len(uids):
                    iter = n_iter
                    break
                if k in self.tokenM.forbidden_uid:
                    n_stop += 1
                    continue

                # don't waste iteration on the "<unk>"s
                assert not k.startswith('Ġ')
                Gk = 'Ġ' + k
                Gk_idx = self.cl.tokenizer.convert_tokens_to_ids(Gk)
                if Gk_idx == self.cl.tokenizer.unk_token_id:
                    continue

                iter += 1
                new_x_raw, new_x_uid = self.tokenM.rename_uid(x_raw, y, k, n_candidate)
                if new_x_raw is None:
                    n_stop += 1
                    continue
                batch = get_batched_data(new_x_raw, [y] * len(new_x_raw), self.txt2idx)
                with torch.no_grad():  # Disable gradient calculation
                    new_prob = self.cl.prob(batch['x'])
                new_pred = torch.argmax(new_prob, dim=-1)
                for uid, p, pr, _x in zip(new_x_uid, new_pred, new_prob, new_x_raw):
                    if p != y:
                        return True, [_x], True
                new_prob_idx = torch.argmin(new_prob[:, y])
                if new_prob[new_prob_idx][y] < old_prob:
                    x_raw = new_x_raw[new_prob_idx]
                    uids[new_x_uid[new_prob_idx]] = uids.pop(k)
                    n_stop = 0

                    old_prob = new_prob[new_prob_idx][y]
                else:
                    n_stop += 1

        return False, ori_x, False

    def attack_all(self, n_candidate=100, n_iter=20, res_save=None, adv_sample_size=2000):
        n_succ = 0
        total_time = 0
        st_time = time.time()
        adv_xs, adv_labels, adv_ids, adv_raw = [], [], [], []
        attack_success_num = 0
        original_mistake_num = 0

        for i in range(self.d.train.get_size()):
            if len(adv_xs) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)

            start_time = time.time()
            right_result, x, asr_flag = self.attack(b['raw'][0], b['y'][0], self.syms['tr'][b['id'][0]], n_candidate,
                                                    n_iter)
            while len(x) == 1:
                x = x[0]

            if right_result:
                n_succ += 1
                total_time += time.time() - start_time
                if not asr_flag:
                    original_mistake_num += 1
            if asr_flag:
                attack_success_num += 1
                adv_xs.append(bd.text2index([x], self.txt2idx)[0])
                adv_labels.append(int(b['y'][0]))
                adv_ids.append(b['id'][0])
                adv_raw.append(x)

            if i % 500 == 0 and i > 0:
                total_time += time.time() - st_time
                print("Avg time cost = %.1f sec" % (total_time / (i + 1)))
                print("Time Cost: %.1f " % (time.time() - st_time))
                # current time
                print("Now time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print("attack_success_num: %d, original_mistake_num: %d" % (attack_success_num, original_mistake_num))
        os.makedirs(os.path.dirname(res_save), exist_ok=True)
        if res_save is not None:
            write_gen_data_time(res_save, st_time, self.d.train.get_size())
            print("Adversarial Sample Number: %d " % (len(adv_xs)))
            with gzip.open(res_save, "wb") as f:
                pickle.dump({
                    "adv_x": adv_xs,
                    "adv_raw": adv_raw,
                    "adv_label": adv_labels,
                    "adv_id": adv_ids}, f)
                print("Save to %s" % res_save)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.train.get_size()))


class CarrotDeadCode(object):

    def __init__(self, dataset, instab, classifier):

        self.txt2idx = dataset.get_txt2idx()
        self.idx2txt = dataset.get_idx2txt()
        self.insM = InsModifier(classifier=classifier,
                                txt2idx=self.txt2idx,
                                idx2txt=self.idx2txt,
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab

    # only support single x: a token-idx list
    def attack(self, x_raw, y, poses, n_candidate=100, n_iter=20):
        ori_x = copy.deepcopy(x_raw)
        self.insM.initInsertDict(poses)

        iter = 0
        n_stop = 0

        batch = get_batched_data([x_raw], [y], self.txt2idx)
        with torch.no_grad():  # Disable gradient calculation
            old_prob = self.cl.prob(batch['x'])[0]
        if torch.argmax(old_prob) != y:
            return True, x_raw, False
        old_prob = old_prob[y]

        while iter < n_iter:
            iter += 1

            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_raw_del, new_insertDict_del = self.insM.remove(x_raw, n_candidate_del)
            new_x_raw_add, new_insertDict_add = self.insM.insert(x_raw, n_candidate_ins)
            new_x_raw = new_x_raw_del + new_x_raw_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if not new_x_raw:  # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            batch = get_batched_data(new_x_raw, [y] * len(new_x_raw), self.txt2idx)
            with torch.no_grad():  # Disable gradient calculation
                new_prob = self.cl.prob(batch['x'])
            new_pred = torch.argmax(new_prob, dim=-1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x_raw):
                if p != y:
                    return True, [_x], True

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y])
            if new_prob[new_prob_idx][y] < old_prob:

                self.insM.insertDict = new_insertDict[new_prob_idx]  # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y]
            else:
                n_stop += 1

            if n_stop >= len(new_x_raw):  # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break

        return False, ori_x, False

    def attack_all(self, n_candidate=100, n_iter=20, res_save=None, adv_sample_size=2000):

        n_succ = 0
        total_time = 0
        st_time = time.time()
        original_mistake_num = 0
        attack_success_num = 0
        adv_xs, adv_labels, adv_ids, adv_raw = [], [], [], []
        for i in range(self.d.train.get_size()):
            if len(adv_xs) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)

            start_time = time.time()

            right_result, x, asr_flag = self.attack(b['raw'][0], b['y'][0], self.inss['stmt_tr'][b['id'][0]],
                                                    n_candidate,
                                                    n_iter)
            while len(x) == 1:
                x = x[0]
            # dd['x_tr'][b['id'][0]] = bd.text2index([x], self.txt2idx)[0]
            # dd['raw_tr'][b['id'][0]] = x

            if right_result:
                n_succ += 1
                total_time += time.time() - start_time
                if not asr_flag:
                    original_mistake_num += 1
            if asr_flag:
                attack_success_num += 1
                adv_xs.append(bd.text2index([x], self.txt2idx)[0])
                adv_labels.append(int(b['y'][0]))
                adv_ids.append(b['id'][0])
                adv_raw.append(x)

            if i % 500 == 0 and i > 0:
                total_time += time.time() - st_time
                print("Avg time cost = %.1f sec" % (total_time / (i + 1)))
                print("Time Cost: %.1f " % (time.time() - st_time))
                # current time
                print("Now time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
                print("attack_success_num: %d, original_mistake_num: %d" % (attack_success_num, original_mistake_num))
        os.makedirs(os.path.dirname(res_save), exist_ok=True)
        if res_save is not None:
            write_gen_data_time(res_save, st_time, self.d.train.get_size())
            print("Adversarial Sample Number: %d " % (len(adv_xs)))
            with gzip.open(res_save, "wb") as f:
                pickle.dump({
                    "adv_x": adv_xs,
                    "adv_raw": adv_raw,
                    "adv_label": adv_labels,
                    "adv_id": adv_ids}, f)
                print("Save to %s" % res_save)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.train.get_size()))


def write_gen_data_time(res_save, st_time, size):
    log_path = os.path.dirname(res_save)
    with open(os.path.join(log_path, "aug_time_cost.txt"), 'a') as f:
        f.write("\n Time Cost: %.1f " % (time.time() - st_time))
        f.write("\n data size: %d " % size)


if __name__ == "__main__":

    root = '../data/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--enhance_method', default='CausalCode', help='CausalCode or carrot alert random')
    parser.add_argument('--model_name', type=str, default="graphcodebert")
    parser.add_argument('--task', default='code_function', help='code_function or code_defect')
    parser.add_argument('--attack_type', default='token', help='dead_code , token')
    parser.add_argument('--data', type=str, default=None)

    # CausalCode Sweep

    parser.add_argument('--iter_select', type=int, default=30, help='Number of iterations:5 10 20')
    parser.add_argument('--index', type=int, default=0, help='Expanded corner mark:0 1 2 3 4')
    parser.add_argument('--wandb_name', type=str, default='1-all-asr-graphcodebert-code_function-CausalCode-token',
                        help='The name of wandb, naming rules: experimental purpose, QA few')

    parser.add_argument('--data_name', type=str, default='')
    parser.add_argument('--begin_epoch', type=int, default=15, help='Starting epoch')
    opt = parser.parse_args()

    model_path = "../transformers/microsoft/graphcodebert-base"

    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
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
    rand_seed = 1726

    torch.manual_seed(rand_seed)
    random.seed(rand_seed)
    numpy.random.seed(rand_seed)

    # Extended sample name
    if opt.enhance_method == 'CausalCode':
        opt.data_name = opt.model_name + '-' + opt.enhance_method + '-' + opt.attack_type
        if opt.task == 'code_defect':
            opt.data_name = opt.data_name + '-index-' + str(opt.index)
    else:
        opt.data_name = opt.model_name + '-' + opt.enhance_method + '-' + opt.attack_type

    save_path = os.path.join(root, opt.task, "dataset", opt.data_name)
    print('data save at :', save_path)
    os.makedirs(save_path, exist_ok=True)

    from graphcodebert import GraphCodeBERTClassifier

    with gzip.open(root + opt.task + "/dataset/origin/data.pkl.gz", "rb") as f:
        dd = pickle.load(f)
    with gzip.open(root + opt.task + '/dataset/origin/data_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open(root + opt.task + '/dataset/origin/data_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)

    if opt.enhance_method == 'CausalCode' or opt.enhance_method == 'alert':
        if opt.task == 'code_defect' and opt.attack_type == 'token':
            model_path = os.path.join(root, opt.task, "model", opt.model_name, opt.enhance_method, opt.attack_type, 'graphcodebert-CausalCode-token-gen-data',
                                      str(opt.index) + ".pt")

        else:
            model_path = os.path.join(root, opt.task, "model", opt.model_name, 'origin', '15.pt')

    print("loading model_path:", model_path)
    classifier = GraphCodeBERTClassifier(model_path=model_path,
                                         num_labels=n_class,
                                         device=device).to(device)

    classifier.eval()

    if opt.enhance_method == 'carrot':
        print("CarrotToken  -----------------------")
        adv_save_path = os.path.join(root, opt.task, 'data',
                                     opt.model_name + '_' + opt.enhance_method + '_' + opt.attack_type)
        print(adv_save_path)
        if opt.attack_type == 'dead_code':
            atk = CarrotDeadCode(original_dataset, instab, classifier)
            os.makedirs(adv_save_path, exist_ok=True)
            atk.attack_all(5, 40,
                           res_save=adv_save_path + "/data.pkl.gz",
                           adv_sample_size=2000)
        elif opt.attack_type == 'token':
            atk = CarrotToken(original_dataset, symtab, classifier)
            os.makedirs(adv_save_path, exist_ok=True)
            atk.attack_all(5, 40,
                           res_save=adv_save_path + "/data.pkl.gz",
                           adv_sample_size=2000)
    elif opt.enhance_method == 'alert':
        gen_data_save_path = os.path.join(root, opt.task, 'data',
                                          opt.model_name + '_' + opt.enhance_method + '_' + opt.attack_type)
        if opt.attack_type == 'dead_code':
            os.makedirs(gen_data_save_path, exist_ok=True)
            print("gen_data_save_path:", gen_data_save_path)
            atk = AlertDeadCode(original_dataset, instab, classifier)
            atk.attack_all(5, 40, res_save=gen_data_save_path + "/data.pkl.gz")
        elif opt.attack_type == 'token':
            os.makedirs(gen_data_save_path, exist_ok=True)
            print("gen_data_save_path:", gen_data_save_path)
            atk = AlertToken(original_dataset, symtab, classifier)
            atk.gen_all(5, 40, res_save=gen_data_save_path + "/data.pkl.gz")
    elif opt.enhance_method == 'CausalCode':
        if opt.attack_type == 'dead_code':
            atk = CausalCodeDeadCode(original_dataset, instab, classifier)
            os.makedirs(save_path, exist_ok=True)
            atk.gen_all(5, 30, res_save=save_path + "/data.pkl.gz")
        elif opt.attack_type == 'token':
            gen = CausalCodeToken(original_dataset, symtab, classifier)
            os.makedirs(save_path, exist_ok=True)
            gen.gen_all(5, 30, res_save=save_path + "/data.pkl.gz")
    elif opt.enhance_method == 'random':
        if opt.attack_type == 'dead_code':
            atk = CausalCodeDeadCode(original_dataset, instab, classifier)
            os.makedirs(save_path, exist_ok=True)
            atk.gen_all(5, 1, res_save=save_path + "/data.pkl.gz")
        elif opt.attack_type == 'token':
            gen = CausalCodeToken(original_dataset, symtab, classifier)
            os.makedirs(save_path, exist_ok=True)
            gen.gen_all(5, 1, res_save=save_path + "/data.pkl.gz")
