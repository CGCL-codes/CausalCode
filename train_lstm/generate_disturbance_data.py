# -*- coding: utf-8 -*-

import shutil

from dataset import OJ104, remove_tail_padding
from modifier import TokenModifier, InsModifier

import torch
import pickle, gzip
import os, time, copy
import build_dataset as bd
import pattern
import random
from copy import deepcopy


class AdversarialTrainingAttacker(object):

    def __init__(self, dataset, symtab, classifier):

        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.cl = classifier
        self.d = dataset
        self.syms = symtab

    def attack(self, x, y, uids, n_candidate=100, n_iter=20):

        ori_x = copy.deepcopy(x)
        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).to(device).permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print("SUCC! Original mistake.")
            return True, x, 0
        old_prob = old_prob[y[0]]
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
                iter += 1
                new_x, new_uid_cand = self.tokenM.rename_uid(x, y, uids[k], k, n_candidate)
                if new_x is None:
                    n_stop += 1
                    print("skip unk\t%s" % k)
                    continue
                new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).to(device).permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        print("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                              (k, self.d.idx2vocab(uid), y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                        return True, [_x], 1
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                          (k, self.d.idx2vocab(int(new_uid_cand[new_prob_idx])),
                           y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
                    print("rej\t%s" % k)
        print("FAIL!")
        return False, ori_x, 2

    def attack_all(self, n_candidate=100, n_iter=20, res_save=None, adv_sample_size=5000):

        n_succ, n_total = 0, 0
        total_time = 0
        adv_xs, adv_labels, adv_ids = [], [], []
        fail_pred_xs, fail_pred_labels, fail_pred_ids = [], [], []
        st_time = time.time()
        for i in range(self.d.train.get_size()):
            if len(adv_xs) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.train.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, x, typ = self.attack(b['x'], b['y'], self.syms['tr'][b['id'][0]], n_candidate, n_iter)
            x = x[0]
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                fail_pred_xs.append(x)
                fail_pred_labels.append(int(b['y'][0]))
                fail_pred_ids.append(b['id'][0])
            if typ == 1:
                adv_xs.append(x)
                adv_labels.append(int(b['y'][0]))
                adv_ids.append(b['id'][0])
            if n_succ <= 0:
                print("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                      % (n_succ / (i + 1)), flush=True)
            else:
                print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                      % (n_succ / (i + 1), total_time / n_succ), flush=True)
            n_total += 1
        if res_save is not None:
            print("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_xs), len(fail_pred_xs)))
            with gzip.open(res_save, "wb") as f:
                unpadding_adv_xs = [remove_tail_padding(adv_x, 0) for adv_x in adv_xs]
                pickle.dump({"fail_pred_x": fail_pred_xs,
                             "fail_pred_label": fail_pred_labels,
                             "fail_pred_id": fail_pred_ids,
                             "adv_x": adv_xs,
                             "adv_raw": self.d.idxs2raw(unpadding_adv_xs, [len(x) for x in unpadding_adv_xs]),
                             "adv_label": adv_labels,
                             "adv_id": adv_ids}, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time() - st_time, n_succ / n_total))


class AdversarialTrainingAttackerRandom(object):

    def __init__(self, dataset, symtab, classifier):

        self.tokenM = TokenModifier(classifier=classifier,
                                    loss=torch.nn.CrossEntropyLoss(),
                                    uids=symtab['all'],
                                    txt2idx=dataset.get_txt2idx(),
                                    idx2txt=dataset.get_idx2txt())
        self.cl = classifier
        self.d = dataset
        self.syms = symtab

    def attack(self, x, y, uids, n_iter=20):

        ori_x = copy.deepcopy(x)
        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).to(device).permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print("SUCC! Original mistake.")
            return True, x, 0
        old_prob = old_prob[y[0]]
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
                iter += 1
                new_x, new_uid_cand = self.tokenM.rename_uid_random(x, uids[k], k)
                if new_x is None:
                    n_stop += 1
                    print("skip unk\t%s" % k)
                    continue
                new_prob = self.cl.prob(torch.tensor(new_x, dtype=torch.long).to(device).permute([1, 0]))
                new_pred = torch.argmax(new_prob, dim=1)
                for uid, p, pr, _x in zip(new_uid_cand, new_pred, new_prob, new_x):
                    if p != y[0]:
                        print("SUCC!\t%s => %s\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                              (k, self.d.idx2vocab(uid), y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                        return True, [_x], 1
                new_prob_idx = torch.argmin(new_prob[:, y[0]])
                if new_prob[new_prob_idx][y[0]] < old_prob:
                    x = [new_x[new_prob_idx]]
                    uids[self.d.idx2vocab(int(new_uid_cand[new_prob_idx]))] = uids.pop(k)
                    n_stop = 0
                    print("acc\t%s => %s\t\t%d(%.5f) => %d(%.5f)" % \
                          (k, self.d.idx2vocab(int(new_uid_cand[new_prob_idx])),
                           y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                    old_prob = new_prob[new_prob_idx][y[0]]
                else:
                    n_stop += 1
                    print("rej\t%s" % k)
        print("FAIL!")
        return False, ori_x, 2

    def attack_all(self, n_iter=20, res_save=None, adv_sample_size=5000):

        n_succ, n_total = 0, 0
        total_time = 0
        adv_xs, adv_labels, adv_ids = [], [], []
        fail_pred_xs, fail_pred_labels, fail_pred_ids = [], [], []
        st_time = time.time()
        for i in range(self.d.train.get_size()):
            if len(adv_xs) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.train.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, x, typ = self.attack(b['x'], b['y'], self.syms['tr'][b['id'][0]], n_iter)
            x = x[0]
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                fail_pred_xs.append(x)
                fail_pred_labels.append(int(b['y'][0]))
                fail_pred_ids.append(b['id'][0])
            if typ == 1:
                adv_xs.append(x)
                adv_labels.append(int(b['y'][0]))
                adv_ids.append(b['id'][0])
            if n_succ <= 0:
                print("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                      % (n_succ / (i + 1)), flush=True)
            else:
                print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                      % (n_succ / (i + 1), total_time / n_succ), flush=True)
            n_total += 1
        if res_save is not None:
            print("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_xs), len(fail_pred_xs)))
            with gzip.open(res_save, "wb") as f:
                unpadding_adv_xs = [remove_tail_padding(adv_x, 0) for adv_x in adv_xs]
                pickle.dump({"fail_pred_x": fail_pred_xs,
                             "fail_pred_label": fail_pred_labels,
                             "fail_pred_id": fail_pred_ids,
                             "adv_x": adv_xs,
                             "adv_raw": self.d.idxs2raw(unpadding_adv_xs, [len(x) for x in unpadding_adv_xs]),
                             "adv_label": adv_labels,
                             "adv_id": adv_ids}, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (time.time() - st_time, n_succ / n_total))


class AdversarialTrainingInsAttacker(object):

    def __init__(self, dataset, instab, classifier):

        self.insM = InsModifier(classifier=classifier,
                                txt2idx=dataset.get_txt2idx(),
                                poses=None)  # wait to init when attack
        self.cl = classifier
        self.d = dataset
        self.inss = instab

    # only support single x: a token-idx list
    def attack(self, x, y, poses, n_candidate=100, n_iter=20):

        self.insM.initInsertDict(poses)

        ori_x = copy.deepcopy(x)
        iter = 0
        n_stop = 0
        old_prob = self.cl.prob(torch.tensor(x, dtype=torch.long).to(device).permute([1, 0]))[0]
        if torch.argmax(old_prob) != y[0]:
            print("SUCC! Original mistake.")
            return True, x, 0
        old_prob = old_prob[y[0]]
        while iter < n_iter:
            iter += 1
            # get insertion candidates
            n_could_del = self.insM.insertDict["count"]
            n_candidate_del = n_could_del
            n_candidate_ins = n_candidate - n_candidate_del
            assert n_candidate_del >= 0 and n_candidate_ins >= 0
            new_x_del, new_insertDict_del = self.insM.remove(x[0], n_candidate_del)
            new_x_add, new_insertDict_add = self.insM.insert(x[0], n_candidate_ins)
            new_x = new_x_del + new_x_add
            new_insertDict = new_insertDict_del + new_insertDict_add
            if new_x == []:  # no valid candidates
                n_stop += 1
                continue

            # find if there is any candidate successful wrong classfied
            feed_new_x = [_x[:self.cl.max_len] for _x in new_x]  # this step is important
            feed_tensor = torch.tensor(feed_new_x, dtype=torch.long)
            new_prob = self.cl.prob(feed_tensor.to(device).permute([1, 0]))
            new_pred = torch.argmax(new_prob, dim=1)
            for insD, p, pr, _x in zip(new_insertDict, new_pred, new_prob, new_x):
                if p != y[0]:
                    print("SUCC!\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f) %d(%.5f)" % \
                          (self.insM.insertDict["count"], insD["count"],
                           y[0], old_prob, y[0], pr[y[0]], p, pr[p]))
                    return True, [_x], 1

            # if not, get the one with the lowest target_label_loss
            new_prob_idx = torch.argmin(new_prob[:, y[0]])
            if new_prob[new_prob_idx][y[0]] < old_prob:
                print("acc\tinsert_n %d => %d\t\t%d(%.5f) => %d(%.5f)" % \
                      (self.insM.insertDict["count"], new_insertDict[new_prob_idx]["count"],
                       y[0], old_prob, y[0], new_prob[new_prob_idx][y[0]]))
                self.insM.insertDict = new_insertDict[new_prob_idx]  # don't forget this step
                n_stop = 0
                old_prob = new_prob[new_prob_idx][y[0]]
            else:
                n_stop += 1
                print("rej\t%s" % "")
            if n_stop >= len(new_x):  # len(new_x) could be smaller than n_candidate
                iter = n_iter
                break
        print("FAIL!")
        return False, ori_x, 2

    def attack_all(self, n_candidate=100, n_iter=20, res_save=None, adv_sample_size=5000):

        n_succ = 0
        total_time = 0
        adv_xs, adv_labels, adv_ids = [], [], []
        fail_pred_xs, fail_pred_labels, fail_pred_ids = [], [], []
        st_time = time.time()
        for i in range(self.d.train.get_size()):
            if len(adv_xs) >= adv_sample_size:
                break
            b = self.d.train.next_batch(1)
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.train.get_size(), b['id'][0], b['y'][0]))
            start_time = time.time()
            tag, x, typ = self.attack(b['x'], b['y'], self.inss['stmt_tr'][b['id'][0]], n_candidate, n_iter)
            x = x[0]
            if tag:
                n_succ += 1
                total_time += time.time() - start_time
                fail_pred_xs.append(x)
                fail_pred_labels.append(int(b['y'][0]))
                fail_pred_ids.append(b['id'][0])
            if typ == 1:
                adv_xs.append(x)
                adv_labels.append(int(b['y'][0]))
                adv_ids.append(b['id'][0])
            if n_succ <= 0:
                print("\tCurr succ rate = %.3f, Avg time cost = NaN sec" \
                      % (n_succ / (i + 1)), flush=True)
            else:
                print("\tCurr succ rate = %.3f, Avg time cost = %.1f sec" \
                      % (n_succ / (i + 1), total_time / n_succ), flush=True)
        if res_save is not None:
            print("Adversarial Sample Number: %d (Out of %d False Predicted Sample)" % (len(adv_xs), len(fail_pred_xs)))
            with gzip.open(res_save, "wb") as f:
                unpadding_adv_xs = [remove_tail_padding(adv_x, 0) for adv_x in adv_xs]
                pickle.dump({"fail_pred_x": fail_pred_xs,
                             "fail_pred_label": fail_pred_labels,
                             "fail_pred_id": fail_pred_ids,
                             "adv_raw": self.d.idxs2raw(unpadding_adv_xs, [len(x) for x in unpadding_adv_xs]),
                             "adv_x": adv_xs,
                             "adv_label": adv_labels,
                             "adv_id": adv_ids}, f)
        print("[Task Done] Time Cost: %.1f sec Succ Rate: %.3f" % (
            time.time() - st_time, n_succ / self.d.train.get_size()))


class InsDist(object):

    def __init__(self, max_len, txt2idx, poses=None):

        self.max_len = max_len
        self.txt2idx = txt2idx
        if poses != None:  # else you need to call initInsertDict later
            self.initInsertDict(poses)
        inserts = [
            ";",
            "{ }",
            "printf ( \"\" ) ;",
            "if ( false ) ;",
            "if ( true ) { }",
            "if ( false ) ; else { }",
            "if ( 0 ) ;",
            "if ( false ) { int cnt = 0 ; for ( int i = 0 ; i < 123 ; i ++ ) cnt += 1 ; }"
            "for ( int i = 0 ; i < 100 ; i ++ ) break ;",
            "for ( int i = 0 ; i < 0 ; i ++ ) { }"
            "while ( false ) ;",
            "while ( 0 ) ;",
            "while ( true ) break ;",
            "for ( int i = 0 ; i < 10 ; i ++ ) { for ( int j = 0 ; j < 10 ; j ++ ) break ; break ; }",
            "do { } while ( false ) ;"]

        self.inserts = [insert.split(" ") for insert in inserts]

    def initInsertDict(self, poses):
        self.insertDict = dict([(pos, []) for pos in poses])
        self.insertDict["count"] = 0

    def _insert2idxs(self, insert):
        idxs = []
        for t in insert:
            if self.txt2idx.get(t) is not None:
                idxs.append(self.txt2idx[t])
            else:
                idxs.append(self.txt2idx['<unk>'])
        return idxs

    # only support one piece of data each time: x is idx-list
    def insert(self, x, n_candidate=5):

        pos_candidates = pattern.InsAddCandidates(self.insertDict, self.max_len)  # exclude outlier poses
        n = len(pos_candidates)
        if n_candidate < n:
            candisIdx = random.sample(range(n), n_candidate)
        else:
            candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]  # sample max(n, n_candidate) poses

        new_x, new_insertDict = [], []
        for pos in pos_candidates:
            inst = random.sample(self.inserts, 1)[0]
            inst_idxs = self._insert2idxs(inst)
            _insertDict = deepcopy(self.insertDict)
            pattern.InsAdd(_insertDict, pos, inst_idxs)
            # print("pos:", pos, "=>", inst, "count", _insertDict["count"])
            _x = pattern.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)

        return new_x, new_insertDict

    # only support one piece of data each time: x is idx-list
    def remove(self, x, n_candidate=5):

        pos_candidates = pattern.InsDeleteCandidates(self.insertDict)  # e.g. [(pos0, 0), (pos0, 1), (pos1, 0), ...]
        n = len(pos_candidates)
        if n_candidate < n:
            candisIdx = random.sample(range(n), n_candidate)
        else:
            candisIdx = random.sample(range(n), n)
        pos_candidates = [pos_candidates[candiIds] for candiIds in candisIdx]

        new_x, new_insertDict = [], []
        for pos, inPosIdx in pos_candidates:
            _insertDict = deepcopy(self.insertDict)
            pattern.InsDelete(_insertDict, pos, inPosIdx)
            # print("pos:", pos, "=>", self.insertDict[pos][inPosIdx], _insertDict["count"])
            _x = pattern.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)

        return new_x, new_insertDict

    def insert_remove_random(self, x):

        new_x, new_insertDict = [], []
        fail_cnt = 0
        while True:
            if fail_cnt >= 10:  # in case of dead loop
                break
            if random.random() > 0.5:  # insert
                pos_candidates = pattern.InsAddCandidates(self.insertDict, self.max_len)  # exclude outlier poses
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand = random.sample(pos_candidates, 1)[0]
                inst = random.sample(self.inserts, 1)[0]
                inst_idxs = self._insert2idxs(inst)
                _insertDict = deepcopy(self.insertDict)
                pattern.InsAdd(_insertDict, pos_cand, inst_idxs)
            else:
                pos_candidates = pattern.InsDeleteCandidates(self.insertDict)
                if pos_candidates == []:
                    fail_cnt += 1
                    continue
                pos_cand, inPosIdx = random.sample(pos_candidates, 1)[0]
                _insertDict = deepcopy(self.insertDict)
                pattern.InsDelete(_insertDict, pos_cand, inPosIdx)
            _x = pattern.InsResult(x, _insertDict)
            new_x.append(_x)
            new_insertDict.append(_insertDict)
            break
        return new_x, new_insertDict


class InsDistRandom(object):

    def __init__(self, dataset, instab, max_len):

        self.insD = InsDist(max_len=max_len,
                            txt2idx=dataset.get_txt2idx(),
                            poses=None)  # wait to init when attack
        self.max_len = max_len
        self.d = dataset
        self.inss = instab

    # only support single x: a token-idx list
    def disturbance(self, x, y, poses, n_iter=10, parser=None):

        self.insD.initInsertDict(poses)

        ori_x = copy.deepcopy(x)

        new_x, new_insertDict = self.insD.insert(x[0])
        for _x in new_x:
            token_idx_list = remove_tail_padding([_x], 0)
            raw = self.d.idxs2raw(token_idx_list, [len(x) for x in token_idx_list])
            try:
                train_raw = ''
                for t in raw[0]:
                    train_raw += t + " "
                print(train_raw)
                parser.parse(train_raw)

                return True, [_x], 1
            except Exception as e:
                print(e)

        print("FAIL!")
        return False, ori_x, 2

    def disturbance_all(self, n_iter=20, res_save=None, dis_sample_size=5000):

        train_dis_xs, train_dis_labels, train_dis_ids = [], [], []
        test_dis_xs, test_dis_labels, test_dis_ids = [], [], []
        st_time = time.time()
        from pycparser import c_parser
        parser = c_parser.CParser()
        for i in range(self.d.train.get_size()):
            if len(train_dis_xs) >= dis_sample_size:
                break
            b = self.d.train.next_batch(1)
            train_raw = ''
            print(b['raw'])
            for t in b['raw'][0]:
                train_raw += t + " "
            print(train_raw)
            parser.parse(train_raw)

            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.train.get_size(), b['id'][0], b['y'][0]))
            tag, x, typ = self.disturbance(b['x'], b['y'], self.inss['stmt_tr'][b['id'][0]], n_iter, parser)
            x = x[0]
            train_dis_xs.append(x)
            train_dis_labels.append(int(b['y'][0]))
            train_dis_ids.append(b['id'][0])
        for i in range(self.d.dev.get_size()):
            b = self.d.dev.next_batch(1)
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.dev.get_size(), b['id'][0], b['y'][0]))
            tag, x, typ = self.disturbance(b['x'], b['y'], self.inss['stmt_tr'][b['id'][0]], n_iter, parser)
            x = x[0]
            train_dis_xs.append(x)
            train_dis_labels.append(int(b['y'][0]))
            train_dis_ids.append(b['id'][0])

        train_unpadding_dis_xs = [remove_tail_padding(dis_x, 0) for dis_x in train_dis_xs]
        train_dis_raw = self.d.idxs2raw(train_unpadding_dis_xs, [len(x) for x in train_unpadding_dis_xs])
        idx2txt, txt2idx = bd.build_vocab(train_dis_raw)
        train_tokens = bd.text2index(train_dis_raw, txt2idx)

        for i in range(self.d.test.get_size()):
            b = self.d.test.next_batch(1)
            print("\t%d/%d\tID = %d\tY = %d" % (i + 1, self.d.test.get_size(), b['id'][0], b['y'][0]))

            tag, x, typ = self.disturbance(b['x'], b['y'], self.inss['stmt_tr'][b['id'][0]], n_iter, parser)
            x = x[0]
            test_dis_xs.append(x)
            test_dis_labels.append(int(b['y'][0]))
            test_dis_ids.append(b['id'][0])

        test_unpadding_dis_xs = [remove_tail_padding(dis_x, 0) for dis_x in train_dis_xs]
        test_dis_raw = self.d.idxs2raw(test_unpadding_dis_xs, [len(x) for x in test_unpadding_dis_xs])
        test_tokens = bd.text2index(test_dis_raw, txt2idx)
        print("Disturbance time: %.2f" % (time.time() - st_time))

        if res_save is not None:
            with gzip.open(res_save, "wb") as f:
                pickle.dump({"raw_tr": train_dis_raw, "y_tr": train_dis_labels,
                             "x_tr": train_tokens,
                             "raw_te": test_dis_raw, "y_te": test_dis_labels,
                             "x_te": test_tokens,
                             "idx2txt": idx2txt, "txt2idx": txt2idx
                             }, f)


if __name__ == "__main__":
    vocab_size = 1000
    embedding_size = 512
    hidden_size = 600
    n_layers = 2
    num_classes = 104
    max_len = 300
    origin_domain = 'small_test'
    poj = OJ104(path='../dataset/' + origin_domain + '/oj.pkl.gz',
                max_len=max_len,
                vocab_size=vocab_size)
    training_set = poj.train
    valid_set = poj.dev
    test_set = poj.test
    root = '../dataset/' + origin_domain + '/'
    shutil.rmtree("../dataset/random_ins/")
    os.makedirs("../dataset/random_ins", exist_ok=True)
    target_path = "../dataset/random_ins/"
    with gzip.open('../dataset/' + origin_domain + '/data_uid.pkl.gz', "rb") as f:
        symtab = pickle.load(f)
    with gzip.open('../dataset/' + origin_domain + '/data_inspos.pkl.gz', "rb") as f:
        instab = pickle.load(f)
    # Copy files to target_path
    os.system("cp ../dataset/" + origin_domain + "/data_uid.pkl.gz " + target_path)
    os.system("cp ../dataset/" + origin_domain + "/data_inspos.pkl.gz " + target_path)

    atk = InsDistRandom(poj, instab, max_len)

    atk.disturbance_all(10, res_save="../dataset/random_ins/oj.pkl.gz", dis_sample_size=2000)

    # atk = AdversarialTrainingAttacker(poj, symtab, classifier)
    # atk.attack_all(40, 50, res_save="../human_eval/lstm_oj_uid_rename_atk_candi40_iter50.advsamples.pkl.gz", dis_sample_size=5000) #atk.attack_all(40, 50, res_save="../model/lstm/uid_rename_atk_candi40_iter50.advsamples.pkl.gz", adv_sample_size=5000)

    # atk = AdversarialTrainingInsAttacker(poj, instab, classifier)
    # atk.attack_all(40, 20, res_save="../human_eval/lstm_oj_stmt_insert_atk_candi40_iter20.advsamples.pkl.gz",
    #                adv_sample_size=5000)  # atk.attack_all(40, 20, res_save="../model/lstm/stmt_insert_atk_candi40_iter20.advsamples.pkl.gz", adv_sample_size=5000)

    # atk = AdversarialTrainingAttackerRandom(poj, symtab, classifier)
    # atk.attack_all(100, res_save="../human_eval/lstm_oj_uid_random_atk_iter100.advsamples.pkl.gz", adv_sample_size=5000)
