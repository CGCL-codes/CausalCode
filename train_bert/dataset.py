# -*- coding: utf-8 -*-

import os
import pickle, gzip
import random
import numpy
import copy


class Dataset(object):

    def __init__(self, xs=[], ys=[], raws=None, ids=None, idx2txt=[], txt2idx={},
                 vocab_size=5000, dtype=None):

        self.__dtype = dtype
        self.__vocab_size = vocab_size
        self.__idx2txt = idx2txt
        self.__txt2idx = txt2idx
        assert len(self.__idx2txt) == self.__vocab_size \
               and len(self.__txt2idx) == self.__vocab_size + 1
        self.__xs = []
        self.__raws = []
        self.__ys = []
        self.__ids = []
        if raws is None:
            assert len(xs) == len(ys)
            raws = [None for _ in ys]
        else:
            assert len(xs) == len(ys) and len(ys) == len(raws)
        if ids is None:
            ids = list(range(len(xs)))
        else:
            assert len(xs) == len(ids)
        for x, y, r, i in zip(xs, ys, raws, ids):
            self.__raws.append(r)
            self.__ys.append(y)
            self.__ids.append(i)
            self.__xs.append([])
            for t in x:
                if t >= self.__vocab_size:
                    self.__xs[-1].append('<unk>')
                else:
                    self.__xs[-1].append(self.__idx2txt[t])
        self.__ys = numpy.asarray(self.__ys, dtype=self.__dtype['int'])
        self.__ids = numpy.asarray(self.__ids, dtype=self.__dtype['int'])
        self.__size = len(self.__raws)

        assert self.__size == len(self.__raws) \
               and len(self.__raws) == len(self.__xs) \
               and len(self.__xs) == len(self.__ys) \
               and len(self.__ys) == len(self.__ids)

        self.__epoch = None
        self.reset_epoch()

    def reset_epoch(self):

        self.__epoch = random.sample(range(self.__size), self.__size)

    def next_batch(self, batch_size=32):

        batch = {"x": [], "y": [], "raw": [], "id": [], "new_epoch": False}
        assert batch_size <= self.__size
        if len(self.__epoch) < batch_size:
            batch['new_epoch'] = True
            self.reset_epoch()
        idxs = self.__epoch[:batch_size]
        self.__epoch = self.__epoch[batch_size:]
        batch['x'] = copy.deepcopy([" ".join(self.__xs[i]) for i in idxs])
        batch['y'] = numpy.take(self.__ys, indices=idxs, axis=0)
        batch['id'] = numpy.take(self.__ids, indices=idxs, axis=0)
        batch['raw'] = copy.deepcopy([self.__raws[i] for i in idxs])
        return batch

    def get_size(self):

        return self.__size

    def get_rest_epoch_size(self):

        return len(self.__epoch)


def remove_tail_padding(token_idx_ndarray, pad_idx):
    """
    :param token_idx_ndarray: numpy.ndarray
    The function of this code is to remove the trailing pad token (represented by `pad_idx`) from the given `token_idx_ndarray` (a NumPy array containing the marker index). It returns a new list containing the marker indices after removing the padding markers.

    Let's explain the implementation details of the code step by step:
    1. Create an empty `stack` list and a `token_idx_list` list, the latter containing elements converted from `token_idx_ndarray`.
    2. Enter the loop, as long as `token_idx_list` is not empty:
    3. Pop an element from `token_idx_list` and assign it to variable `t`.
    4. Check if `t` is equal to `pad_idx` (padding mark). If so, continue with the next loop, skipping the current iteration.
    5. If `t` is not equal to `pad_idx`, it means that a non-padded mark is found. Add it back to the `token_idx_list` and use the `break` statement to break out of the loop.
    6. After the loop ends, return the updated `token_idx_list`, in which the tail padding mark has been removed.
    The purpose of this code is to find the last non-filled token in `token_idx_ndarray` and return all the tokens after it. By iterating through the list, starting from the end, the code finds the first non-filled token and returns all tokens after it as a result.
    """
    stack = []
    token_idx_list = list(token_idx_ndarray)
    while token_idx_list:
        t = token_idx_list.pop()
        if t == pad_idx:
            continue
        else:
            token_idx_list.append(t)
            break
    return token_idx_list


class OJ104(object):

    def __init__(self, path='../dataset/oj.pkl.gz', vocab_size=-1,
                 valid_ratio=0.2, dtype='32', adv_train_path=None, adv_train_size=None):

        self.__dtypes = self.__dtype(dtype)

        with gzip.open(path, "rb") as f:
            d = pickle.load(f)

        if vocab_size > 0:
            self.__idx2txt = d['idx2txt'][:self.__vocab_size]
            self.__vocab_size = vocab_size
        else:
            self.__idx2txt = d['idx2txt']
            self.__vocab_size = len(self.__idx2txt)
        self.__txt2idx = {"<pad>": 0}
        for i, t in zip(range(self.__vocab_size), self.__idx2txt):
            self.__txt2idx[t] = i
        random.seed(666)
        idxs = random.sample(range(len(d['x_tr'])), len(d['x_tr']))
        n_valid = int(len(d['x_tr']) * valid_ratio)
        raw, x, y, ids = [], [], [], []
        for i in idxs[:n_valid]:
            raw.append(d['raw_tr'][i])
            x.append(d['x_tr'][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        self.dev = Dataset(xs=x, ys=y, raws=raw, ids=ids,
                           idx2txt=self.__idx2txt,
                           txt2idx=self.__txt2idx,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        raw, x, y, ids = [], [], [], []
        for i in idxs[n_valid:]:
            raw.append(d['raw_tr'][i])
            x.append(d['x_tr'][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        if adv_train_path is not None:  # add adversarial samples for training
            raw = None
            with gzip.open(adv_train_path, "rb") as f:
                tmp_d = pickle.load(f)
                adv_x = tmp_d["adv_x"]
                adv_y = tmp_d["adv_label"]
            if adv_train_size is not None:
                print("adv_train_size:", adv_train_size)
                tmp_idxs = random.sample(range(len(adv_x)), len(adv_x))
                adv_x_ = [adv_x[i] for i in tmp_idxs]
                adv_y_ = [adv_y[i] for i in tmp_idxs]
                adv_x, adv_y = adv_x_, adv_y_
            for _x, _y in zip(adv_x, adv_y):
                ids.append(ids[-1] + 1)
                x.append(remove_tail_padding(_x, 0))  # token idx 0 used as padding.
                y.append(_y)
            print("[Adversarial Training] adversarial sample number: %d" % len(adv_x), flush=True)
        self.train = Dataset(xs=x, ys=y, raws=raw, ids=ids,
                             idx2txt=self.__idx2txt,
                             txt2idx=self.__txt2idx,
                             vocab_size=self.__vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(xs=d['x_te'],
                            ys=d['y_te'],
                            raws=d['raw_te'],
                            idx2txt=self.__idx2txt,
                            txt2idx=self.__txt2idx,
                            vocab_size=self.__vocab_size,
                            dtype=self.__dtypes)

    def __dtype(self, dtype='32'):

        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_dtype(self):

        return self.__dtypes

    def get_vocab_size(self):

        return self.__vocab_size

    def get_idx2txt(self):

        return copy.deepcopy(self.__idx2txt)

    def get_txt2idx(self):

        return copy.deepcopy(self.__txt2idx)

    def vocab2idx(self, vocab):

        if vocab in self.__txt2idx.keys():
            return self.__txt2idx[vocab]
        else:
            return self.__txt2idx['<unk>']

    def idx2vocab(self, idx):

        if 0 <= idx < len(self.__idx2txt):
            return self.__idx2txt[idx]
        else:
            return '<unk>'


class CodeChef(object):

    def __init__(self, path='../dataset/data.pkl.gz', vocab_size=-1,
                 valid_ratio=0.2, dtype='32', adv_train_path=None, adv_train_size=None):

        self.__dtypes = self.__dtype(dtype)

        with gzip.open(path, "rb") as f:
            d = pickle.load(f)

        if vocab_size > 0:
            self.__idx2txt = d['idx2txt'][:self.__vocab_size]
            self.__vocab_size = vocab_size
        else:
            self.__idx2txt = d['idx2txt']
            self.__vocab_size = len(self.__idx2txt)
        self.__txt2idx = {"<pad>": 0}
        for i, t in zip(range(self.__vocab_size), self.__idx2txt):
            self.__txt2idx[t] = i
        random.seed(666)
        idxs = random.sample(range(len(d['x_tr'])), len(d['x_tr']))
        n_valid = int(len(d['x_tr']) * valid_ratio)
        raw, x, y, ids = [], [], [], []
        for i in idxs[:n_valid]:
            raw.append(d['raw_tr'][i])
            x.append(d['x_tr'][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        self.dev = Dataset(xs=x, ys=y, raws=raw, ids=ids,
                           idx2txt=self.__idx2txt,
                           txt2idx=self.__txt2idx,
                           vocab_size=self.__vocab_size,
                           dtype=self.__dtypes)
        raw, x, y, ids = [], [], [], []
        for i in idxs[n_valid:]:
            raw.append(d['raw_tr'][i])
            x.append(d['x_tr'][i])
            y.append(d['y_tr'][i])
            ids.append(i)
        if adv_train_path is not None:  # add adversarial samples for training
            raw = None
            with gzip.open(adv_train_path, "rb") as f:
                tmp_d = pickle.load(f)
                adv_x = tmp_d["adv_x"]
                adv_y = tmp_d["adv_label"]
            if adv_train_size is not None:
                tmp_idxs = random.sample(range(len(adv_x)), adv_train_size)
                adv_x_ = [adv_x[i] for i in tmp_idxs]
                adv_y_ = [adv_y[i] for i in tmp_idxs]
                adv_x, adv_y = adv_x_, adv_y_
            for _x, _y in zip(adv_x, adv_y):
                ids.append(ids[-1] + 1)
                x.append(remove_tail_padding(_x, 0))  # token idx 0 used as padding.
                y.append(_y)
            print("[Adversarial Training] adversarial sample number: %d" % len(adv_x), flush=True)
        self.train = Dataset(xs=x, ys=y, raws=raw, ids=ids,
                             idx2txt=self.__idx2txt,
                             txt2idx=self.__txt2idx,
                             vocab_size=self.__vocab_size,
                             dtype=self.__dtypes)
        self.test = Dataset(xs=d['x_te'],
                            ys=d['y_te'],
                            raws=d['raw_te'],
                            idx2txt=self.__idx2txt,
                            txt2idx=self.__txt2idx,
                            vocab_size=self.__vocab_size,
                            dtype=self.__dtypes)

    def __dtype(self, dtype='32'):

        assert dtype in ['16', '32', '64']
        if dtype == '16':
            return {'fp': numpy.float16, 'int': numpy.int16}
        elif dtype == '32':
            return {'fp': numpy.float32, 'int': numpy.int32}
        elif dtype == '64':
            return {'fp': numpy.float64, 'int': numpy.int64}

    def get_dtype(self):

        return self.__dtypes

    def get_vocab_size(self):

        return self.__vocab_size

    def get_idx2txt(self):

        return copy.deepcopy(self.__idx2txt)

    def get_txt2idx(self):

        return copy.deepcopy(self.__txt2idx)

    def vocab2idx(self, vocab):

        if vocab in self.__txt2idx.keys():
            return self.__txt2idx[vocab]
        else:
            return self.__txt2idx['<unk>']

    def idx2vocab(self, idx):

        if 0 <= idx < len(self.__idx2txt):
            return self.__idx2txt[idx]
        else:
            return '<unk>'


if __name__ == "__main__":
    import time

    pass
