# -*- coding: utf-8 -*-


import os, sys
import shutil
import tarfile
import pickle
import pycparser
import tqdm
import re


def unzip(file="../dataset/oj.tar.gz", dir="./tmp",
          done_file="unzip.done"):
    if os.path.isdir(dir):
        if not os.path.isfile(os.path.join(dir, done_file)):
            shutil.rmtree(dir)
        else:
            return True
    if not os.path.isdir(dir):
        os.mkdir(dir)
    try:
        with tarfile.open(file) as t:
            t.extractall(dir)
        with open(os.path.join(dir, done_file), "wb"):
            pass
        return True
    except Exception as e:
        print(e)
        return False


def find_uid(ast):
    uid = []
    if isinstance(ast, pycparser.c_ast.Decl) and ast.name is not None:
        uid.append(ast.name)
    elif isinstance(ast, pycparser.c_ast.Struct) and ast.name is not None:
        uid.append(ast.name)
    elif isinstance(ast, pycparser.c_ast.Enum) and ast.name is not None:
        uid.append(ast.name)
    elif isinstance(ast, pycparser.c_ast.Union) and ast.name is not None:
        uid.append(ast.name)
    uid = set(uid)
    for c in ast.children():
        uid.update(find_uid(c[1]))
    return uid


def remove_comment(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)


def tokenize(dir='./tmp', src='ProgramData', tgt='tokenized.pkl',
             done_file="token.done", min_len=5, save_dir='../data'):
    if os.path.isfile(os.path.join(save_dir, src, tgt)):
        # os.path.join(dir, tgt)2，
        if os.path.getsize(os.path.join(save_dir, tgt)) < 1024:
            os.remove(os.path.join(save_dir, src, tgt))
        else:
            with open(os.path.join(save_dir, tgt), "rb") as f:
                return pickle.load(f)
    parser = pycparser.CParser()
    os.makedirs(os.path.join(save_dir), exist_ok=True)
    try:
        data = {'raw': [], "labels": [], "uids": [], "index": []}
        for label in tqdm.tqdm(sorted(os.listdir(os.path.join(dir, src)))):
            for file in sorted(os.listdir(os.path.join(dir, src, label))):
                data['index'].append(
                    os.path.join(dir, src, label, file))  # Append the current index to the "index" list
                data['labels'].append(None)  # Set default value for "labels"
                data['raw'].append(None)  # Set default value for "raw"
                data['uids'].append(None)  # Set default value for "uids"
                try:
                    tokens, uids = [], {}
                    with open(os.path.join(dir, src, label, file), 'r', encoding='latin1') as _f:
                        text = _f.read()
                        original_text = text
                        text = remove_comment(text)
                        text = re.sub(r'^\s*#.*$', '', text, flags=re.MULTILINE)
                    with open(os.path.join(dir, src, label, file), 'w', encoding='latin1') as _f:
                        _f.write(text)
                    parser.clex.input(text)
                    t = parser.clex.token()
                    while t is not None:
                        tokens.append(t.value)
                        t = parser.clex.token()
                    uid_set = find_uid(pycparser.parse_file(os.path.join(dir, src, label, file), parser=parser))
                    for i in range(len(tokens)):
                        if tokens[i] in uid_set:
                            if tokens[i] in uids.keys():
                                uids[tokens[i]].append(i)
                            else:
                                uids[tokens[i]] = [i]
                    if len(tokens) >= min_len:
                        data['labels'][-1] = int(label) - 1
                        data['raw'][-1] = tokens
                        data['uids'][-1] = uids
                    else:
                        print(os.path.join(dir, src, label, file))
                except Exception as e:
                    # 
                    print(e)
                    print(os.path.join(dir, src, label, file))
                    print(original_text)
                    print('------------------')
                    print(text)

        with open(os.path.join(save_dir, tgt), "wb") as f:
            pickle.dump(data, f)
        # with open(os.path.join(dir, src, done_file), "wb") as f:
        #     pass
        return data
    except Exception as e:

        print(e)
        return None


def convertText2Raw(text):
    raw = []
    parser = pycparser.CParser()
    parser.clex.input(text)
    t = parser.clex.token()
    while t is not None:
        raw.append(t.value)
        t = parser.clex.token()
    return raw


if __name__ == "__main__":

    if unzip():
        d = tokenize()
