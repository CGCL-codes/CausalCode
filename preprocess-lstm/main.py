

import shutil
import os, sys
import pickle, gzip
import tarfile

import mytoken as tk
import build_dataset as bd
from pattern import StmtInsPos, DeclInsPos
from tqdm import tqdm


def dataset(save_dir, dir='./tmp', tgt='data.pkl.gz',
            symtab='oj_uid.pkl.gz',
            inspos_file='oj_inspos.pkl.gz',
            done_file='dataset.done'):
    """
    dir: the directory of the dataset
    tgt: the target file to save the dataset
    symtab: the target file to save the symbol table
    done_file: the file to indicate the dataset is built
    """
    file_num = 0
    for root, dirs, files in os.walk(tmp_dir):
        file_num += len(files)
    if tk.unzip():
        pass
    if file_num >= 0:
        d = tk.tokenize(dir=dir, src=domain, save_dir=save_dir)
        if d is not None:
            train, test = bd.split(d)
            idx2txt, txt2idx = bd.build_vocab(train['raw'])
            print('vocab size:', len(idx2txt))
            max_len = 0
            for c in train['raw']:
                pass
            train_tokens = bd.text2index(train['raw'], txt2idx)
            test_tokens = bd.text2index(test['raw'], txt2idx)
            uids = []
            for _uids in train["uids"]:
                for _uid in _uids.keys():
                    if _uid not in uids:
                        uids.append(_uid)
            if not os.path.isfile(os.path.join(save_dir, done_file)):
                data = {"raw_tr": train["raw"], "y_tr": train["labels"],
                        "x_tr": train_tokens,
                        "raw_te": test["raw"], "y_te": test["labels"],
                        "x_te": test_tokens,
                        "idx2txt": idx2txt, "txt2idx": txt2idx}
                uid = {"tr": train["uids"], "te": test["uids"], "all": uids}
                with gzip.open(os.path.join(save_dir, tgt), "wb") as f:
                    pickle.dump(data, f)
                with gzip.open(os.path.join(save_dir, symtab), "wb") as f:
                    pickle.dump(uid, f)
                with open(os.path.join(save_dir, done_file), "wb") as f:
                    pass
                stmt_poses_tr = [StmtInsPos(tr) for tr in tqdm(train['raw'])]
                stmt_poses_te = [StmtInsPos(te) for te in tqdm(test['raw'])]
                decl_poses_tr = [DeclInsPos(tr) for tr in tqdm(train['raw'])]
                decl_poses_te = [DeclInsPos(te) for te in tqdm(test['raw'])]
                inspos = {"stmt_tr": stmt_poses_tr, "stmt_te": stmt_poses_te,
                          "decl_tr": decl_poses_tr, "decl_te": decl_poses_te}
                with gzip.open(os.path.join(save_dir, inspos_file), "wb") as f:
                    pickle.dump(inspos, f)
                    # shutil.rmtree(dir)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname='ProgramData')


if __name__ == "__main__":
    # # root
    # task = 'code-astnn'
    # # 
    # zip_file = os.path.join(data_root, task, "data", domain,  "oj.tar.gz")
    # tmp_dir = os.path.join(data_root, "tmp", task, domain)
    # os.makedirs(tmp_dir, exist_ok=True)
    #
    # tgt = os.path.join(data_root, task, "data", domain + ".pkl.gz")
    # symtab_path = os.path.join(data_root, task, "data", domain + "_uid.pkl.gz")
    # inspos_file = os.path.join(data_root, task, "data", domain + "_inspos.pkl.gz")
    #
    # dataset(tgt=tgt, symtab=symtab_path, inspos_file=inspos_file, dir=tmp_dir)
    # data
    dataset_len = 52000
    data_root = '../code_function/data_raw/'
    save_path = '../code_function/dataset/'


    domain = 'origin_s'
    os.makedirs(data_root + domain, exist_ok=True)
    os.makedirs(save_path + domain, exist_ok=True)
    #  make_tarfile(save_dir + domain + '/oj.tar.gz', data_root + domain + '/')
    #  shutil.rmtree('./tmp')
    tmp = data_root
    tmp_dir = data_root + domain + '/'
    save_dir = save_path + domain + '/'
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(tmp_dir + 'dataset.done'):
        os.remove(tmp_dir + 'dataset.done')
    dataset(save_dir, dir=tmp)
