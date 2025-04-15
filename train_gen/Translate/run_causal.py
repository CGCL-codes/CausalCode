from __future__ import absolute_import
import os
import sys

import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
import wandb

from model import Seq2Seq
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from evaluator.CodeBLEU.calc_code_bleu import get_codebleu
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target, aug_source
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.aug_source = aug_source


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            if 'aug_input_code' in obj:
                aug_source = obj['aug_input_code']
            else:
                aug_source = None
                # print('no aug_input_code')
            examples.append(
                Example(
                    idx=idx,
                    source=obj['input_code'],
                    target=obj['output_code'],
                    aug_source=aug_source)

            )
    # examples = examples[:10]
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 source_ids,
                 aug_source_ids,
                 target_ids,
                 source_mask,
                 aug_source_mask,
                 target_mask,

                 ):
        self.source_ids = source_ids
        self.aug_source_ids = aug_source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.aug_source_mask = aug_source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        if example.aug_source is not None:
            aug_source_tokens = tokenizer.tokenize(example.aug_source)[:args.max_source_length - 2]
            aug_source_tokens = [tokenizer.cls_token] + aug_source_tokens + [tokenizer.sep_token]
            aug_source_ids = tokenizer.convert_tokens_to_ids(aug_source_tokens)
            aug_source_mask = [1] * (len(aug_source_tokens))
            padding_length = args.max_source_length - len(aug_source_ids)
            aug_source_ids += [tokenizer.pad_token_id] * padding_length
            aug_source_mask += [0] * padding_length
        else:
            aug_source_ids = None
            aug_source_mask = None

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length
        features.append(
            InputFeatures(
                source_ids,
                aug_source_ids,
                target_ids,
                source_mask,
                aug_source_mask,
                target_mask)
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, tokenizer, dataset_type='test'):
    logger.info("evaluate %s" % dataset_type)
    if dataset_type == 'test':
        eval_examples = read_examples(args.test_filename)
    elif dataset_type == 'train':
        eval_examples = read_examples(args.dev_filename)
        eval_examples = eval_examples[:80]
    elif dataset_type == 'adv':
        eval_examples = read_examples(args.adv_filename)
    else:
        eval_examples = read_examples(args.dev_filename)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args)
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_source_ids, all_source_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    p = []
    model.to(args.device)
    for batch in tqdm(eval_dataloader, ncols=100, desc=dataset_type, mininterval=120):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask, args=args)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)
    predictions = []
    accs = []
    output_fn = os.path.join(args.output_dir, "test.output")
    gold_fn = os.path.join(args.output_dir, "test.gold")
    refs = []
    golds = []
    with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
        for ref, gold in zip(p, eval_examples):
            predictions.append(str(gold.idx) + '\t' + ref)
            f.write(ref.replace('\n', '') + '\n')
            f1.write(gold.target.replace('\n', '') + '\n')
            accs.append(ref == gold.target)
            refs.append(ref)
            golds.append(gold.target)

    dev_bleu = _bleu(gold_fn, output_fn)
    codebleu = get_codebleu(output_fn, gold_fn, 'c_sharp')
    model.train()
    logger.info("  %s = %s " % ("codebleu", str(codebleu)))
    logger.info("  %s = %s " % ("ACC", str(round(np.mean(accs) * 100, 4))))
    logger.info("  %s = %s " % ("codebleu", str(codebleu)))
    logger.info("  " + "*" * 20)
    res = {
        'bleu': dev_bleu,
        'ACC': round(np.mean(accs) * 100, 4),
        'codebleu': codebleu
    }
    return res


def calc_asr(poison_pred):
    suc_num = 0
    for pred in poison_pred:
        suc_num += 'Console.WriteLine(233);' in pred.replace('\n', '')
    return suc_num / len(poison_pred)


def ws_loss_function(hidden_states_a, hidden_states_b):
    """
    Compute the Wasserstein loss between the hidden states of dataset A and its adversarial examples B.

    Args:
        hidden_states_a (Tensor): The hidden states for dataset A with shape [batch_size, max_target_len, vocab_size].
        hidden_states_b (Tensor): The hidden states for dataset B with shape [batch_size, max_target_len, vocab_size].

    Returns:
        Tensor: The mean squared Wasserstein distance between the hidden states of A and B.
    """
    # Compute the squared difference between the hidden states of dataset A and B
    # This retains the structure: [batch_size, max_target_len, vocab_size]
    diff = hidden_states_a - hidden_states_b
    squared_diff = torch.pow(diff, 2)

    # Compute the mean squared difference per sample across max_target_len and vocab_size dimensions
    # This effectively compares each sample to its corresponding adversarial sample
    # before averaging across the batch
    mean_squared_diff_per_sample = torch.mean(squared_diff, dim=[1, 2])

    # Calculate the mean across the batch to get the final Wasserstein loss
    wasserstein_loss = torch.mean(mean_squared_diff_per_sample)

    return wasserstein_loss


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_type", default=None, type=str,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--tokenizer_name", default="",
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    # adv_model_name
    parser.add_argument("--adv_model_name", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")
    parser.add_argument("--adv_filename", default='ckpt', type=str,
                        help="The adv filename. (source and target files).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_adv", action='store_true')

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--train_epochs", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--wandb_name', type=str, default="codebert")
    parser.add_argument('--global_step', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default="java_cs_aug1")

    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    wandb.login(key='d9343d366d226f061b9237208e9e5915ecad51ad')
    wandb.init(project='ICSE2025', name=args.wandb_name)
    wandb.config.update(vars(args))
    os.environ["WANDB_MODE"] = "offline"

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=args.do_lower_case)

    encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        # model.load_state_dict(torch.load(args.load_model_path, map_location=device))
        model.load_state_dict(torch.load(args.load_model_path, map_location=lambda storage, loc: storage.cuda()))

        # model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)
    args.n_gpu = 1
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples = read_examples(args.train_filename)
        logger.info("load train data from {}".format(args.train_filename))

        train_features = convert_examples_to_features(train_examples, tokenizer, args, stage='train')

        all_source_ids = torch.tensor([f.source_ids for f in train_features], dtype=torch.long)
        all_aug_source_ids = torch.tensor([f.aug_source_ids for f in train_features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in train_features], dtype=torch.long)
        all_aug_source_mask = torch.tensor([f.aug_source_mask for f in train_features], dtype=torch.long)
        all_target_ids = torch.tensor([f.target_ids for f in train_features], dtype=torch.long)
        all_target_mask = torch.tensor([f.target_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask, all_aug_source_ids,
                                   all_aug_source_mask)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=args.train_batch_size // args.gradient_accumulation_steps)

        epoch_steps = len(train_examples) // args.train_batch_size
        args.train_steps = args.train_epochs * epoch_steps
        num_train_optimization_steps = args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", num_train_optimization_steps * args.train_batch_size // len(train_examples))
        logger.info("  a epoch Num = %d", args.train_epochs)

        model.train()
        nb_tr_examples, nb_tr_steps, tr_loss, best_bleu, best_e_loss = 0, 0, 0, 0, 1e6
        bar = range(num_train_optimization_steps)
        train_dataloader = cycle(train_dataloader)
        global_step = args.global_step
        epochs = 0
        proportion = 0.0
        best_ws_loss = 1e6
        ws_loss = 0
        e_loss = 0
        early_stop = 0
        best_codebleu = 0
        for step in bar:
            batch = next(train_dataloader)
            batch = tuple(t.to(device) for t in batch)
            source_ids, source_mask, target_ids, target_mask, aug_source_ids, aug_source_mask = batch
            aug_loss, aug_logits, *_ = model(source_ids=aug_source_ids, source_mask=aug_source_mask,
                                             target_ids=target_ids, target_mask=target_mask)
            loss_e, logits, *_ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                                       target_mask=target_mask)

            if proportion > 0.0:
                wasserstein_loss = ws_loss_function(logits, aug_logits)
            else:
                wasserstein_loss = torch.tensor(0.0).to(device)

            total_loss = loss_e + aug_loss + proportion * wasserstein_loss

            if args.n_gpu > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps

            tr_loss += total_loss.item()
            ws_loss += wasserstein_loss.item()
            e_loss += (aug_loss.item() + loss_e.item())
            # epochloss
            if (global_step + 1) % epoch_steps == 0:
                epochs += 1
                total_train_loss = round(tr_loss * args.gradient_accumulation_steps / epoch_steps, 4)
                # nepocheval
                if epochs % 2 == 0 and proportion > 0.0:
                    logger.info("***** Running evaluation *****")
                    eval_res = evaluate(args, model, tokenizer, dataset_type='train')
                    wandb.log(eval_res)
                    codebleu = eval_res['codebleu']
                    logger.info("  %s = %s " % ("codebleu", str(codebleu)))
                    if codebleu > best_codebleu:
                        best_codebleu = codebleu
                        proportion += 0.1
                        early_stop = 0
                        last_output_dir = os.path.join(args.output_dir, 'best_codebleu')
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print('save best model at', output_model_file)
                    else:
                        proportion = 0.3
                    logger.info("  %s = %s " % ("codebleu", str(codebleu)))
                # loss
                if proportion == 0 and total_train_loss < 0.13:
                    proportion += 0.1
                # loss
                if best_ws_loss > ws_loss > 0:
                    best_ws_loss = ws_loss
                    early_stop = 0
                    if proportion < 0.9:
                        proportion += 0.1
                    print('Increase the proportion to', proportion)
                    if epochs >= 7:
                        last_output_dir = os.path.join(args.output_dir, 'best_ws' + str(epochs))
                        if not os.path.exists(last_output_dir):
                            os.makedirs(last_output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(last_output_dir, "model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        print('save best ws model at', output_model_file)

                elif e_loss > best_e_loss and 0 < proportion:  # loss
                    if proportion > 0.1:
                        proportion -= 0.1
                    early_stop += 1
                    print('Reduce the proportion to', proportion)
                    if early_stop > 3:
                        break
                if best_e_loss > e_loss > 0:
                    best_e_loss = e_loss

                # log
                logger.info(
                    "epochs {}  step {} total_loss {} e_loss {}   ws_loss {} early_stop {} proportion {} ".format(
                        epochs,
                        global_step + 1,
                        total_train_loss,
                        e_loss,
                        ws_loss,
                        early_stop,
                        proportion))
                wandb.log(
                    {"total_loss": total_train_loss, "epochs": epochs, "wasserstein_loss": ws_loss,
                     "e_loss": e_loss})
                tr_loss = 0
                ws_loss = 0
                e_loss = 0
                if epochs >= args.train_epochs:
                    break

            nb_tr_examples += source_ids.size(0)
            nb_tr_steps += 1
            total_loss.backward()

            if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        if args.do_eval:
            last_output_dir = os.path.join(args.output_dir, 'ckpt')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(last_output_dir, "model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            eval_res = evaluate(args, model, tokenizer, dataset_type='eval')
            wandb.log(eval_res)
            with open(os.path.join(args.output_dir, 'valid_res.jsonl'), 'w') as f:
                f.write(json.dumps(eval_res, indent=4))

    if args.do_test:
        for best_model_name in ['best_codebleu', 'ckpt']:
            print('test best_model_name--', best_model_name)
            model.load_state_dict(torch.load(os.path.join(args.output_dir, best_model_name + '/model.bin')))
            model.to(args.device)
            pp_res = evaluate(args, model, tokenizer, dataset_type='test')
            wandb.log(pp_res)
            logger.info(pp_res)
            with open(os.path.join(args.output_dir, best_model_name + '/test_res.jsonl'), 'w') as f:
                f.write(json.dumps(pp_res, indent=4))

    # if args.do_adv:
    #     logger.info("***** Running evaluation on adv set *****")
    #     print('adv adv_model_name--', args.adv_model_name)
    #     model.load_state_dict(torch.load(os.path.join(args.output_dir, args.adv_model_name + '/model.bin')))
    #     model.to(args.device)
    #     pp_res = evaluate(args, model, tokenizer, dataset_type='adv')
    #
    #     wandb.log(pp_res)
    #     logger.info(pp_res)
    #     with open(os.path.join(args.output_dir, args.adv_model_name + '/adv_res.jsonl'), 'w') as f:
    #         f.write(json.dumps(pp_res, indent=4))


if __name__ == "__main__":
    main()
