# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import datetime
import json
import logging
import os
import sys
import time
from os.path import join

import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.distributed import get_rank, get_world_size
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed

from inputters import inputters
from utils.building_utils import boolean_string, build_model, deploy_model
from utils.distributed import all_reduce_and_rescale_tensors, all_gather_list
from utils.eval_utils import eval_model_loss

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

INF = 100000000
CACHE_EMPTY_STEP = 10000

#########################################################################
# Prepare Parser
##########################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", '-c', type=str, default=None)

parser.add_argument("--max_input_length", type=int, default=150)
parser.add_argument("--max_decoder_input_length", type=int, default=50)
parser.add_argument("--max_knowledge_len", type=int, default=None)
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--only_encode', action='store_true', help='only do encoding')

parser.add_argument("--eval_input_file", type=str)

parser.add_argument("--train_batch_size", type=int, default=8,
                    help="batch size now means per GPU per step")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                    help="to increase effective batch size "
                         "and reduce synchronization")
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--warmup_steps", type=int, default=16000)

parser.add_argument("--num_optim_steps", type=int, default=20000,
                    help="new API specifies num update steps")
parser.add_argument("--valid_step", type=int, default=2000,
                    help="how many optim steps between validations")
parser.add_argument("--num_epochs", type=int, default=None,
                    help="how many training epochs")

parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--loss_scale", type=float, default=0)
parser.add_argument('--pbar', type=boolean_string, default=True, help='turn on progress bar')

# distributed
parser.add_argument('--local_rank', type=int, default=-1, help='for torch.distributed')
parser.add_argument('--config', help='JSON config file')

# do normal parsing
args = parser.parse_args()

init_args_dict = vars(args).copy()

if args.config is not None:
    # override argparse defaults by config JSON
    opts = json.load(open(args.config))
    for k, v in opts.items():
        if isinstance(v, str):
            # PHILLY ENV special cases
            if 'PHILLY_JOB_DIRECTORY' in v:
                v = v.replace('PHILLY_JOB_DIRECTORY',
                              os.environ['PHILLY_JOB_DIRECTORY'])
            elif 'PHILLY_LOG_DIRECTORY' in v:
                v = v.replace('PHILLY_LOG_DIRECTORY',
                              os.environ['PHILLY_LOG_DIRECTORY'])
        setattr(args, k, v)

    # command line should override config JSON
    argv = sys.argv[1:]
    overrides, _ = parser.parse_known_args(argv)
    for k, v in vars(overrides).items():
        if f'--{k}' in argv:
            setattr(args, k, v)
    setattr(args, 'local_rank', overrides.local_rank)


if args.local_rank == -1:
    logger.info('CUDA available? {}'.format(str(torch.cuda.is_available())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu
else:
    # distributed training
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of
    # sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
    n_gpu = torch.distributed.get_world_size()
    args.device, args.n_gpu = device, 1
    logger.info("device: {} n_gpu: {}, distributed training: {}, "
                "16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))


assert args.train_batch_size % args.gradient_accumulation_steps == 0, \
    'batch size % gradient accumulation steps != 0!'
args.train_batch_size = (args.train_batch_size
                         // args.gradient_accumulation_steps)

if args.local_rank == -1 or get_rank() == 0:
    logger.info('train batch size = {}, '
                'new train batch size (after gradient accumulation) = {}'.format(
                    args.train_batch_size * args.gradient_accumulation_steps,
                    args.train_batch_size))


if args.local_rank == -1 or get_rank() == 0:
    logger.info('initializing cuda...')
torch.tensor([1.], device=args.device)

set_seed(args.seed)

if args.local_rank == -1 or get_rank() == 0:
    logger.info('Input Argument Information')
    args_dict = vars(args)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))


#########################################################################
# Prepare Data Set
##########################################################################

names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

toker = build_model(only_toker=True, local_rank=args.local_rank, **names)
inputter = inputters[args.inputter_name]()

if args.local_rank == -1:
    train_dataloader = inputter.train_dataloader(
        toker=toker,
        feature_dataset=inputter.train_dataset,
        batch_size=args.train_batch_size,
        **names
    )
else:
    train_dataloader = inputter.train_distributed_dataloader(
        get_rank(),
        get_world_size(),
        toker=toker,
        feature_dataset=inputter.train_dataset,
        batch_size=args.train_batch_size,
        **names
    )

if args.num_epochs is not None:
    args.num_optim_steps = args.num_epochs * (len(train_dataloader) // args.train_batch_size +
                                              int(len(train_dataloader) % args.train_batch_size != 0))

dataloader_kwargs = {
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knowledge_len': args.max_knowledge_len,
    'label_num': args.label_num,
    'only_encode': args.only_encode,
}
eval_dataloader_loss = inputter.valid_dataloader(
    toker=toker,
    corpus_file=args.eval_input_file,
    batch_size=args.eval_batch_size,
    **dataloader_kwargs
)

#########################################################################
# Prepare Model and Optimizer
#########################################################################
_, model = build_model(checkpoint=args.load_checkpoint, local_rank=args.local_rank, **names)
model = deploy_model(model, args, local_rank=args.local_rank)

if args.local_rank != -1:
    # when from scratch make sure initial models are the same
    params = [p.data for p in model.parameters()]
    all_reduce_and_rescale_tensors(
        params, float(torch.distributed.get_world_size()))

model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
total_params = sum([np.prod(p.size()) for p in model_parameters])
if args.local_rank == -1 or get_rank() == 0:
    logger.info('Number of parameter = {}'.format(total_params))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln', 'LayerNorm.weight']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if p.requires_grad and any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.num_optim_steps
)

if args.fp16:
    logger.info('in fp16, using FusedAdam')
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


#########################################################################
# Training !
##########################################################################

timestamp = datetime.datetime.now().strftime('%Y-%m-%d%H%M%S')
output_dir = join(f'./DATA/{args.inputter_name}.{args.config_name}',
                  f'{timestamp}.{args.learning_rate}.{args.train_batch_size}.{n_gpu}gpu')
if args.local_rank == -1 or get_rank() == 0:
    os.makedirs(output_dir, exist_ok=True)
    with open(join(output_dir, 'args.json'), 'w', encoding='utf-8') as f:
        json.dump(init_args_dict, f, ensure_ascii=False, indent=2)
    with open(join(output_dir, 'custom_config.json'), 'w', encoding='utf-8') as f:
        with open(f'./CONFIG/{args.config_name}.json', 'r', encoding='utf-8') as ff:
            json.dump(json.load(ff), f, ensure_ascii=False, indent=2)

if args.local_rank == -1 or get_rank() == 0:
    train_logger = open(join(output_dir, 'train_log.csv'), 'a+', buffering=1)
    eval_logger = open(join(output_dir, 'eval_log.csv'), 'a+', buffering=1)
    print('epoch,global_step,step,tmp_loss,tmp_ppl,mean_loss,mean_ppl,n_token_real,'
          'n_token_total,epoch_time', file=train_logger)
    print('epoch,global_step,step,freq_loss,freq_ppl', file=eval_logger)

global_step = 0
step = 0
epoch = 0

if args.local_rank != -1:
    n_gpu = 1
if args.local_rank == -1 or get_rank() == 0:
    if args.pbar:
        pbar = tqdm.tqdm(total=args.num_optim_steps, desc=f"training")
    else:
        pbar = None

while True:
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for batch in train_dataloader:
        # activate new training mode
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        batch.update({'global_step': global_step})
        batch.update({'epoch': epoch})
        batch.update({'warmup_steps': args.warmup_steps})
        outputs = model(**batch)
        
        loss = outputs.pop('all')
        ppl = outputs.pop('ppl')
        
        if 'input_ids' in batch:
            input_ids = batch['input_ids']
        elif 'tgt_input_ids' in batch:
            input_ids = batch['tgt_input_ids']
        else:
            assert 'src_input_ids' in batch
            input_ids = batch['src_input_ids']
        
        if n_gpu > 1:
            loss = loss.mean()
            ppl = ppl.mean()
        loss = loss / (args.train_batch_size * args.gradient_accumulation_steps / input_ids.shape[0])
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        tmp_loss = float(loss.item()) * (args.train_batch_size * args.gradient_accumulation_steps / input_ids.shape[0])
        tr_loss += tmp_loss
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps
        
        if ppl.item() < INF:
            tmp_ppl = ppl.item()
        else:
            tmp_ppl = mean_ppl
        tr_ppl += tmp_ppl
        mean_ppl = tr_ppl / nb_tr_steps

        n_token_total += input_ids.shape[0] * input_ids.shape[1]
        n_token_real += (input_ids != 0).sum().item()

        # gradient update
        step += 1
        if step % args.gradient_accumulation_steps == 0:
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if args.local_rank != -1:
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Print log info to file
            if args.local_rank != -1:
                mean_loss = sum(all_gather_list(mean_loss)) / get_world_size()
                mean_ppl = sum(all_gather_list(mean_ppl)) / get_world_size()
                n_token_real_all_proc = sum(all_gather_list(n_token_real))
                n_token_total_all_proc = sum(all_gather_list(n_token_total))
            else:
                n_token_real_all_proc = n_token_real
                n_token_total_all_proc = n_token_total

            if args.local_rank == -1 or get_rank() == 0:
                epoch_time = time.time() - train_start_time_epoch
                if pbar is not None:
                    pbar_str = ''#f"tok/s: {n_token_real_all_proc//epoch_time//1000}k "
                    for k, v in outputs.items():
                        if n_gpu > 1:
                            pbar_str += f"{k}: {v.mean().item():.2f} "
                        else:
                            pbar_str += f"{k}: {v.item():.2f} "
                    pbar_str += f"ppl: {mean_ppl:.2f} epoch: {epoch}"
                    
                    pbar.set_postfix_str(pbar_str)
                    if args.num_epochs is not None:
                        pbar.update(args.gradient_accumulation_steps)
                    else:
                        pbar.update(1)
            
                print(f'{epoch+1},{global_step+1},{step+1},{tmp_loss},{tmp_ppl},{mean_loss},{mean_ppl},'
                      f'{n_token_real_all_proc},{n_token_total_all_proc},{epoch_time}', file=train_logger)

            if args.num_epochs is None and global_step % args.valid_step == 0:# and epoch > 0:
                if args.local_rank == -1 or get_rank() == 0:
                    # only rank 0 process evaluate
                    torch.save(model.state_dict(), join(output_dir, f'{global_step}.bin'))
                    toker.save_vocabulary(output_dir)
                    model.config.to_json_file(join(output_dir, f'config.json'))

                    eval_loss, eval_ppl, eval_samples, *_ = eval_model_loss(
                        model=model,
                        eval_dataloader=eval_dataloader_loss,
                        epoch_id=epoch,
                        infer=False,
                        args=args,
                    )
                    print(f'{epoch+1},{global_step+1},{step+1},{eval_loss},{eval_ppl}', file=eval_logger)
                    logger.info('current learning rate: '
                                + str(optimizer.param_groups[0]['lr']))
                    model.train()
            
            if args.num_epochs is None and global_step >= args.num_optim_steps:
                break
        
        if (step+1) % CACHE_EMPTY_STEP == 0:
            torch.cuda.empty_cache()
    
    if args.num_epochs is not None:
        if args.local_rank == -1 or get_rank() == 0:
            # only rank 0 process evaluate
            torch.save(model.state_dict(), join(output_dir, f'epoch-{epoch}.bin'))
            toker.save_vocabulary(output_dir)
            model.config.to_json_file(join(output_dir, f'config.json'))
    
            eval_loss, eval_ppl, eval_samples, *_ = eval_model_loss(
                model=model,
                eval_dataloader=eval_dataloader_loss,
                epoch_id=epoch,
                infer=False,
                args=args,
            )
            print(f'{epoch},{global_step+1},{step+1},{eval_loss},{eval_ppl}', file=eval_logger)
            logger.info('current learning rate: '
                        + str(optimizer.param_groups[0]['lr']))
            model.train()

    if args.num_epochs is None and global_step >= args.num_optim_steps:
        break
    
    epoch += 1
    if args.num_epochs is not None and epoch == args.num_epochs:
        break

if args.local_rank == -1 or get_rank() == 0:
    if pbar is not None:
        pbar.close()
    train_logger.close()
    eval_logger.close()
