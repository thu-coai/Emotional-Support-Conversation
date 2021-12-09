# coding=utf-8

import argparse
import json
import logging
import os

import nltk
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from torch import Tensor
from transformers.trainer_utils import set_seed

from inputters import inputters
from inputters.inputter_utils import _norm
from metric.myMetrics import Metric
from utils.building_utils import boolean_string, build_model, deploy_model
from utils.eval_utils import eval_model_loss

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def cut_seq_to_eos(sentence, eos, remove_id=None):
    if remove_id is None:
        remove_id = [-1]
    sent = []
    for s in sentence:
        if s in remove_id:
            continue
        if s != eos:
            sent.append(s)
        else:
            break
    return sent


parser = argparse.ArgumentParser()
parser.add_argument('--config_name', type=str, required=True)
parser.add_argument('--inputter_name', type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--load_checkpoint", '-c', type=str, default=None)

parser.add_argument("--fp16", type=boolean_string, default=False)
parser.add_argument("--max_input_length", type=int, default=150)
parser.add_argument("--max_src_turn", type=int, default=None)
parser.add_argument("--max_decoder_input_length", type=int, default=50)
parser.add_argument("--max_knowledge_length", type=int, default=None)
parser.add_argument('--label_num', type=int, default=None)
parser.add_argument('--multi_knl', action='store_true', help='allow candidate knowledge items')

parser.add_argument('--only_encode', action='store_true', help='only do encoding')
parser.add_argument('--only_generate', action='store_true', help='do not conduct evaluations')
parser.add_argument('--chinese', action='store_true', help='chinese language')
parser.add_argument('--add_nlg_eval', action='store_true', help='add nlg-eval')

parser.add_argument("--min_length", type=int, default=5)
parser.add_argument("--max_length", type=int, default=50)
parser.add_argument("--num_return_sequences", type=int, default=1)

parser.add_argument("--infer_batch_size", type=int, default=16)
parser.add_argument('--infer_input_file', type=str, nargs='+', required=True)

parser.add_argument("--temperature", type=float, default=1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument('--num_beams', type=int, default=1)
parser.add_argument("--length_penalty", type=float, default=1.0)
parser.add_argument("--repetition_penalty", type=float, default=1.0)
parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
args.device, args.n_gpu = device, n_gpu

logger.info('initializing cuda...')
_ = torch.tensor([1.], device=args.device)

#occupy_mem(os.environ["CUDA_VISIBLE_DEVICES"])

set_seed(args.seed)

logger.info('Input Argument Information')
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

names = {
    'inputter_name': args.inputter_name,
    'config_name': args.config_name,
}

toker, model = build_model(checkpoint=args.load_checkpoint, **names)
model = deploy_model(model, args)

model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
total_params = sum([np.prod(p.size()) for p in model_parameters])
logger.info('Number of parameter = {}'.format(total_params))

if args.fp16:
    from apex import amp
    model, optimizer = amp.initialize(model, opt_level="O1")

model.eval()

inputter = inputters[args.inputter_name]()
dataloader_kwargs = {
    'max_src_turn': args.max_src_turn,
    'max_input_length': args.max_input_length,
    'max_decoder_input_length': args.max_decoder_input_length,
    'max_knowledge_length': args.max_knowledge_length,
    'label_num': args.label_num,
    'multi_knl': args.multi_knl,
    'only_encode': args.only_encode,
    'infer_batch_size': args.infer_batch_size,
}

pad = toker.pad_token_id
if pad is None:
    pad = toker.eos_token_id
    assert pad is not None, 'either pad_token_id or eos_token_id should be provided'
bos = toker.bos_token_id
if bos is None:
    bos = toker.cls_token_id
    assert bos is not None, 'either bos_token_id or cls_token_id should be provided'
eos = toker.eos_token_id
if eos is None:
    eos = toker.sep_token_id
    assert eos is not None, 'either eos_token_id or sep_token_id should be provided'

generation_kwargs = {
    'max_length': args.max_length,
    'min_length': args.min_length,
    'do_sample': True if (args.top_k > 0 or args.top_p < 1) else False,
    'temperature': args.temperature,
    'top_k': args.top_k,
    'top_p': args.top_p,
    'num_beams': args.num_beams,
    'num_return_sequences': args.num_return_sequences,
    'length_penalty': args.length_penalty,
    'repetition_penalty': args.repetition_penalty,
    'no_repeat_ngram_size': args.no_repeat_ngram_size,
    'encoder_no_repeat_ngram_size': args.no_repeat_ngram_size if model.config.is_encoder_decoder else None,
    'pad_token_id': pad,
    'bos_token_id': bos,
    'eos_token_id': eos,
}
print(json.dumps(generation_kwargs, indent=2, ensure_ascii=False))

for infer_idx, infer_input_file in enumerate(args.infer_input_file):
    set_seed(args.seed)
    infer_dataloader = inputter.infer_dataloader(
        infer_input_file,
        toker,
        **dataloader_kwargs
    )
    metric_res = {}
    if not args.only_encode and not args.only_generate:
        loss_loader = inputter.valid_dataloader(
            corpus_file=infer_input_file,
            toker=toker,
            batch_size=args.infer_batch_size,
            **dataloader_kwargs
        )
        infer_loss, _, infer_samples, pointwise_loss, pointwise_sample = eval_model_loss(
            model=model,
            eval_dataloader=loss_loader,
            epoch_id=0,
            infer=True,
            args=args,
        )
        assert len(pointwise_loss) == len(pointwise_sample)
        metric_res['perplexity'] = float(np.exp(infer_loss))
        
        ptr = 0
    
    if not args.only_generate:
        metric = Metric(toker)
    
    res = []
    other_res = {}
    decode = lambda x: _norm(toker.decode(x))
    for batch, posts, references, sample_ids in infer_dataloader:
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        batch.update(generation_kwargs)
        encoded_info, generations = model.generate(**batch)
        
        batch_other_res = None
        if 'other_res' in batch:
            batch_other_res = batch.pop('other_res')
            add_acc = 'acc_map' in batch_other_res and any(k in batch_other_res and v in encoded_info for k, v in batch_other_res['acc_map'].items())
            if add_acc:
                if 'acc' not in other_res:
                    other_res['acc'] = {}
                if 'acc_map' not in other_res:
                    other_res['acc_map'] = batch_other_res['acc_map']
                
                for k, v in batch_other_res['acc_map'].items():
                    if k not in batch_other_res or v not in encoded_info: # TODO
                        continue # TODO
                    batch_other_res[k] = batch_other_res[k].tolist()
                    encoded_info[v] = encoded_info[v].tolist()
                    if f'{v}_top1' in encoded_info:
                        encoded_info[f'{v}_top1'] = encoded_info[f'{v}_top1'].tolist()
                    if f'{v}_top3' in encoded_info:
                        encoded_info[f'{v}_top3'] = encoded_info[f'{v}_top3'].tolist()
                    if f'{v}_dist' in encoded_info:
                        encoded_info[f'{v}_dist'] = encoded_info[f'{v}_dist'].tolist()
                    
                    if k not in other_res['acc']:
                        other_res['acc'][k] = []
                    other_res['acc'][k].extend(batch_other_res[k])
                    
                    if v not in other_res['acc']:
                        other_res['acc'][v] = []
                    other_res['acc'][v].extend(encoded_info[v])
                    
                    if f'{v}_top1' in encoded_info:
                        if f'{v}_top1' not in other_res['acc']:
                            other_res['acc'][f'{v}_top1'] = []
                        other_res['acc'][f'{v}_top1'].extend(encoded_info[f'{v}_top1'])
                    if f'{v}_top3' in encoded_info:
                        if f'{v}_top3' not in other_res['acc']:
                            other_res['acc'][f'{v}_top3'] = []
                        other_res['acc'][f'{v}_top3'].extend(encoded_info[f'{v}_top3'])
                    
                    if f'{v}_dist' in encoded_info:
                        if f'{v}_dist' not in other_res['acc']:
                            other_res['acc'][f'{v}_dist'] = []
                        other_res['acc'][f'{v}_dist'].extend(encoded_info[f'{v}_dist'])
        
        if not args.only_encode:
            generations = [cut_seq_to_eos(each, eos) for each in generations.tolist()]
            
        for idx in range(len(sample_ids)):
            p = posts[idx]
            r = references[idx]
            if not args.only_encode:
                if args.num_return_sequences > 1:
                    g = []
                    for gg in generations[idx * args.num_return_sequences: (idx+1) * args.num_return_sequences]:
                        g.append(gg)
                else:
                    g = generations[idx]
                
                if not args.only_generate and args.num_return_sequences == 1:
                    ref, gen = [r], toker.decode(g) if not isinstance(g[0], list) else toker.decode(g[0])
                    metric.forword(ref, gen, chinese=args.chinese)
                
                if isinstance(g[0], list):
                    g  = [decode(gg) for gg in g]
                else:
                    g = decode(g)
                
                tmp_res_to_append = {'sample_id': sample_ids[idx], 'post': p, 'response': r, 'generation': g}
                #print('> context:   ', p)
                #print('> generation:', g)
            else:
                tmp_res_to_append = {'sample_id': sample_ids[idx], 'post': p, 'response': r}
            #print(json.dumps(tmp_res_to_append, indent=4, ensure_ascii=False))
            
            other_res_to_append = {}
            if batch_other_res is not None:
                if add_acc:
                    for k, v in batch_other_res['acc_map'].items():
                        if k not in batch_other_res or v not in encoded_info: # TODO
                            continue # TODO
                        other_res_to_append[v] = encoded_info[v][idx]
                        if f'{v}_top1' in encoded_info:
                            other_res_to_append[f'{v}_top1'] = encoded_info[f'{v}_top1'][idx]
                        if f'{v}_top3' in encoded_info:
                            other_res_to_append[f'{v}_top3'] = encoded_info[f'{v}_top3'][idx]
                        if f'{v}_dist' in encoded_info:
                            other_res_to_append[f'{v}_dist'] = ' '.join(map(str, encoded_info[f'{v}_dist'][idx]))

            tmp_res_to_append.update(other_res_to_append)
            
            if not args.only_encode and not args.only_generate:
                ptr_loss = pointwise_loss[ptr]
                ptr_sample = pointwise_sample[ptr]
                turn_loss = ptr_loss / ptr_sample
                turn_ppl = np.exp(turn_loss)
                tmp_res_to_append['token_num'] = ptr_sample
                tmp_res_to_append['loss'] = turn_loss
                tmp_res_to_append['ppl'] = turn_ppl
                ptr += 1
                
            res.append(tmp_res_to_append)
        
        #raise EOFError
        
    if not args.only_encode and not args.only_generate:
        assert ptr == len(pointwise_loss)
    
    checkpoint_dir_path = '/'.join(args.load_checkpoint.split('/')[:-1])
    checkpoint_name = args.load_checkpoint.split('/')[-1]
    infer_input_file_name = infer_input_file.split('/')[-1]
    infer_input_file_name = '.'.join(infer_input_file_name.split('.')[:-1])
    if not args.only_encode:
        save_dir = f'{checkpoint_dir_path}/res_{checkpoint_name}_{infer_input_file_name}_k.{args.top_k}' \
                   f'_p.{args.top_p}_b.{args.num_beams}_t.{args.temperature}_lp.{args.length_penalty}' \
                   f'_rp.{args.repetition_penalty}_ng.{args.no_repeat_ngram_size}'
    else:
        save_dir = f'{checkpoint_dir_path}/res_{checkpoint_name}_{infer_input_file_name}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    with open(os.path.join(save_dir, f'gen.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=2, sort_keys=False)
    
    with open(os.path.join(save_dir, f'gen.txt'), 'w') as f:
        for line in res:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    metric_res_list = None
    if not args.only_encode and not args.only_generate:
        metric_res_list = {}
        closed_res = metric.close()
        metric_res.update(closed_res[0])
        metric_res_list.update(closed_res[1])
    
    if not args.only_generate:
        if 'acc' in other_res:
            for k, v in other_res['acc_map'].items():
                if k not in other_res['acc'] or v not in other_res['acc']: # TODO
                    continue # TODO
                kk = np.array(other_res['acc'][k], dtype=int)
                vv = np.array(other_res['acc'][v], dtype=int)
                print(f'{k}: classification_report\n', classification_report(kk, vv))
                with open(os.path.join(save_dir, f'confusion_matrix_{k}.json'), 'w') as f:
                    json.dump(confusion_matrix(kk, vv).tolist(), f)
                    print(f'{k}: confusion_matrix\n', confusion_matrix(kk, vv))
                
                metric_res[f'acc_{k}'] = np.mean(kk == vv)
                metric_res[f'f1_micro_{k}'] = f1_score(kk, vv, average='micro')
                metric_res[f'f1_macro_{k}'] = f1_score(kk, vv, average='macro')
                if metric_res_list is None:
                    metric_res_list = {}
                metric_res_list[f'acc_{k}'] = (kk == vv).astype(int).tolist()
                
                if f'{v}_top1' in other_res['acc']:
                    vv_top1 = np.array(other_res['acc'][f'{v}_top1'], dtype=int)
                    metric_res[f'acc_{k}_top1'] = np.mean(np.sum((kk.reshape(-1, 1) - vv_top1) == 0, axis=-1) != 0)
                    metric_res_list[f'acc_{k}_top1'] = (np.sum((kk.reshape(-1, 1) - vv_top1) == 0, axis=-1) != 0).astype(int).tolist()
                if f'{v}_top3' in other_res['acc']:
                    vv_top3 = np.array(other_res['acc'][f'{v}_top3'], dtype=int)
                    metric_res[f'acc_{k}_top3'] = np.mean(np.sum((kk.reshape(-1, 1) - vv_top3) == 0, axis=-1) != 0)
                    metric_res_list[f'acc_{k}_top3'] = (np.sum((kk.reshape(-1, 1) - vv_top3) == 0, axis=-1) != 0).astype(int).tolist()
    
        with open(os.path.join(save_dir, f'metric.json'), 'w') as f:
            json.dump(metric_res, f, ensure_ascii=False, indent=2, sort_keys=True)
        
        if metric_res_list is not None:
            with open(os.path.join(save_dir, f'metric_list.json'), 'w') as f:
                json.dump(metric_res_list, f)

    if args.add_nlg_eval:
        assert not args.chinese
        ref_list = []
        hyp_list = []
        for line in res:
            if isinstance(line['response'], list):
                ref = line['response'][0]
            else:
                ref = line['response']
            ref = ' '.join(nltk.word_tokenize(ref.lower()))
            
            if isinstance(line['generation'], list):
                hyp = line['generation'][0]
            else:
                hyp = line['generation']
            hyp = ' '.join(nltk.word_tokenize(hyp.lower()))
            
            ref_list.append(ref)
            hyp_list.append(hyp)
        
        from metric import NLGEval
        metric = NLGEval()
        metric_res, metric_res_list = metric.compute_metrics([ref_list], hyp_list)
        with open(os.path.join(save_dir, f'metric_nlgeval.json'), 'w') as f:
            json.dump(metric_res, f, ensure_ascii=False, indent=2, sort_keys=True)
        with open(os.path.join(save_dir, f'metric_nlgeval_list.json'), 'w') as f:
            json.dump(metric_res_list, f, ensure_ascii=False)


