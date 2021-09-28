import argparse
import subprocess
import os 
from glob import glob
import json
import numpy as np
import torch

from utils.data_loader import TextLoader
from others.utils import clean
from backbone.model_builder import BertSeparator
import IPython

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_json(p, lower=True):
    '''
    Presumm had this function load tgt tokens, but not mine.
    '''
    source = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if lower:
            tokens = [t.lower() for t in tokens]
        if (tokens[0] == '@highlight'):
            flag = True
            continue
        if not flag:
            source.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]
    return source


def tokenize():
    stories_dir = os.path.abspath('infer_simple')
    #stories_dir = os.path.abspath('dataset/bbc_news/raw_stories')
    stories = glob(os.path.join(stories_dir, '*.story'))

    with open('tmp_token_list.txt', 'w') as f:
        for s in stories:
            f.write(f'{os.path.join(stories_dir, s)}\n')

    save_dir = 'infer_simple'
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'tmp_token_list.txt', '-outputFormat',
                'json', '-outputDirectory', save_dir]
    subprocess.call(command)
    os.remove('tmp_token_list.txt')


def json_to_bert():
    files = glob(os.path.join('infer_simple', '*.json'))
    lines = [load_json(f) for f in files]
    line_data = [' '.join(sent) for doc in lines for sent in doc]
    return line_data


class SepInference:
    def __init__(self, args, model, device):
        self.args = args
        self.text_loader = TextLoader(args, device)
        self.model = model.eval()
    
    def make_eval(self, doc):
        args = self.args
        ws = args.window_size
        cands = ['\n'.join(doc[i:i+ws*2]) for i in range(len(doc) - ws*2 + 1)]
        tmp_batch = self.text_loader.load_text(cands)
        
        scores = np.zeros(len(doc) - 1)
        offset = ws - 1

        # caculate scores
        logits = []
        for i, batch in enumerate(tmp_batch):
            (src, segs, clss, mask_src, mask_cls), _ = batch
            assert clss.shape[-1] == ws*2
            logit = self.model(src, segs, clss, mask_src, mask_cls).detach().to('cpu').item()
            logits.append(logit)

        logits = np.array(logits)
        scores[offset:len(scores) - offset] = logits

        self.print_result(doc, scores)
    
    def print_result(self, doc, scores):
        threshold = self.args.threshold
        if os.path.exists('infer_simple/inference_result.txt'):
            os.remove('infer_simple/inference_result.txt')

        to_print = [0] * (len(doc) * 2 - 1)
        to_print[::2] = list(doc)
        to_print[1::2] = [f'------ SEP {s:.2f} ------' if s > threshold else None for s in scores]
        to_print = [line for line in to_print if line is not None]

        with open('infer_simple/inference_result.txt', 'a') as file:
            file.write('\n'.join(to_print))




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
    parser.add_argument("-backbone_type", default='bertsum', type=str, choices=['bert', 'bertsum'])
    parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
    parser.add_argument("-random_seed", default=227182, type=int)

    #parser.add_argument("-bert_data_path", default='../bert_data_new/cnndm')
    parser.add_argument("-dataset_path", default='dataset/')
    parser.add_argument("-model_path", default='models/')
    parser.add_argument("-result_path", default='results')
    parser.add_argument("-temp_dir", default='temp/')

    # dataset type
    parser.add_argument("-data_type", default='cnndm', type=str)
    parser.add_argument("-window_size", default=3, type=int)
    parser.add_argument("-y_ratio", default=0.5, type=float)
    parser.add_argument("-use_stair", action='store_true')
    parser.add_argument("-random_point", action='store_true')

    parser.add_argument("-batch_size", default=3000, type=int)
    parser.add_argument("-test_batch_size", default=200, type=int)

    parser.add_argument("-max_pos", default=512, type=int)
    parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)
    parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-load_from_extractive", default='', type=str)

    parser.add_argument("-sep_optim", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-lr_bert", default=2e-3, type=float)
    parser.add_argument("-lr_dec", default=2e-3, type=float)
    parser.add_argument("-use_bert_emb", type=str2bool, nargs='?',const=True,default=False)

    parser.add_argument("-share_emb", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=False, default=False)
    parser.add_argument("-dec_dropout", default=0.2, type=float)
    parser.add_argument("-dec_layers", default=6, type=int)
    parser.add_argument("-dec_hidden_size", default=768, type=int)
    parser.add_argument("-dec_heads", default=8, type=int)
    parser.add_argument("-dec_ff_size", default=2048, type=int)
    parser.add_argument("-enc_hidden_size", default=512, type=int)
    parser.add_argument("-enc_ff_size", default=512, type=int)
    parser.add_argument("-enc_dropout", default=0.2, type=float)
    parser.add_argument("-enc_layers", default=6, type=int)

    # params for EXT
    parser.add_argument("-ext_dropout", default=0.2, type=float)
    parser.add_argument("-ext_layers", default=2, type=int)
    parser.add_argument("-ext_hidden_size", default=768, type=int)
    parser.add_argument("-ext_heads", default=8, type=int)
    parser.add_argument("-ext_ff_size", default=2048, type=int)

    parser.add_argument("-label_smoothing", default=0.1, type=float)
    parser.add_argument("-generator_shard_size", default=32, type=int)
    parser.add_argument("-alpha",  default=0.6, type=float)
    parser.add_argument("-beam_size", default=5, type=int)
    parser.add_argument("-min_length", default=15, type=int)
    parser.add_argument("-max_length", default=150, type=int)
    parser.add_argument("-max_tgt_len", default=140, type=int)

    # parser.add_argument("-param_init", default=0, type=float)
    # parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
    # parser.add_argument("-optim", default='adam', type=str)
    # parser.add_argument("-lr", default=0.0001, type=float)
    # parser.add_argument("-beta1", default= 0.9, type=float)
    # parser.add_argument("-beta2", default=0.999, type=float)
    # parser.add_argument("-warmup_steps", default=1000, type=int)
    # parser.add_argument("-warmup_steps_bert", default=8000, type=int)
    # parser.add_argument("-warmup_steps_dec", default=8000, type=int)
    # parser.add_argument("-max_grad_norm", default=0, type=float)

    # parser.add_argument("-save_checkpoint_steps", default=500, type=int)
    # parser.add_argument("-accum_count", default=1, type=int)
    # parser.add_argument("-report_every", default=10, type=int)
    # parser.add_argument("-train_steps", default=10000, type=int)
    # parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)
    # parser.add_argument("-valid_steps", default=500, type=int)

    parser.add_argument('-visible_gpus', default='1', type=str)
    parser.add_argument('-gpu_ranks', default='0', type=str)
    parser.add_argument('-log_dir', default='logs/traineval')

    # Eval
    parser.add_argument("-test_from", default='models/model_w3_fixed_step_50000.pt')
    parser.add_argument("-threshold", default=0.0, type=float)

    args = parser.parse_args()
    device = 'cuda'
    device_id = 0

    # tokenize using stanford NLP
    tokenize()

    checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
    model = BertSeparator(args, device_id, checkpoint)

    inferencer = SepInference(args, model, device)
    inferencer.make_eval(doc=json_to_bert())
