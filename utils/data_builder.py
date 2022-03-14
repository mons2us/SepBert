import gc
from glob import glob
import json
import os
import random
import re
import subprocess
from collections import Counter
from tqdm import tqdm

import torch
from multiprocess import Pool

import gluonnlp as nlp
from kobert.utils import get_tokenizer, tokenizer
from kobert.utils import download as _download
from kobert.pytorch_kobert import get_pytorch_kobert_model
from gluonnlp.data import SentencepieceTokenizer

from others.logging import logger
from others.tokenization import BertTokenizer
#from pytorch_transformers import XLNetTokenizer

from others.utils import clean


def get_kobert_vocab(cachedir="./tmp/"):
    # Add BOS,EOS vocab
    vocab_info = tokenizer
    vocab_file = _download(
        vocab_info["url"], vocab_info["fname"], vocab_info["chksum"], cachedir=cachedir
    )

    vocab_b_obj = nlp.vocab.BERTVocab.from_sentencepiece(
        vocab_file, padding_token="[PAD]", bos_token="[BOS]", eos_token="[EOS]"
    )

    return vocab_b_obj


nyt_remove_words = ["photo", "graph", "chart", "map", "table", "drawing"]

def recover_from_corenlp(s):
    s = re.sub(r' \'{\w}', '\'\g<1>', s)
    s = re.sub(r'\'\' {\w}', '\'\'\g<1>', s)


def load_json(p, lower):
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

def load_pt(pth):
    loaded = torch.load(pth)
    return loaded


def divide_dataset(args, tot_len):
    train_ratio = args.train_ratio
    train_len = int(tot_len * train_ratio)
    valid_len = (tot_len - train_len) // 2
    test_len = tot_len - (train_len + valid_len)
    return train_len, valid_len, test_len


def tokenize(args):
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            if (not s.endswith('story')):
                continue
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-annotators', 'tokenize,ssplit',
                '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat',
                'json', '-outputDirectory', tokenized_stories_dir]
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
                tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


class BertData:
    def __init__(self, args, vocab, tokenizer):
        self.args = args
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'

        self.pad_idx = self.vocab["[PAD]"]
        self.cls_idx = self.vocab["[CLS]"]
        self.sep_idx = self.vocab["[SEP]"]
        self.mask_idx = self.vocab["[MASK]"]
        self.bos_idx = self.vocab["[BOS]"]
        self.eos_idx = self.vocab["[EOS]"]

    def preprocess(self, src, is_test=False):
        if len(src) == 0:
            return None

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens_per_sent)]

        src_txt = src

        src = [src[i][: self.args.max_src_ntokens_per_sent] for i in idxs]
        src = src[: self.args.max_src_nsents]        
        src = [self.tokenizer(sent) for sent in src]

        src_subtokens = [[self.cls_token] + sent + [self.sep_token] for sent in src]
        # src_subtokens = ' {} {} '.format(self.sep_token, self.cls_token).join([self.tokenizer(sent) for sent in src])
        # #src_subtokens = self.tokenizer(src)
        # src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        src_token_ids = [self.tokenizer.convert_tokens_to_ids(s) for s in src_subtokens]
        #src_subtoken_idxs = [self.add_special_token(lines) for lines in src_token_ids]
        src_subtoken_idxs = [lines for lines in src_token_ids]

        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_idx]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_idx]

        segments_ids = self.get_token_type_ids(src_subtoken_idxs)

        src_subtoken_idxs = [x for sublist in src_subtoken_idxs for x in sublist]
        segments_ids = [x for sublist in segments_ids for x in sublist]

        # tgt_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(tgt)[: self.args.max_tgt_ntokens]
        # tgt_subtoken_idxs = self.add_sentence_token(tgt_subtoken_idxs)

        # if (not is_test) and len(tgt_subtoken_idxs) < self.args.min_tgt_ntokens:
        #     return None

        cls_ids = self.get_cls_index(src_subtoken_idxs)
        return src_subtoken_idxs, segments_ids, cls_ids, src_txt

    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def add_sentence_token(self, token_ids):
        return [self.bos_idx] + token_ids + [self.eos_idx]

    def get_token_type_ids(self, src_token):
        seg = []
        for i, v in enumerate(src_token):
            if i % 2 == 0:
                seg.append([0] * len(v))
            else:
                seg.append([1] * len(v))
        return seg

    def get_cls_index(self, src_doc):
        cls_index = [index for index, value in enumerate(src_doc) if value == self.cls_idx]
        return cls_index

# class BertData():
#     def __init__(self, args):
#         self.args = args
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#         self.sep_token = '[SEP]'
#         self.cls_token = '[CLS]'
#         self.pad_token = '[PAD]'
#         self.tgt_bos = '[unused0]'
#         self.tgt_eos = '[unused1]'
#         self.tgt_sent_split = '[unused2]'
#         self.sep_vid = self.tokenizer.vocab[self.sep_token]
#         self.cls_vid = self.tokenizer.vocab[self.cls_token]
#         self.pad_vid = self.tokenizer.vocab[self.pad_token]

#     def preprocess(self, src, use_bert_basic_tokenizer=False, is_test=False):
#         if ((not is_test) and len(src) == 0):
#             return None

#         idxs = [i for i, s in enumerate(src)]
#         src_txt = src

#         # add cls/sep tokens front and end
#         text = ' {} {} '.format(self.sep_token, self.cls_token).join(src_txt)
#         src_subtokens = self.tokenizer.tokenize(text)
#         src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]
#         src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)

#         _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
#         segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
#         segments_ids = []
#         for i, s in enumerate(segs):
#             if (i % 2 == 0):
#                 segments_ids += s * [0]
#             else:
#                 segments_ids += s * [1]
#         cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]

#         return src_subtoken_idxs, segments_ids, cls_ids, src_txt


# --------------------------------
#    make dataset for training
#    BertSep model
# --------------------------------
def generate_sepdata(args):
    '''
    Using train_articles, ..., this function turns them into splitted articles (Bert-tokenized)
    Before that, train_articles.pt should be made such that they include every sources
    from oooo.0.bert.pt
    '''
    vocab = get_kobert_vocab()
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)

    train_dataset_y = torch.load(os.path.join(args.dataset_path, args.data_type, 'train_ydocs.pt'))
    train_dataset_n = torch.load(os.path.join(args.dataset_path, args.data_type, 'train_ndocs.pt'))

    valid_dataset_y = torch.load(os.path.join(args.dataset_path, args.data_type, 'valid_ydocs.pt'))
    valid_dataset_n = torch.load(os.path.join(args.dataset_path, args.data_type, 'valid_ndocs.pt'))

    #data_list = torch.load(os.path.join(args.dataset_path, args.data_type, f'bert_data/{corpus}_dataset.pt'))

    # shuffle dataset
    # if True:
    #     random.shuffle(data_list)

    logger.info("Start making dataset---")
    trainset = _make_data(args=args, dataset=[train_dataset_y, train_dataset_n], use_stair=args.use_stair,
                            random_point=args.random_point, corpus_type='train', vocab=vocab, tokenizer=tokenizer)

    valset = _make_data(args=args, dataset=[valid_dataset_y, valid_dataset_n], use_stair=args.use_stair,
                        random_point=args.random_point, corpus_type='valid', vocab=vocab, tokenizer=tokenizer)

    logger.info("Done.")


    # memory clear
    trainset, valset = [], []
    gc.collect()


def _make_data(args, dataset, use_stair=True, random_point=False, corpus_type='', vocab=None, tokenizer=None):
    '''
    For given dataset(list), split them into y/n datasets and generate final dataset used for training.
    '''
    ws = args.window_size
    y_cands, n_cands = dataset[0], dataset[1]
    tot_len = dataset.__len__()
    n_len = len(n_cands)

    y_dataset, n_dataset = [], []

    # Define Bert preprocessor
    bert = BertData(args, vocab, tokenizer)

    # create mixed dataset (y)
    for lh, rh in tqdm(y_cands, desc="making y_dataset"):
        if not random_point:
            si = 0
            sj = 0
        else:
            si = random.randint(0, (len(y_cands[i]) - ws))
            sj = random.randint(0, (len(y_cands[i+1]) - ws))

        tmp_article_y = lh[si:si + ws] + rh[sj:sj + ws]
        y_dataset.append(tmp_article_y)
        #y_dataset.append({'src': tmp_article_y, 'label': 1})

    # create normal dataset (n)
    if use_stair and ws > 1:
            stair_idx = 0
            count_idx = 0
            stair_lh = [(s, ws * 2 - s) for s in range(1, ws)]
            stair_rh = [(ls, rs) if ls > rs else (rs, ls) for (ls, rs) in stair_lh]
            stairs = stair_lh + stair_rh
            single_num = n_len // (len(stairs) + 1) # average number of dataset per each stair

    for j in tqdm(range(n_len - 1), desc="making n_dataset"):
        si = 0 if (not random_point) else random.randint(0, (len(n_cands[j]) - ws * 2))
        if (not use_stair) or (ws == 1):
            tmp_article_n = n_cands[j][si:si + ws * 2]
            n_dataset.append(tmp_article_n) # append
        else:
            # !!TODO!! 여기 잘못만듦 random인 경우 n_cands[j+1]의 시작점도 따로 샘플링 해야됨
            if stair_idx <= (len(stairs) - 1):
                tmp_article_n = n_cands[j][si:si + stairs[stair_idx][0]] + n_cands[j+1][si:si + stairs[stair_idx][1]]
                n_dataset.append(tmp_article_n) # append
                count_idx += 1
                if count_idx == single_num: # go to next stair
                    stair_idx += 1
                    count_idx = 0
            else:
                tmp_article_n = n_cands[j][si:si + ws*2]
                n_dataset.append(tmp_article_n) # append

    # Preprocess using Bert preprocessor
    fin_dataset = []
    # !!TODO!! bert.preprocess 부분 imap 추가
    for lab_idx, _set in enumerate([n_dataset, y_dataset]):
        print(f"working on label index: {lab_idx}")
        for source in tqdm(_set):
            # if (args.lower):
            #     source = [s.lower() for s in source]
            src_subtoken_idxs, segments_ids, cls_ids, src_txt = bert.preprocess(source, is_test=False)
            data_dict = {'src': src_subtoken_idxs,
                        'segs': segments_ids,
                        'clss': cls_ids,
                        'src_txt': src_txt,
                        'sep_label': lab_idx}
            fin_dataset.append(data_dict)

    random.shuffle(fin_dataset)


    # save
    save_pth = os.path.join(args.dataset_path, args.data_type, f'bertsep_data/bertsep_dt_{corpus_type}_w{args.window_size}.pt')
    torch.save(fin_dataset, save_pth)
    logger.info(f"SepBert Dataset for {corpus_type} saved at: {save_pth}")

    return


def generate_basedata(args):
    '''
    From tokenized .json format documents,
    create list type articles which 
    '''
    target_dir = os.path.join(args.dataset_path, f'{args.data_type}/tokenized_texts')
    files = [f for f in glob(os.path.join(target_dir, '*.json'))]

    a_lst = [(f, args) for f in files]
    pool = Pool(args.n_cpus)

    dataset = []
    with tqdm(total=len(a_lst)) as pbar:
        for d in pool.imap_unordered(_format_to_lines, a_lst):
            dataset.append(d)
            pbar.update()

    pool.close()
    pool.join()

    # Clean documents
    def _clean_doc(documents):
        # remove beginning sign, e.g. (cnn) at the very beginning of an article if exists.
        # They might induce bias into the model
        # other than obvious beggining sign of an article can be removed using min_src_ntokens_per_sent
        clean_set = []
        for doc in documents:
            if not doc:
                continue
            
            if args.data_type == 'cnndm':
                # remove divider(--)
                if '--' in doc[0]:
                    cut_index = doc[0].index('--')
                    doc[0] = doc[0][cut_index + 1:]
                    
                if '(cnn)' in ''.join(doc[0]):
                    doc[0] = ' '.join(doc[0]).replace('( cnn )', '').strip().split(' ')

            doc_clean = [sent for sent in doc if
                            (len(sent) >= args.min_src_ntokens_per_sent) # 10
                        and (len(sent) < args.max_src_ntokens_per_sent)] # 200

            ## !! TODO !! args.min_src_nsents 현재 10인데 이거는 sep_acc 계산할 때만 하도록 수정
            if len(doc_clean) >= args.min_src_nsents: # 10
                doc_clean = [' '.join(sent).strip() for sent in doc_clean]
                clean_set.append(doc_clean)
        return clean_set

    tmp_num = len(dataset)
    dataset = _clean_doc(dataset)
    logger.info(f"Doc number after cleansing: {tmp_num} --> {len(dataset)}")

    # Divide into train/val/test
    if args.test_only:
        train_len, valid_len = 0, 0
    else:
        train_len, valid_len, _ = divide_dataset(args, len(dataset))
    
    random.shuffle(dataset)
    corpora = {'train': dataset[:train_len],
            'valid': dataset[train_len:train_len+valid_len],
            'test': dataset[train_len+valid_len:]}

    dataset = []
    gc.collect()

    for k in corpora.keys():
        save_pth = os.path.join(args.dataset_path, args.data_type, f'bert_data/{k}_dataset.pt')
        torch.save(corpora[k], save_pth)
        logger.info(f"Dataset: {k} saved. Length: {len(corpora[k])}")

def _format_to_lines(params):
    f, args = params
    source = load_json(f, args.lower)
    return source

