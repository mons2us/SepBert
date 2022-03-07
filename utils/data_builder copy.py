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
    corpus_type = ['train', 'valid', 'test']
    vocab = get_kobert_vocab()
    tokenizer = nlp.data.BERTSPTokenizer(get_tokenizer(), vocab, lower=False)

    for corpus in corpus_type:
        logger.info(f"corpus type: {corpus}")
        data_list = torch.load(os.path.join(args.dataset_path, args.data_type, f'bert_data/{corpus}_dataset.pt'))

        # shuffle dataset
        if True:
            random.shuffle(data_list)

        logger.info("Start making dataset---")
        fin_dataset = _make_data(args=args,
                                dataset=data_list,
                                y_ratio=args.y_ratio,
                                use_stair=args.use_stair,
                                random_point=args.random_point,
                                corpus_type=corpus,
                                vocab=vocab,
                                tokenizer=tokenizer)

        logger.info("Done.")

        # save
        fix_flag = 'random' if args.random_point else 'fixed'
        save_pth = os.path.join(args.dataset_path, args.data_type, f'bertsep_data/bertsep_dt_{corpus}_w{args.window_size}_{fix_flag}.pt')
        torch.save(fin_dataset, save_pth)
        logger.info(f"SepBert Dataset for {corpus} saved at: {save_pth}")

    # memory clear
    fin_dataset, sep_dataset = [], []
    gc.collect()


def _make_data(args, dataset, y_ratio=0.5, use_stair=True, random_point=False, corpus_type='', vocab=None, tokenizer=None):
    '''
    For given dataset(list), split them into y/n datasets and generate final dataset used for training.
    '''
    ws = args.window_size
    y_ratio = 0.5 if corpus_type in ['valid', 'test'] else y_ratio
    
    tot_len = dataset.__len__()
    y_len = int(tot_len * y_ratio)
    n_len = tot_len - y_len

    y_cands, n_cands = dataset[:y_len], dataset[y_len:]
    y_dataset, n_dataset = [], []

    # Define Bert preprocessor
    bert = BertData(args, vocab, tokenizer)

    # create mixed dataset (y)
    for i in tqdm(range(y_len - 1), desc="making y_dataset"):
        if not random_point:
            si = 0
            sj = 0
        else:
            si = random.randint(0, (len(y_cands[i]) - ws))
            sj = random.randint(0, (len(y_cands[i+1]) - ws))

        tmp_article_y = y_cands[i][si:si + ws] + y_cands[i+1][sj:sj + ws]
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
    return fin_dataset


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



# def format_to_bert(args, corpora):
#     corpus_type = ['train', 'valid', 'test']
#     for corpus in corpus_type:
#         a_lst = []
#         for json_f in glob(os.path.join(args.raw_path, '*' + corpus_type + '.*.json')):
#             real_name = json_f.split('/')[-1]
#             a_lst.append((corpus_type, json_f, args, os.path.join(args.save_path, real_name.replace('json', 'bert.pt'))))

#         pool = Pool(args.n_cpus)
#         for d in pool.imap(_format_to_bert, a_lst):
#             pass

#         pool.close()
#         pool.join()


# def _format_to_bert(params):
#     corpus_type, json_file, args, save_file = params
#     is_test = corpus_type == 'test'
#     if (os.path.exists(save_file)):
#         logger.info('Ignore %s' % save_file)
#         return

#     bert = BertData(args)

#     logger.info('Processing %s' % json_file)
#     jobs = json.load(open(json_file))
#     datasets = []
#     for d in jobs:
#         source, tgt = d['src'], d['tgt']
#         sent_labels = greedy_selection(source[:args.max_src_nsents], tgt, 3)
#         if (args.lower):
#             source = [' '.join(s).lower().split() for s in source]
#             tgt = [' '.join(s).lower().split() for s in tgt]
#         b_data = bert.preprocess(source, tgt, sent_labels, use_bert_basic_tokenizer=args.use_bert_basic_tokenizer, is_test=is_test)

#         if (b_data is None):
#             continue
#         src_subtoken_idxs, sent_labels, tgt_subtoken_idxs, segments_ids, cls_ids, src_txt, tgt_txt = b_data
#         b_data_dict = {"src": src_subtoken_idxs, "tgt": tgt_subtoken_idxs,
#                         "src_sent_labels": sent_labels, "segs": segments_ids, 'clss': cls_ids,
#                         'src_txt': src_txt, "tgt_txt": tgt_txt}
#         datasets.append(b_data_dict)
#     logger.info('Processed instances %d' % len(datasets))
#     logger.info('Saving to %s' % save_file)
#     torch.save(datasets, save_file)
#     # remove object and collect garbage
#     datasets = []
#     gc.collect()


















# def format_to_lines(args):
#     corpus_mapping = {}
#     for corpus_type in ['valid', 'test', 'train']:
#         temp = []
#         for line in open(pjoin(args.map_path, 'mapping_' + corpus_type + '.txt')):
#             temp.append(hashhex(line.strip()))
#         corpus_mapping[corpus_type] = {key.strip(): 1 for key in temp}
#     train_files, valid_files, test_files = [], [], []
#     for f in glob.glob(pjoin(args.raw_path, '*.json')):
#         real_name = f.split('/')[-1].split('.')[0]
#         if (real_name in corpus_mapping['valid']):
#             valid_files.append(f)
#         elif (real_name in corpus_mapping['test']):
#             test_files.append(f)
#         elif (real_name in corpus_mapping['train']):
#             train_files.append(f)
#         # else:
#         #     train_files.append(f)

#     corpora = {'train': train_files, 'valid': valid_files, 'test': test_files}
#     for corpus_type in ['train', 'valid', 'test']:
#         a_lst = [(f, args) for f in corpora[corpus_type]]
#         pool = Pool(args.n_cpus)
#         dataset = []
#         p_ct = 0
#         for d in pool.imap_unordered(_format_to_lines, a_lst):
#             dataset.append(d)
#             if (len(dataset) > args.shard_size):
#                 pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#                 with open(pt_file, 'w') as save:
#                     # save.write('\n'.join(dataset))
#                     save.write(json.dumps(dataset))
#                     p_ct += 1
#                     dataset = []

#         pool.close()
#         pool.join()
#         if (len(dataset) > 0):
#             pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#             with open(pt_file, 'w') as save:
#                 # save.write('\n'.join(dataset))
#                 save.write(json.dumps(dataset))
#                 p_ct += 1
#                 dataset = []

# ------------------------
#     using customset
#      (no urls)
# ------------------------
# def format_to_lines(args):
#     corpus_mapping = {}
#     test_files = []
#     for f in glob(pjoin(args.raw_path, '*.json')):
#         test_files.append(f)

#     corpora = {'test': test_files}
#     for corpus_type in ['test']:
#         a_lst = [(f, args) for f in corpora[corpus_type]]
#         pool = Pool(args.n_cpus)
#         dataset = []
#         p_ct = 0
#         for d in pool.imap_unordered(_format_to_lines, a_lst):
#             dataset.append(d)
#             if (len(dataset) > args.shard_size):
#                 pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#                 with open(pt_file, 'w') as save:
#                     # save.write('\n'.join(dataset))
#                     save.write(json.dumps(dataset))
#                     p_ct += 1
#                     dataset = []

#         pool.close()
#         pool.join()
#         if (len(dataset) > 0):
#             pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#             with open(pt_file, 'w') as save:
#                 # save.write('\n'.join(dataset))
#                 save.write(json.dumps(dataset))
#                 p_ct += 1
#                 dataset = []


# def _format_to_lines(params):
#     f, args = params
#     print(f)
#     source, tgt = load_json(f, args.lower)
#     return {'src': source, 'tgt': tgt}


# def format_xsum_to_lines(args):
#     if (args.dataset != ''):
#         datasets = [args.dataset]
#     else:
#         datasets = ['train', 'test', 'valid']

#     corpus_mapping = json.load(open(pjoin(args.raw_path, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')))

#     for corpus_type in datasets:
#         mapped_fnames = corpus_mapping[corpus_type]
#         root_src = pjoin(args.raw_path, 'restbody')
#         root_tgt = pjoin(args.raw_path, 'firstsentence')
#         # realnames = [fname.split('.')[0] for fname in os.listdir(root_src)]
#         realnames = mapped_fnames

#         a_lst = [(root_src, root_tgt, n) for n in realnames]
#         pool = Pool(args.n_cpus)
#         dataset = []
#         p_ct = 0
#         for d in pool.imap_unordered(_format_xsum_to_lines, a_lst):
#             if (d is None):
#                 continue
#             dataset.append(d)
#             if (len(dataset) > args.shard_size):
#                 pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#                 with open(pt_file, 'w') as save:
#                     save.write(json.dumps(dataset))
#                     p_ct += 1
#                     dataset = []

#         pool.close()
#         pool.join()
#         if (len(dataset) > 0):
#             pt_file = "{:s}.{:s}.{:d}.json".format(args.save_path, corpus_type, p_ct)
#             with open(pt_file, 'w') as save:
#                 save.write(json.dumps(dataset))
#                 p_ct += 1
#                 dataset = []


# def _format_xsum_to_lines(params):
#     src_path, root_tgt, name = params
#     f_src = pjoin(src_path, name + '.restbody')
#     f_tgt = pjoin(root_tgt, name + '.fs')
#     if (os.path.exists(f_src) and os.path.exists(f_tgt)):
#         print(name)
#         source = []
#         for sent in open(f_src):
#             source.append(sent.split())
#         tgt = []
#         for sent in open(f_tgt):
#             tgt.append(sent.split())
#         return {'src': source, 'tgt': tgt}
#     return None




# # Dataset Generation
# # for separation accuracy (not classification acc!)
# def _make_sepdata(args, dataset):
#     '''
#     generate mixed documents for given list of cleaned data
#     '''
#     # Define Bert preprocessor
#     bert = BertData(args)

#     # Generate mixed documents
#     mixed_doc_set = []
#     assert args.test_sep_len < len(dataset)
#     max_len = len(dataset) - 1 if args.test_sep_len == -1 else args.test_sep_len

#     for i in range(max_len):
#         lh_count = min(random.randint(7, 10), len(dataset[i]))
#         rh_count = min(random.randint(7, 10), len(dataset[i+1]))
#         lh_doc = dataset[i][:lh_count]
#         rh_doc = dataset[i+1][:rh_count]
#         gt = lh_count - 1

#         src_doc = lh_doc + rh_doc
#         mixed_doc_set.append((src_doc, gt))
    
#     fin_dataset = []
#     for idx, (d, _gt) in enumerate(mixed_doc_set):
#         source, tgt = d, ''
#         sent_labels = ''
#         src_subtoken_idxs, segments_ids, cls_ids, src_txt = bert.preprocess(source,
#                                                                             use_bert_basic_tokenizer=args.use_bert_basic_tokenizer,
#                                                                             is_test=False)
#         data_dict = {'src': src_subtoken_idxs,
#                     'segs': segments_ids,
#                     'clss': cls_ids,
#                     'src_txt': src_txt,
#                     'sep_gt': _gt}
#         fin_dataset.append(data_dict)

#     random.shuffle(fin_dataset)
#     return fin_dataset