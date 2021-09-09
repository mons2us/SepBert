import os

import numpy as np
import torch
from tqdm import tqdm
import random
from tensorboardX import SummaryWriter

import distributed
from utils.data_loader import TextLoader
from utils.reporter import ReportMgr, Statistics
from others.logging import logger

def build_evaluator(args, device_id, model):
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    tensorboard_log_dir = args.model_path
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)
    evaluator = Evaluator(args, model, n_gpu, gpu_rank, report_manager)
    return evaluator


def build_sep_evaluator(args, device_id, model):
    n_gpu = args.world_size
    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    tensorboard_log_dir = args.model_path
    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")
    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)
    evaluator = SepEvaluator(args, model, device_id)
    return evaluator


class Evaluator:
    def __init__(self, args, model, n_gpu=1, gpu_rank=1, report_manager=None):
        self.args = args
        self.model = model
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

        # to eval mode
        self.model.eval()

    def test_cls(self, test_iter_fct=None):
        assert test_iter_fct
        logger.info("Evaluation: Classification Starts.")
        args = self.args
        test_iter = test_iter_fct()

        test_stats = Statistics()
        test_norm = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter)):
                test_norm += batch.batch_size
                src = batch.src
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask_src
                mask_cls = batch.mask_cls
                sep_label = batch.sep_label

                sep_pred = self.model(src, segs, clss, mask, mask_cls)
                test_loss = self.loss(sep_pred, sep_label.float())
                test_n_correct = (sep_label == (sep_pred > 0)).sum().to('cpu').item()
                
                test_batch_stats = Statistics(float(test_loss.detach().to('cpu').numpy()), test_norm, test_n_correct)
                test_stats.update(test_batch_stats)

                # initializae normalization denominator
                test_norm = 0
            self._report_step(0, 0, valid_stats=test_stats)

    def _report_step(self, learning_rate, step, train_stats=None,
                    valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    # def test_sep(self):
    #     args = self.args
    #     dataset_pth = os.path.join(args.dataset_path, args.data_type, f'bertsep_data/bertsep_dt_sep.pt')
    #     test_dataset = torch.load(dataset_pth)

    # def _test_sep_single_doc(self):
    #     pass



# !!TODO!! 문서가 3개 이상인 경우도 추가
class SepEvaluator:
    '''
    NLP tokenizing이 된 text를 input으로 사용
    해당 text 두개(7~10 + 7~10)를 붙여서 window size대로 슬라이딩을 하면서 bert tokenizing 수행
    '''
    def __init__(self, args, model, device):
        self.args = args
        self.text_loader = TextLoader(args, device)
        self.model = model

    def eval(self):
        args = self.args
        max_mode = args.test_max_mode

        # path to save result in .txt file
        save_pth = 'sep_result.txt'
        if os.path.exists(save_pth):
            os.remove(save_pth)

        def _separate(scores, max_mode='max_all'):
            if max_mode == 'max_all':
                sep_point = np.argmax(scores) if max(scores) > 0 else -1e9
                return sep_point, None
            elif max_mode == 'max_one':
                sep_point = np.where(scores > 0)[0]
                count_max = len(sep_point)
                sep_point = -1e9 if len(sep_point) != 1 else sep_point
                return sep_point, count_max

        ws = self.args.window_size
        mixed_doc_set = self.load_dataset()

        acc_cnt, err_cnt = 0, 0
        count_max = 0
        for idx, (d, _gt) in tqdm(enumerate(mixed_doc_set), total=len(mixed_doc_set)):
            scores = np.zeros(len(d) - 1)
            offset = ws - 1
            tmp_scores = self.eval_single_doc(d, _gt)
            scores[offset:len(scores) - offset] = tmp_scores

            pred_result, tmp_count_max = _separate(scores, max_mode)
            count_max += tmp_count_max if tmp_count_max else 0

            if pred_result == _gt:
                acc_cnt += 1
                self.print_result(save_pth, d, scores, True)
            else:
                err_cnt += 1
                self.print_result(save_pth, d, scores, False)

        if max_mode == 'max_all':
            print(f"Evaluation Result: {acc_cnt/(acc_cnt+err_cnt)*100:.2f}%")
        elif max_mode == 'max_one':
            print(f"Evaluation Result: {acc_cnt/(acc_cnt+err_cnt)*100:.2f}%  Average Sep Points: {count_max/(acc_cnt+err_cnt):.2f}")
        

    def load_dataset(self):
        args = self.args
        dataset = [data['src_txt'] for data in torch.load('dataset/cnndm/bert_data/test_articles.pt')]
        dataset = self._make_sepdata(dataset)
        return dataset


    def _make_sepdata(self, dataset):
        '''
        generate mixed documents for given list of cleaned data
        '''
        args = self.args

        mixed_doc_set = []
        assert args.test_sep_len < len(dataset)
        max_len = len(dataset) - 1 if args.test_sep_len == -1 else args.test_sep_len

        for i in range(max_len):
            lh_count = min(random.randint(7, 10), len(dataset[i]))
            rh_count = min(random.randint(7, 10), len(dataset[i+1]))
            lh_doc = dataset[i][:lh_count]
            rh_doc = dataset[i+1][:rh_count]
            gt = lh_count - 1

            src_doc = lh_doc + rh_doc
            mixed_doc_set.append((src_doc, gt))
        
        return mixed_doc_set
    

    def eval_single_doc(self, doc, gt):
        '''
        for a given mixed document,
        calculate sentence by sentence logit score to decide separation points.
        '''
        args = self.args
        ws = args.window_size
        cands = ['\n'.join(doc[i:i+ws*2]) for i in range(len(doc) - ws*2 + 1)]
        tmp_batch = self.text_loader.load_text(cands)

        logits = []
        for i, batch in enumerate(tmp_batch):
            (src, segs, clss, mask_src, mask_cls), _ = batch
            assert clss.shape[-1] == ws*2
            logit = self.model(src, segs, clss, mask_src, mask_cls).detach().to('cpu').item()
            logits.append(logit)

        return np.array(logits)


    def print_result(self, pth, doc, scores, flag):
        to_print = [0] * (len(doc) * 2 - 1)
        to_print[::2] = list(doc)
        to_print[1::2] = list(map(str, scores))

        prob_crit = 0.8 if flag else 0.3
        if random.random() > prob_crit:
            flag_text = '[CORRECT PREDICTION]\n' if flag else '[WRONG PREDICTION]\n'
            with open(pth, 'a') as file:
                file.write(flag_text)
                file.write('\n'.join(to_print))
                file.write('\n'*4)





