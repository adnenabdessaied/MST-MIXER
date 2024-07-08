import os
import pandas as pd
import h5py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer
from itertools import chain


ADDITIONAL_SPECIAL_TOKENS = [
    '<place_holder>', '<s0>', '<s1>', '<s2>', '<s3>', '<s4>', '<s5>']

SPECIAL_TOKENS_DICT = {
    'bos_token': '<s>',
    'eos_token': '</s>',
    'pad_token': '<pad>',
    'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS
}

S0_TOK = '<s0>'  # frame
S1_TOK = '<s1>'  # mot
S2_TOK = '<s2>'  # question

def load_file(file_name):
    annos = None
    if os.path.splitext(file_name)[-1] == '.csv':
        return pd.read_csv(file_name)
    with open(file_name, 'r') as fp:
        if os.path.splitext(file_name)[1]== '.txt':
            annos = fp.readlines()
            annos = [line.rstrip() for line in annos]
        if os.path.splitext(file_name)[1] == '.json':
            annos = json.load(fp)

    return annos


def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


class NextQADataset(Dataset):
    def __init__(self, config, split):

        super().__init__()
        self.config = config
        self.split = split
        self.bart_max_input_len = config['bart_max_input_len']
        self.bart_size = config['bart_size']
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-{}'.format(self.bart_size))
        self.vocab_size = self.tokenizer.vocab_size

        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        self.vocab_size += len(ADDITIONAL_SPECIAL_TOKENS)
        self.tokenizer.save_pretrained(os.path.join(self.config['log_dir'], 'bart_tokenizer'))

        sample_list_file = os.path.join(self.config['nextqa_root'], '{}.csv'.format(split))
        self.sample_list = load_file(sample_list_file)

        vid_feat_file = os.path.join(self.config['nextqa_vid_feat'], 'app_mot_{}.h5'.format(split))
        print('Load {}...'.format(vid_feat_file))
        self.frame_feats = {}
        self.mot_feats = {}
        with h5py.File(vid_feat_file, 'r') as fp:
            vids = fp['ids']
            feats = fp['feat']
            for vid, feat in zip(vids, feats):
                self.frame_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
                self.mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)

        if self.config['overfit_size'] > 0:
            self.sample_list = self.sample_list[:self.config['overfit_size']]
        
    def __len__(self):
        return len(self.sample_list)

    def get_video_feature(self, video_name):
        """
        :param video_name:
        :return:
        """
       
        app_feat = self.frame_feats[video_name]
        app_feat = torch.from_numpy(app_feat).type(torch.float32)

        mot_feat = self.mot_feats[video_name]
        mot_feat = torch.from_numpy(mot_feat).type(torch.float32)

        return app_feat, mot_feat


    def __getitem__(self, idx):
        cur_sample = self.sample_list.loc[idx]
        video_name, ques, ans, qid = str(cur_sample['video']), str(cur_sample['question']),\
                                    str(cur_sample['answer']), str(cur_sample['qid'])
        
        input_ids = tokenize(ques, self.tokenizer)
        lm_labels = tokenize(ans, self.tokenizer)

        app_feat, mot_feat = self.get_video_feature(video_name)

        bos, eos, ques_state = self.tokenizer.convert_tokens_to_ids(['<s>', '</s>', '<s2>'])

        # Add state tokens
        input_ids.insert(0, ques_state)
        lm_labels.append(eos)
        question_interval = [0, len(input_ids)]

        input_ids = torch.Tensor(input_ids).long()
        lm_labels = torch.Tensor(lm_labels).long()

        return input_ids, lm_labels, app_feat, mot_feat, question_interval, video_name


    def padding(self, seq, pad_token, max_len=None):
        if max_len is None:
            max_len = max([i.size(0) for i in seq])
        if len(seq[0].size()) == 1:
            result = torch.ones((len(seq), max_len)).long() * pad_token
        else:
            result = torch.ones((len(seq), max_len, seq[0].size(-1))).float()
        for i in range(len(seq)):
            result[i, :seq[i].size(0)] = seq[i]
        return result

    def collate_fn(self, batch):
        input_ids_list, lm_labels_list, app_feat_list, mot_feat_list, question_interval_list, vid_ids_list = [], [], [], [], [], []
        for i in batch:
            input_ids_list.append(i[0])
            lm_labels_list.append(i[1])
            app_feat_list.append(i[2])
            mot_feat_list.append(i[3])
            question_interval_list.append(i[4])
            vid_ids_list.append(i[5])

        app_feats = torch.stack(app_feat_list, dim=0).float()
        mot_feats = torch.stack(mot_feat_list, dim=0).float()

        question_intervals = np.array(question_interval_list)

        pad_token, app_sep, mot_sep, ph_token = self.tokenizer.convert_tokens_to_ids(
            ['<pad>', '<s0>', '<s1>', '<place_holder>'])

        # All the visual features will not be masked because we do not perform any padding on them
        video_mask = torch.ones((len(batch), 16*2 + 2)) == 1  # NOTE *2: 2 modalities | +2: the state tokens | each modality has length 16
        # Now we create a dummy input for the video tokens (sole purpose is to reserve the spot of the seperators)
        dummy = torch.ones((len(batch), 16)) * ph_token
        video_place_holder_ids = torch.cat(
            [torch.ones((len(batch), 1)) * app_sep, dummy,
             torch.ones((len(batch), 1)) * mot_sep, dummy,
            ], dim=-1).long()

        input_ids = self.padding(input_ids_list, pad_token)
        lm_labels = self.padding(lm_labels_list, -100)
        text_mask = input_ids != pad_token
        input_mask = torch.cat([video_mask, text_mask], dim=1)

        # Now we get the intervals of the visual input tokens
        # Here the interval do not change across the batch dimension
        app_interval = [0, 16 + 1]  # the last token is not part of this modality
        mot_interval = [16 + 1, 2 * 16 + 2]
        vis_state_vector_idx = [app_interval[0], mot_interval[0]]

        # adapt the question and history interval -- shifted to the right by the visual input length
        question_intervals += 2 * 16 + 2
        question_intervals = question_intervals.tolist()
        
        question_state_vector_idx = [x[0] for x in question_intervals]
        
        batch = {
            'input_ids': input_ids,
            'video_place_holder_ids': video_place_holder_ids,
            'app_feats': app_feats,
            'mot_feats': mot_feats,
            'lm_labels': lm_labels,
            'input_mask': input_mask,
            'app_interval': app_interval,
            'mot_interval': mot_interval,
            'question_intervals': question_intervals,
            'vis_state_vector_idx': vis_state_vector_idx,
            'question_state_vector_idx': question_state_vector_idx
        }
        return batch

def get_dataset(config, split):
    
    bart_max_input_len = config['bart_max_input_len']
    bart_size = config['bart_size']

    sample_list_file = os.path.join(config['nextqa_root'], '{}.csv'.format(split))
    sample_list = load_file(sample_list_file)

    vid_feat_file = os.path.join(config['nextqa_vid_feat'], 'app_mot_{}.h5'.format(split))
    print('Load {}...'.format(vid_feat_file))
    app_feats = {}
    mot_feats = {}
    with h5py.File(vid_feat_file, 'r') as fp:
        vids = fp['ids']
        feats = fp['feat']
        for vid, feat in zip(vids, feats):
            app_feats[str(vid)] = feat[:, :2048]  # (16, 2048)
            mot_feats[str(vid)] = feat[:, 2048:]  # (16, 2048)
    
    return sample_list, app_feats, mot_feats

