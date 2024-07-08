import os
import pickle
import pyhocon
from copy import deepcopy
import json
from tqdm import tqdm
import numpy as np
import torch
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
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

S0_TOK = '<s0>'  # I3D_flow
S1_TOK = '<s1>'  # I3D_rgb
S2_TOK = '<s2>'  # sam obj
S3_TOK = '<s3>'  # audio
S4_TOK = '<s4>'  # history
S5_TOK = '<s5>'  # question



def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


class AVSDDataset(Dataset):
    def __init__(self, config, split):

        super().__init__()
        self.config = config
        self.split = split
        self.bart_max_input_len = config['bart_max_input_len']
        self.bart_size = config['bart_size']
        self.cap_sum = config['cap_sum']
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-{}'.format(self.bart_size))
        self.vocab_size = self.tokenizer.vocab_size

        self.tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        self.vocab_size += len(ADDITIONAL_SPECIAL_TOKENS)
        self.tokenizer.save_pretrained(os.path.join(self.config['log_dir'], 'bart_tokenizer'))
        self.processed_dir = os.path.join(self.config['avsd_processed'], 'hist_with_{}_rounds'.format(self.config['n_history']), split)
        self.paths = list(map(lambda p: os.path.join(self.processed_dir, p), os.listdir(self.processed_dir)))

        if self.config['overfit'] > 0:
            self.paths = self.paths[:self.config['overfit_size']]
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        pth  = self.paths[index]
        with open(pth, 'rb') as f:
            item = pickle.load(f)

        question_sep = self.tokenizer.convert_tokens_to_ids('<s5>')

        input_ids = item['input_ids']
        history_end = (input_ids == question_sep).nonzero(as_tuple=True)[0]

        history_interval = [0, history_end.item()]  # The last token is the question state token (not part of the history)
        question_interval = [history_end.item(), input_ids.size(0)]

        lm_labels = item['lm_labels']
        i3d_rgb = item['i3d_rgb']
        i3d_flow = item['i3d_flow']
        sam = item['sam']
        vgg = item['vgg']
        vid = item['vid']

        return input_ids, lm_labels, history_interval, question_interval, i3d_rgb, i3d_flow, sam, vgg, vid

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
        input_ids_list, lm_labels_list, history_interval_list, question_interval_list, i3d_rgb_list, i3d_flow_list, sam_list, vggish_list, vid_ids_list = [], [], [], [], [], [], [], [], []
        for i in batch:
            input_ids_list.append(i[0])
            lm_labels_list.append(i[1])
            history_interval_list.append(i[2])
            question_interval_list.append(i[3])
            i3d_rgb_list.append(i[4])
            i3d_flow_list.append(i[5])
            sam_list.append(i[6])
            vggish_list.append(i[7])
            vid_ids_list.append(i[8])

        history_intervals = np.array(history_interval_list)
        question_intervals = np.array(question_interval_list)


        min_len_i3d_flow = min([feat.shape[0] for feat in i3d_flow_list])
        min_len_i3d_rgb = min([feat.shape[0] for feat in i3d_rgb_list])
        min_len_sam = min([feat.shape[0] for feat in sam_list])
        min_len_vggish = min([feat.shape[0] for feat in vggish_list])

        min_length = min([self.config['vis_feat_length'], min_len_i3d_flow, min_len_i3d_rgb, min_len_sam, min_len_vggish])

        # Sample equally-distant features from the visual features for each sample within the batch
        for i in range(len(i3d_rgb_list)):
            sample_idx_i3d_rgb = np.round(np.linspace(0, i3d_rgb_list[i].shape[0] - 1, min_length)).astype(int)
            i3d_rgb_list[i] = i3d_rgb_list[i][sample_idx_i3d_rgb, :]
        i3d_rgb = torch.from_numpy(np.array(i3d_rgb_list)).float()

        for i in range(len(i3d_flow_list)):
            sample_idx_i3d_flow = np.round(np.linspace(0, i3d_flow_list[i].shape[0] - 1, min_length)).astype(int)
            i3d_flow_list[i] = i3d_flow_list[i][sample_idx_i3d_flow, :]
        i3d_flow = torch.from_numpy(np.array(i3d_flow_list)).float()

        for i in range(len(sam_list)):
            sample_idx_sam = np.round(np.linspace(0, sam_list[i].shape[0] - 1, min_length)).astype(int)
            sam_list[i] = sam_list[i][sample_idx_sam, :]
        sam = torch.from_numpy(np.array(sam_list)).float()

        for i in range(len(vggish_list)):
            sample_idx_vggish = np.round(np.linspace(0, vggish_list[i].shape[0] - 1, min_length)).astype(int)
            vggish_list[i] = vggish_list[i][sample_idx_vggish, :]
        vggish = torch.from_numpy(np.array(vggish_list)).float()

        pad_token, i3d_flow_sep, i3d_rgb_sep, sam_sep, audio_sep, ph_token = self.tokenizer.convert_tokens_to_ids(
            ['<pad>', '<s0>', '<s1>', '<s2>', '<s3>', '<place_holder>'])

        # All the visual features will not be masked because we do not perform any padding on them
        video_mask = torch.ones((len(batch), min_length*4 + 4)) == 1  # NOTE *4: 4 modalities | +4: the state tokens
        # Now we create a dummy input for the video tokens (sole purpose is to reserve the spot of the seperators)
        dummy = torch.ones((len(batch), min_length)) * ph_token
        video_place_holder_ids = torch.cat(
            [torch.ones((len(batch), 1)) * i3d_rgb_sep, dummy,
             torch.ones((len(batch), 1)) * i3d_flow_sep, dummy,
             torch.ones((len(batch), 1)) * sam_sep, dummy,
             torch.ones((len(batch), 1)) * audio_sep, dummy,
            ], dim=-1).long()

        input_ids = self.padding(input_ids_list, pad_token)
        lm_labels = self.padding(lm_labels_list, -100)
        text_mask = input_ids != pad_token
        input_mask = torch.cat([video_mask, text_mask], dim=1)

        # Now we get the intervals of the visual input tokens
        # Here the interval do not change across the batch dimension
        i3d_rgb_interval = [0, min_length + 1]  # the last token is not part of this modality
        i3d_flow_interval = [min_length + 1, 2 * min_length + 2]
        sam_interval = [2 * min_length + 2, 3 * min_length + 3]
        audio_interval = [3 * min_length + 3, 4 * min_length + 4]
                
        vis_state_vector_idx = [i3d_rgb_interval[0], i3d_flow_interval[0], sam_interval[0], audio_interval[0]]

        # adapt the question and history interval -- shifted to the right by the visual input length
        history_intervals += 4 * min_length + 4
        question_intervals += 4 * min_length + 4
        history_intervals = history_intervals.tolist()
        question_intervals = question_intervals.tolist()
        
        history_state_vector_idx = [x[0] + 1 for x in history_intervals]  # +1 because the history starts with <s><s4> .....
        question_state_vector_idx = [x[0] for x in question_intervals]  # +1 because the history starts with <s><s4> .....
        
        batch = {
            'input_ids': input_ids,
            'video_place_holder_ids': video_place_holder_ids,
            'i3d_rgb': i3d_rgb,
            'i3d_flow': i3d_flow,
            'sam': sam,
            'vggish': vggish,
            'lm_labels': lm_labels,
            'input_mask': input_mask,
            'i3d_rgb_interval': i3d_rgb_interval,
            'i3d_flow_interval': i3d_flow_interval,
            'sam_interval': sam_interval,
            'audio_interval': audio_interval,
            'history_intervals': history_intervals,
            'question_intervals': question_intervals,
            'vis_state_vector_idx': vis_state_vector_idx,
            'history_state_vector_idx': history_state_vector_idx,
            'question_state_vector_idx': question_state_vector_idx
        }
        return batch


def get_dataset(config, split, tokenizer):
    if split != 'test':
        dialog_pth = config[f'avsd_{split}']
    else:
        dialog_pth = config['avsd_test_dstc{}'.format(config['dstc'])]
    n_history = config['n_history']
    dialog_data = json.load(open(dialog_pth, 'r'))
    dialog_list = []
    vid_set = set()
    undisclosed_only = split == 'test'
    pbar = tqdm(dialog_data['dialogs'])

    pbar.set_description('[INFO] Generating {} items | DSTC {}'.format(split, config['dstc']))
    for dialog in pbar:
        if config['dstc'] != 10:
            caption = [tokenize(dialog['caption'], tokenizer)] + [tokenize(dialog['summary'], tokenizer)]
        else:
            caption = [tokenize('no', tokenizer)]

        questions = [tokenize(d['question'], tokenizer) for d in dialog['dialog']]
        answers = [tokenize(d['answer'], tokenizer) for d in dialog['dialog']]
        vid = dialog["image_id"]
        vid_set.add(vid)
        if undisclosed_only:
            it = range(len(questions) - 1, len(questions))
        else:
            it = range(len(questions))
        qalist=[]
        history = []
        if undisclosed_only:
            for n in range(len(questions)-1):
                qalist.append(questions[n])
                qalist.append(answers[n])
            history=qalist[max(-len(qalist),-n_history*2):]
        for n in it:
            if undisclosed_only:
                assert dialog['dialog'][n]['answer'] == '__UNDISCLOSED__'
            question = questions[n]
            answer = answers[n]
            history.append(question)
            if n_history == 0:
                item = {'vid': vid, 'history': [question], 'answer': answer, 'caption': caption}
            else:
                item = {'vid': vid, 'history': history, 'answer': answer, 'caption': caption}
            dialog_list.append(item)
            qalist.append(question)
            qalist.append(answer)
            history=qalist[max(-len(qalist),-n_history*2):]

    all_features = {}
    fea_types = ['vggish', 'i3d_flow', 'i3d_rgb', 'sam']

    dataname = '<FeaType>/<ImageID>.npy'
    for ftype in fea_types:
        if undisclosed_only:
            basename = dataname.replace('<FeaType>', ftype+'_testset')
        else:
            basename = dataname.replace('<FeaType>', ftype)
        features = {}
        for vid in vid_set:
            filename = basename.replace('<ImageID>', vid)
            filepath = config['avsd_feature_path'] + filename
            features[vid] = filepath
        all_features[ftype] = features
    return dialog_list, all_features


def build_input_from_segments(caption, history_orig, reply, tokenizer, add_state_tokens=True, drop_caption=False):
    """ Build a sequence of input from 3 segments: caption(caption+summary) history and last reply """

    bos, eos, hist_state, ques_state = tokenizer.convert_tokens_to_ids(['<s>', '</s>', '<s4>', '<s5>'])
    sep = eos
    
    instance = {}
    instance["lm_labels"] = reply + [eos]
    caption = list(chain(*caption))

    # Add state tokens if applicable
    if add_state_tokens:
        caption.insert(0, hist_state)
        history = deepcopy(history_orig)
        history[-1].insert(0, ques_state)
    else:
        history = history_orig

    if not drop_caption:
        # sequence = [[bos] + list(chain(*caption))] + history + [reply + ([eos] if with_eos else [])]

        # NOTE It is important not to include the reply in the input of the encoder -- > the decoder will just
        # learn to copy it --> low train/val loss but no learning is happening
        sequence = [[bos] + caption + [eos]] + [[sep] + s for s in history] + [[eos]]
    else:
        sequence = [[bos]] + [[hist_state]] + [[sep] + s for s in history] + [[eos]]

    instance["input_ids"] = list(chain(*sequence))
    return instance


def parse_args():
    parser = ArgumentParser(description='debug dataloader')
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='train or val')

    parser.add_argument(
        '--model',
        type=str,
        default='mixer',
        help='model name to train or test')

    parser.add_argument(
        '--log_dataset',
        action='store_true',
        default=False,
        help='Whether or not to log the processed data')

    parser.add_argument(
        '--add_state_tokens',
        action='store_true',
        default=True,
        help='Whether or not to add state tokens')

    parser.add_argument(
        '--log_dir',
        type=str,
        default='processed/avsd',
        help='Output directory')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    split = args.split

    config = pyhocon.ConfigFactory.parse_file(
        'config/mst_mixer.conf')[args.model]
    config['expand_rnd'] = False
    config['debugging'] = False
    config['overfit'] = False
    args.log_dir = os.path.join(args.log_dir, 'hist_with_{}_rounds'.format(config['n_history']) ) 
    if args.log_dataset:
        log_dir = os.path.join(args.log_dir, split)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        tokenizer = BartTokenizer.from_pretrained('facebook/bart-{}'.format(config['bart_size']))
        tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
        dialogs, features = get_dataset(config, split, tokenizer)
        pbar = tqdm(dialogs)
        pbar.set_description('[{}] Logging processed data'.format(split))
        counter = 0
        for dialog in pbar:
            vid = dialog['vid']
            his = dialog['history']
            cap = dialog['caption']
            ans = dialog['answer']

            if np.random.rand() < config['caption_drop_rate']:
                instance = build_input_from_segments(
                    cap, his, ans, tokenizer, add_state_tokens=args.add_state_tokens, drop_caption=True)
            else:
                instance = build_input_from_segments(
                    cap, his, ans, tokenizer, add_state_tokens=args.add_state_tokens, drop_caption=False)
            
            input_ids = torch.Tensor(instance["input_ids"]).long()
            lm_labels = torch.Tensor(instance["lm_labels"]).long()

            vgg = np.load(features["vggish"][vid])
            i3d_flow = np.load(features["i3d_flow"][vid])
            i3d_rgb = np.load(features["i3d_rgb"][vid])
            sam = np.load(features["sam"][vid])

            item = {
                'input_ids': input_ids,
                'lm_labels': lm_labels,
                'i3d_rgb': i3d_rgb,
                'i3d_flow': i3d_flow,
                'sam': sam,
                'vgg': vgg,
                'vid': vid
            }
            counter += 1
            pth = os.path.join(log_dir, str(counter) + '.pkl')
            with open(pth, 'wb') as f:
                pickle.dump(item, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        avsd_dataset = AVSDDataset(config, 'val')
        avsd_dataloader = DataLoader(avsd_dataset, batch_size=4, shuffle=False, collate_fn=avsd_dataset.collate_fn)

        for i, data in enumerate(avsd_dataloader):
            print('{}/{}'.format(i, len(avsd_dataloader)))
        print(avsd_dataset.max_len)

    print('[INFO] Done...')
