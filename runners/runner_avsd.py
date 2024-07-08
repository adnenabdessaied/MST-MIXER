import time
import os
import glog as log
import numpy as np
import json
import torch
import torch.nn.functional as F
from runners.runner import Runner
from copy import deepcopy
from optim_utils import init_optim
from transformers.models.bart.configuration_bart import BartConfig
from models.avsd_bart import AVSDBart

from custom_datasets.avsd import build_input_from_segments
from time import time


class AVSDRunner(Runner):
    def __init__(self, config, tokenizer, vocab_size):
        super(AVSDRunner, self).__init__(config)
        bart_config = BartConfig.from_json_file(self.config['bart_config'])

        self.model = AVSDBart.from_pretrained(
            'facebook/bart-{}'.format(self.config['bart_size']), config=bart_config)

        # Resize the embedding to match the vocab with additional special toks
        # This takes care of resizing weights of related parts of the network
        # pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print(pytorch_total_params)

        if vocab_size != bart_config.vocab_size:
            self.model.resize_token_embeddings(vocab_size)

        self.model.to(self.config['device'])
        if not self.config['generating']:
            self.optimizer, self.scheduler = init_optim(self.model, self.config)
        self.tokenizer = tokenizer

    def forward(self, batch):

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()

        ########################################################
        input_ids = batch['input_ids']
        video_place_holder_ids = batch['video_place_holder_ids']
        i3d_rgb = batch['i3d_rgb']
        i3d_flow = batch['i3d_flow']
        sam = batch['sam']
        vggish = batch['vggish']
        lm_labels = batch['lm_labels']
        input_mask = batch['input_mask']

        i3d_rgb_interval = batch['i3d_rgb_interval']
        i3d_flow_interval = batch['i3d_flow_interval']
        sam_interval = batch['sam_interval']
        audio_interval = batch['audio_interval']
        history_intervals = batch['history_intervals']
        question_intervals = batch['question_intervals']
        vis_state_vector_idx = batch['vis_state_vector_idx']
        history_state_vector_idx = batch['history_state_vector_idx']
        question_state_vector_idx = batch['question_state_vector_idx']

        ########################################################
        bart_output = self.model(
            input_ids=input_ids,
            video_place_holder_ids=video_place_holder_ids,
            i3d_rgb=i3d_rgb,
            i3d_flow=i3d_flow,
            sam=sam,
            vggish=vggish,
            attention_mask=input_mask,
            labels=lm_labels,
            i3d_rgb_interval=i3d_rgb_interval,
            i3d_flow_interval=i3d_flow_interval,
            sam_interval=sam_interval,
            audio_interval=audio_interval,
            history_intervals=history_intervals,
            question_intervals=question_intervals,
            vis_state_vector_idx=vis_state_vector_idx,
            history_state_vector_idx=history_state_vector_idx,
            question_state_vector_idx=question_state_vector_idx,
            output_attentions=True,
            return_dict=True
        )

        output = {}

        if self.config['print_output']:
            logits = bart_output['logits']
            probs = F.softmax(logits, dim=-1)
            preds = torch.topk(probs, 1)[1].squeeze(-1)
            preds = preds.tolist()
            lm_labels_list = lm_labels[:, 1:].tolist()
            lm_labels_list = [[s for s in label if s != -1] for label in lm_labels_list] 
            reponses = ''
            labels = ''
            for pred, label in zip(preds, lm_labels_list):
                reponses += self.tokenizer.decode(pred) + '\n'
                labels += self.tokenizer.decode(label) + '\n'
            
            output['reponses'] = reponses
            output['gt'] = labels
        

        gen_key = 'gen_loss (x{})'.format(self.config['gen_coeff'])
        gen_loss = bart_output['gen_loss']
        gen_loss = self.config['gen_coeff'] * gen_loss


        elbo_global_key = 'elbo_loss_global (x{})'.format(self.config['elbo_global_coeff'])
        if bart_output['elbo_loss_global'] is not None:
            elbo_global_loss = bart_output['elbo_loss_global']
            elbo_global_loss = self.config['elbo_global_coeff'] * elbo_global_loss
        else:
            elbo_global_loss = torch.tensor(0.0)

        elbo_local_key = 'elbo_loss_local (x{})'.format(self.config['elbo_local_coeff'])
        if bart_output['elbo_loss_local'] is not None:
            elbo_local_loss = bart_output['elbo_loss_local']
            elbo_local_loss = self.config['elbo_local_coeff'] * elbo_local_loss
        else:
            elbo_local_loss = torch.tensor(0.0)

        total_loss = gen_loss + elbo_global_loss + elbo_local_loss

        output['losses'] = {
            gen_key: gen_loss,
            elbo_local_key: elbo_local_loss,
            elbo_global_key: elbo_global_loss,
            'tot_loss': total_loss
        }
        del bart_output
        return output


    def generate(self, dataset, tag, tokenizer, gen_subset_num=None):

        self.model.eval()
        responses = {}
        i3d_flow_sep, i3d_rgb_sep, sam_sep, audio_sep, ph_token = tokenizer.convert_tokens_to_ids(
            ['<s0>', '<s1>', '<s2>', '<s3>', '<place_holder>'])
        
        # Generate the repsonse for each round
        log.info('[INFO] Generating responses for {} samples'.format(len(dataset)))
        with torch.no_grad():
            for counter, dialog in enumerate(dataset):
                start_time = time()
                vid = dialog['vid']

                i3d_rgb = np.load(os.path.join(self.config['avsd_i3d_rgb_test'], vid + '.npy'))
                i3d_flow = np.load(os.path.join(self.config['avsd_i3d_flow_test'], vid + '.npy'))
                sam = np.load(os.path.join(self.config['avsd_objects_test'], vid + '.npy'))
                vggish = np.load(os.path.join(self.config['avsd_audio_test'], vid + '.npy'))

                min_length = min([self.config['vis_feat_length'], i3d_rgb.shape[0], i3d_flow.shape[0], sam.shape[0], vggish.shape[0]])
                sample_idx_i3d_rgb = np.round(np.linspace(0, i3d_rgb.shape[0] - 1, min_length)).astype(int)
                sample_idx_i3d_flow = np.round(np.linspace(0, i3d_flow.shape[0] - 1, min_length)).astype(int)
                sample_idx_sam = np.round(np.linspace(0, sam.shape[0] - 1, min_length)).astype(int)
                sample_idx_vggish = np.round(np.linspace(0, vggish.shape[0] - 1, min_length)).astype(int)

                i3d_rgb = torch.from_numpy(i3d_rgb[sample_idx_i3d_rgb, :]).float()
                i3d_flow = torch.from_numpy(i3d_flow[sample_idx_i3d_flow, :]).float()
                sam = torch.from_numpy(sam[sample_idx_sam, :]).float()
                vggish = torch.from_numpy(vggish[sample_idx_vggish, :]).float()

                dummy = torch.ones((1, min_length)) * ph_token
                video_place_holder_ids = torch.cat(
                    [torch.ones((1, 1)) * i3d_rgb_sep, dummy,
                     torch.ones((1, 1)) * i3d_flow_sep, dummy,
                     torch.ones((1, 1)) * sam_sep, dummy,
                     torch.ones((1, 1)) * audio_sep, dummy,
                    ], dim=-1).long()
                # Now we get the intervals of the visual input tokens
                # Here the interval do not change across the batch dimension
                i3d_rgb_interval = [0, min_length + 1]  # the last token is not part of this modality
                i3d_flow_interval = [min_length + 1, 2 * min_length + 2]
                sam_interval = [2 * min_length + 2, 3 * min_length + 3]
                audio_interval = [3 * min_length + 3, 4 * min_length + 4]
                vis_state_vector_idx = [i3d_rgb_interval[0], i3d_flow_interval[0], sam_interval[0], audio_interval[0]]

                
                response = self.beam_search_generation(
                    dialog['caption'], dialog['history'],
                    i3d_rgb, i3d_flow, sam, vggish,
                    i3d_rgb_interval, i3d_flow_interval, sam_interval, audio_interval,
                    vis_state_vector_idx, video_place_holder_ids, tokenizer)

                # Decode the response
                response = self.tokenizer.decode(response)
                responses[vid] = response
                # all_graphs[vid] = graphs
                time_elapsed = int(time() - start_time)
                print('Generating resonse {} / {} -- took {}s'.format(counter + 1, len(dataset), time_elapsed))
                
        # Create a file with all responses
        with open(self.config['avsd_test_dstc{}'.format(self.config['dstc'])], 'r') as f:
            test_data = json.load(f)
        test_dialogs = deepcopy(test_data['dialogs'])
        # Filter the predicted dialogs
        test_dialogs = list(filter(lambda diag: diag['image_id'] in responses, test_dialogs))

        for i, dialog in enumerate(test_dialogs):
            vid_id = dialog['image_id']
            gen_response = responses[vid_id]
            round_num_to_answer = len(dialog['dialog'])-1
            assert dialog['dialog'][round_num_to_answer]['answer'] == '__UNDISCLOSED__'
            dialog['dialog'][round_num_to_answer]['answer'] = gen_response
            test_dialogs[i] = dialog

        # Log the file
        file_name = 'results_dstc{}_beam_depth_{}'.format(self.config['dstc'], self.config['beam_depth'])
        if gen_subset_num is not None:
            file_name += f'-part_{gen_subset_num}'
        file_name = f'{tag}_' + file_name
        output_path = os.path.join(self.config['output_dir_dstc{}'.format(self.config['dstc'])], file_name + '.json')
        with open(output_path, 'w') as f:
            json.dump({'dialogs': test_dialogs}, f, indent=4)
        log.info('Results logged to {}'.format(output_path))
        print(os.getcwd())
        # Switch back to training mode
        self.model.train()


    def beam_search_generation(
        self, caption, history,
        i3d_rgb, i3d_flow, sam, vggish,
        i3d_rgb_interval, i3d_flow_interval, sam_interval, audio_interval,
        vis_state_vector_idx, video_place_holder_ids, tokenizer):

        eos_token = tokenizer.eos_token_id
        unk_token = tokenizer.unk_token_id
        question_sep = tokenizer.convert_tokens_to_ids('<s5>')

        gen_ans = [eos_token]
        hyplist = [([], 0.0, [eos_token])]
        best_state = None
        comp_hyplist = []

        i3d_rgb = i3d_rgb.unsqueeze(0).cuda()
        i3d_flow = i3d_flow.unsqueeze(0).cuda()
        sam = sam.unsqueeze(0).cuda()
        vggish = vggish.unsqueeze(0).cuda()
        video_place_holder_ids = video_place_holder_ids.cuda()
        text_shift_len = video_place_holder_ids.size(-1)

        drop_caption = self.config['dstc'] == 10
        instance = build_input_from_segments(caption, history, gen_ans, tokenizer, drop_caption=drop_caption)

        input_ids = torch.tensor(instance['input_ids'])
        history_end = (input_ids == question_sep).nonzero(as_tuple=True)[0]
        history_intervals = [[0 + text_shift_len, history_end.item() + text_shift_len]]  # The last token is the question state token (not part of the history)
        question_intervals = [[history_end.item() + text_shift_len, input_ids.size(0) + text_shift_len]]

        history_state_vector_idx = [x[0] + 1 for x in history_intervals]  # +1 because the history starts with <s><s4> .....
        question_state_vector_idx = [x[0] for x in question_intervals]  # +1 because the history starts with <s><s4> .....
       
        input_ids = input_ids.long().cuda().unsqueeze(0)
        encoder_outputs = None

        for i in range(self.config['max_generation_length']):
            new_hyplist = []
            argmin = 0
            for out, lp, st in hyplist:
                decoder_input_ids = torch.tensor(st).long().cuda().unsqueeze(0)

                bart_output = self.model(
                    input_ids=input_ids,
                    video_place_holder_ids=video_place_holder_ids,
                    i3d_rgb=i3d_rgb,
                    i3d_flow=i3d_flow,
                    sam=sam,
                    vggish=vggish,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    i3d_rgb_interval=i3d_rgb_interval,
                    i3d_flow_interval=i3d_flow_interval,
                    sam_interval=sam_interval,
                    audio_interval=audio_interval,
                    history_intervals=history_intervals,
                    question_intervals=question_intervals,
                    vis_state_vector_idx=vis_state_vector_idx,
                    history_state_vector_idx=history_state_vector_idx,
                    question_state_vector_idx=question_state_vector_idx,
                    output_attentions=True,
                    generate=True,
                    return_dict=True
                )

                if encoder_outputs is None:
                    encoder_outputs = [
                        bart_output['encoder_last_hidden_state'],
                        bart_output['encoder_hidden_states'],
                        bart_output['encoder_attentions'],
                        bart_output['encoder_QAs_local'],
                        bart_output['encoder_PAs_local'],
                        bart_output['encoder_QA_global'],
                        bart_output['encoder_PA_global'],
                        bart_output['encoder_state_vectors']
                    ]

                logits = bart_output['logits'][:,-1,:].squeeze()  # get the logits of the last token
                logp = F.log_softmax(logits, dim=0)
                lp_vec = logp.cpu().data.numpy() + lp
                if i >= self.config['min_generation_length']:
                    new_lp = lp_vec[eos_token] + self.config['length_penalty'] * (len(out) + 1)
                    comp_hyplist.append((out, new_lp))
                    if best_state is None or best_state < new_lp:
                        best_state = new_lp
                count = 1
                for o in np.argsort(lp_vec)[::-1]:  # reverse the order
                    if o in [eos_token, unk_token]:
                        continue
                    new_lp = lp_vec[o]
                    if len(new_hyplist) == self.config['beam_depth']:
                        if new_hyplist[argmin][1] < new_lp:
                            new_st = deepcopy(st)
                            new_st.append(int(o))
                            new_hyplist[argmin] = (out + [o], new_lp, new_st)
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                        else:
                            break
                    else:
                        new_st = deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist.append((out + [o], new_lp, new_st))
                        if len(new_hyplist) == self.config['beam_depth']:
                            argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    count += 1
            hyplist = new_hyplist
        
        if len(comp_hyplist) > 0:
            maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1]
            return maxhyps[0][0]
        else:
            return []
