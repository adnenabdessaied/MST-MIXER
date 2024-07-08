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
from models.nextqa_bart import AVSDBart
from time import time


def tokenize(obj, tokenizer):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


class NEXTQARunner(Runner):
    def __init__(self, config, tokenizer, vocab_size):
        super(NEXTQARunner, self).__init__(config)
        bart_config = BartConfig.from_json_file(self.config['bart_config'])

        self.model = AVSDBart.from_pretrained(
            'facebook/bart-{}'.format(self.config['bart_size']), config=bart_config)

        # Resize the embedding to match the vocab with additional special toks
        # This takes care of resizing weights of related parts of the network

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
        app_feats = batch['app_feats']
        mot_feats = batch['mot_feats']
        lm_labels = batch['lm_labels']
        input_mask = batch['input_mask']

        app_interval = batch['app_interval']
        mot_interval = batch['mot_interval']
        question_intervals = batch['question_intervals']
        vis_state_vector_idx = batch['vis_state_vector_idx']
        question_state_vector_idx = batch['question_state_vector_idx']
        ########################################################
        
        bart_output = self.model(
            input_ids=input_ids,
            video_place_holder_ids=video_place_holder_ids,
            i3d_rgb=app_feats,
            i3d_flow=mot_feats,
            attention_mask=input_mask,
            labels=lm_labels,
            i3d_rgb_interval=app_interval,
            i3d_flow_interval=mot_interval,
            question_intervals=question_intervals,
            vis_state_vector_idx=vis_state_vector_idx,
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


    def generate(self, dataset, app_feats, mot_feats, tag, tokenizer, start_idx_gen, end_idx_gen, gen_subset_num=None):

        self.model.eval()
        results = {}
        app_sep, mot_sep, ph_token = tokenizer.convert_tokens_to_ids(
            ['<s0>', '<s1>', '<place_holder>'])
        
        # Generate the repsonse for each round
        log.info('[INFO] Generating responses for {} samples'.format(len(dataset)))
        with torch.no_grad():
            counter = 0
            for idx in range(start_idx_gen, end_idx_gen):
                start_time = time()
                cur_sample = dataset.loc[idx]
                video_name, ques, ans, qid = str(cur_sample['video']), str(cur_sample['question']),\
                                            str(cur_sample['answer']), str(cur_sample['qid'])
                if video_name not in results:
                    results[video_name] = {}

                input_ids = tokenize(ques, tokenizer)

                app_feat = app_feats[video_name]
                app_feat = torch.from_numpy(app_feat).type(torch.float32)

                mot_feat = mot_feats[video_name]
                mot_feat = torch.from_numpy(mot_feat).type(torch.float32)

                bos, eos, ques_state = self.tokenizer.convert_tokens_to_ids(['<s>', '</s>', '<s2>'])

                # Add state tokens
                input_ids.insert(0, ques_state)

                input_ids = torch.Tensor(input_ids).long()

                dummy = torch.ones((1, 16)) * ph_token
                video_place_holder_ids = torch.cat(
                    [torch.ones((1, 1)) * app_sep, dummy,
                     torch.ones((1, 1)) * mot_sep, dummy,
                    ], dim=-1).long()
                
                # Now we get the intervals of the visual input tokens
                # Here the interval do not change across the batch dimension
                app_interval = [0, 16 + 1]  # the last token is not part of this modality
                mot_interval = [16 + 1, 2 * 16 + 2]
                vis_state_vector_idx = [app_interval[0], mot_interval[0]]

                response = self.beam_search_generation(
                    input_ids,
                    app_feat, mot_feat,
                    app_interval, mot_interval,
                    vis_state_vector_idx, video_place_holder_ids, tokenizer)

                # Decode the response
                response = self.tokenizer.decode(response)

                results[video_name][qid] = response
                time_elapsed = int(time() - start_time)
                print('Generating resonse {} / {} -- took {}s'.format(counter + 1, len(dataset), time_elapsed))
                counter += 1
                
        # Create a file with all responses
        file_name = 'results_nextqa_beam_depth_{}'.format(self.config['beam_depth'])
        if gen_subset_num is not None:
            file_name += f'-part_{gen_subset_num}'
        file_name = f'{tag}_' + file_name
        output_path = os.path.join(self.config['output_dir_nextqa'], file_name + '.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        log.info('Results logged to {}'.format(output_path))
        print(os.getcwd())
        # Switch back to training mode
        self.model.train()


    def beam_search_generation(
        self, input_ids,
        app_feat, mot_feat,
        app_interval, mot_interval,
        vis_state_vector_idx, video_place_holder_ids, tokenizer):

        eos_token = tokenizer.eos_token_id
        unk_token = tokenizer.unk_token_id
        question_sep = tokenizer.convert_tokens_to_ids('<s2>')

        gen_ans = [eos_token]
        hyplist = [([], 0.0, [eos_token])]
        best_state = None
        comp_hyplist = []

        app_feat = app_feat.unsqueeze(0).cuda()
        mot_feat = mot_feat.unsqueeze(0).cuda()
        video_place_holder_ids = video_place_holder_ids.cuda()
        text_shift_len = video_place_holder_ids.size(-1)

        question_intervals = [[0 + text_shift_len, input_ids.size(0) + text_shift_len]]  # The last token is the question state token (not part of the history)

        question_state_vector_idx = [x[0] for x in question_intervals] 
       
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
                    i3d_rgb=app_feat,
                    i3d_flow=mot_feat,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=decoder_input_ids,
                    i3d_rgb_interval=app_interval,
                    i3d_flow_interval=mot_interval,
                    question_intervals=question_intervals,
                    vis_state_vector_idx=vis_state_vector_idx,
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
