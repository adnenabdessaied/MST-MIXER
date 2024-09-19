import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput
from typing import Optional, Tuple


class ELBO(nn.Module):
    def __init__(self):
        super(ELBO, self).__init__()
    
    def forward(self, QA, PA):
        QA_flattened = QA.view(-1).unsqueeze(-1)
        PA_flattened = PA.view(-1).unsqueeze(-1)

        QA_flattened = torch.cat([torch.zeros_like(QA_flattened), QA_flattened], dim=-1)
        PA_flattened = torch.cat([torch.zeros_like(PA_flattened), PA_flattened], dim=-1)

        log_QA = F.log_softmax(QA_flattened, dim=1)
        log_PA = F.log_softmax(PA_flattened, dim=1)

        QA_dist = torch.exp(log_QA)

        loss_QA = torch.mean(log_QA * QA_dist)
        loss_PA = torch.mean(log_PA * QA_dist)

        loss = loss_QA - loss_PA

        return loss

def seperate_nextqa_input_modalities(
    features, i3d_rgb_interval, i3d_flow_interval, question_intervals,
    vis_state_vector_idx, question_state_vector_idx,
    attention_values=None):
    """ We separate the multimodal input hidden states. The state token embeddings are left out (+1 while indexing)

    Args:
        features (_type_): _description_
        i3d_rgb_interval (_type_): _description_
        i3d_flow_interval (_type_): _description_
        sam_interval (_type_): _description_
        audio_interval (_type_): _description_
        history_intervals (_type_): _description_
        question_intervals (_type_): _description_

    Returns:
        _type_: _description_
    """
    features_copy = features.clone() # .detach()
    i3d_rgb_hidden = features_copy[:, i3d_rgb_interval[0]+1:i3d_rgb_interval[1], :]
    i3d_flow_hidden = features_copy[:, i3d_flow_interval[0]+1:i3d_flow_interval[1], :]
    
    question_hidden = []
    features_split = torch.split(features_copy, 1, dim=0)
    for ques_inter, feat in zip(question_intervals, features_split):
        ques_idx = torch.arange(ques_inter[0]+1, ques_inter[1]).unsqueeze(0).unsqueeze(-1).repeat(1, 1, feat.size(-1)).to(feat.device)
        question_hidden.append(torch.gather(feat, 1, ques_idx))
    
    if attention_values is None:
        i3d_rgb_att = None
        i3d_flow_att = None
        question_att = None
    else:
        attention_values = attention_values.mean(1)
        i3d_rgb_att = attention_values[:, vis_state_vector_idx[0], vis_state_vector_idx[0]+1:vis_state_vector_idx[1]]
        i3d_flow_att = attention_values[:, vis_state_vector_idx[1], vis_state_vector_idx[1]+1:question_state_vector_idx[0]]
        question_att = [attention_values[i, question_state_vector_idx[i], question_intervals[i][0] + 1: question_intervals[i][1]] for i in range(len(question_state_vector_idx))]

    features_list = [i3d_rgb_hidden, i3d_flow_hidden, question_hidden]
    att = [i3d_rgb_att, i3d_flow_att, question_att]
    
    return features_list, att


def seperate_input_modalities(
    features, i3d_rgb_interval, i3d_flow_interval, sam_interval, audio_interval, history_intervals, question_intervals,
    vis_state_vector_idx, history_state_vector_idx, question_state_vector_idx,
    attention_values=None):
    """ We separate the multimodal input hidden states. The state token embeddings are left out (+1 while indexing)

    Args:
        features (_type_): _description_
        i3d_rgb_interval (_type_): _description_
        i3d_flow_interval (_type_): _description_
        sam_interval (_type_): _description_
        audio_interval (_type_): _description_
        history_intervals (_type_): _description_
        question_intervals (_type_): _description_

    Returns:
        _type_: _description_
    """
    features_copy = features.clone() # .detach()
    i3d_rgb_hidden = features_copy[:, i3d_rgb_interval[0]+1:i3d_rgb_interval[1], :]
    i3d_flow_hidden = features_copy[:, i3d_flow_interval[0]+1:i3d_flow_interval[1], :]
    sam_hidden = features_copy[:, sam_interval[0]+1:sam_interval[1], :]
    audio_hidden = features_copy[:, audio_interval[0]+1:audio_interval[1], :]
    
    history_hidden = []
    question_hidden = []
    features_split = torch.split(features_copy, 1, dim=0)
    for hist_inter, ques_inter, feat in zip(history_intervals, question_intervals, features_split):
        hist_idx = torch.arange(hist_inter[0]+1, hist_inter[1]).unsqueeze(0).unsqueeze(-1).repeat(1, 1, feat.size(-1)).to(feat.device)
        history_hidden.append(torch.gather(feat, 1, hist_idx))

        ques_idx = torch.arange(ques_inter[0]+1, ques_inter[1]).unsqueeze(0).unsqueeze(-1).repeat(1, 1, feat.size(-1)).to(feat.device)
        question_hidden.append(torch.gather(feat, 1, ques_idx))
    
    if attention_values is None:
        i3d_rgb_att = None
        i3d_flow_att = None
        sam_att = None
        audio_att = None
        history_att = None
        question_att = None
    else:
        attention_values = attention_values.mean(1)
        i3d_rgb_att = attention_values[:, vis_state_vector_idx[0], vis_state_vector_idx[0]+1:vis_state_vector_idx[1]]
        i3d_flow_att = attention_values[:, vis_state_vector_idx[1], vis_state_vector_idx[1]+1:vis_state_vector_idx[2]]
        sam_att = attention_values[:, vis_state_vector_idx[2], vis_state_vector_idx[2]+1:vis_state_vector_idx[3]]
        audio_att = attention_values[:, vis_state_vector_idx[3], vis_state_vector_idx[3]+1:history_state_vector_idx[0] - 1]
        history_att = [attention_values[i, history_state_vector_idx[i], history_intervals[i][0] + 1 : history_intervals[i][1]] for i in range(len(history_state_vector_idx))]
        question_att = [attention_values[i, question_state_vector_idx[i], question_intervals[i][0] + 1: question_intervals[i][1]] for i in range(len(question_state_vector_idx))]

    features_list = [i3d_rgb_hidden, i3d_flow_hidden, sam_hidden, audio_hidden, history_hidden, question_hidden]
    att = [i3d_rgb_att, i3d_flow_att, sam_att, audio_att, history_att, question_att]
    
    return features_list, att


def get_knn_graph(features, num_nn, device):
    features = features.permute((1, 2, 0))
    cosine_sim_pairwise = F.cosine_similarity(features, features.unsqueeze(1), dim=-2)
    cosine_sim_pairwise = cosine_sim_pairwise.permute((2, 0, 1))
    num_nn = min(num_nn, cosine_sim_pairwise.size(-1))
    adj_mat = torch.zeros_like(cosine_sim_pairwise).to(device)
    _, to_keep = torch.topk(cosine_sim_pairwise, num_nn, dim=-1, sorted=False)
    adj_mat = adj_mat.scatter(-1, to_keep, torch.ones_like(adj_mat).to(device))
    return adj_mat


def track_features_vis(features, att, top_k, device, node_idx=None):
    top_k = min(features.size(1), top_k)
    if att is None:
        node_idx = torch.randint(low=0, high=features.size(1), size=(features.size(0), top_k))
    else:
        _, node_idx = torch.topk(att, top_k, dim=-1, sorted=False)

    node_idx = node_idx.unsqueeze(-1).repeat(1, 1, features.size(-1)).to(device)

    selected_features = torch.gather(features, 1, node_idx)

    return selected_features, node_idx


def track_features_text(features, att, top_k, device, node_idx=None):
    hidden_dim = features[0].size(-1)
    min_len = min([feat.size(1) for feat in features])
    top_k = min(min_len, top_k)
    if att is None:
        node_idx = [torch.randint(low=0, high=feat.size(1), size=(feat.size(0), top_k)) for feat in features]
    else:
        node_idx = [torch.topk(a, top_k, dim=-1, sorted=False)[-1] for a in att]

    node_idx = [idx.unsqueeze(-1).repeat(1, 1, hidden_dim).to(device) for idx in node_idx]

    selected_features = [torch.gather(feat, 1, idx) for feat, idx in zip(features, node_idx)]
    selected_features = torch.cat(selected_features, dim=0)

    return selected_features, node_idx


def diag_tensor(tensors):
    device = tensors[0].device
    n = sum([t.size(-1) for t in tensors])
    bsz = tensors[0].size(0)
    diag_tensor = torch.zeros((bsz, n, n)).float().to(device)
    delimiter = 0
    delimiters = [0]
    for t in tensors:
        diag_tensor[:, delimiter:delimiter+t.size(-1), delimiter:delimiter+t.size(-1)] = t
        delimiter += t.size(-1)
        delimiters.append(delimiter)

    return diag_tensor, delimiters


def embed_graphs(features, delimiters):
    state_vectors = []
    for i in range(len(delimiters) - 1):
        state_vectors.append(features[:, delimiters[i]:delimiters[i+1], :].mean(dim=1))
    return state_vectors


class AVSDEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    QAs_local = None
    PAs_local = None
    QA_global = None
    PA_global = None
    state_vectors = None


class AVSDSeq2SeqModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    QAs_local = None
    PAs_local = None
    QA_global = None
    PA_global = None
    state_vectors = None


class AVSDSeq2SeqLMOutput(ModelOutput):

    gen_loss: Optional[torch.FloatTensor] = None
    elbo_loss_global: Optional[torch.FloatTensor] = None
    elbo_loss_local: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_QAs_local = None
    encoder_PAs_local = None
    encoder_QA_global = None
    encoder_PA_global = None
    encoder_state_vectors = None
