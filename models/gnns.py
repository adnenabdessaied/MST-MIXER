import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGATConv, DenseGCNConv, DenseSAGEConv
from torch.nn.parameter import Parameter
from typing import Optional, Tuple
from .utils import get_knn_graph
import torch_sparse


class BartAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        # `past_key_value[0].shape[2] == key_value_states.shape[1]`
        # is checking that the `sequence_length` of the `past_key_value` is the same as
        # the provided `key_value_states` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value is not None
            and past_key_value[0].shape[2] == key_value_states.shape[1]
        ):
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class MLPModule(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers=3, dropout=0.3, use_non_linear=False, use_batch_norm=False):
        super(MLPModule, self).__init__()
        
        self.use_batch_norm = use_batch_norm
        self.dropout = dropout 

        self.fcs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        if num_layers == 1:
            self.fcs.append(nn.Linear(d_in, d_out))
        else:
            self.fcs.append(nn.Linear(d_in, d_hidden))
            self.batch_norms.append(nn.BatchNorm1d(d_hidden))
            for _ in range(num_layers - 2):
                self.fcs.append(nn.Linear(d_hidden, d_hidden))
                self.batch_norms.append(nn.BatchNorm1d(d_hidden))
            
            self.fcs.append(nn.Linear(d_hidden, d_out))

        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_non_linear=use_non_linear


    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, X):
        for fc, bn in zip(self.fcs[:-1], self.batch_norms):
            X = fc(X)
            X = self.act_fn(X)
            if self.use_batch_norm:
                if X.dim() > 2:
                    X = X.transpose(1, 2)
                X = bn(X)
                if X.dim() > 2:
                    X = X.transpose(1, 2)
            X = self.dropout(X)
        X = self.fcs[-1](X)
        return X


class GATModule(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers=3, dropout=0.3, concat=True, heads=2, use_non_linear=False, use_batch_norm=False):
        super(GATModule, self).__init__()
        self.gnns = nn.ModuleList()
        if concat:
            d_hidden = d_hidden // heads
            d_out = d_out // heads

        self.gnns.append(DenseGATConv(d_in, d_hidden, heads=heads, concat=concat, dropout=dropout))

        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm1d(d_hidden * heads if concat else d_hidden))

        for _ in range(num_layers - 2):
            self.gnns.append(DenseGATConv(
                d_hidden * heads if concat else d_hidden, d_hidden,
                heads=heads,
                concat=concat,
                dropout=dropout)
            )
            self.batch_norms.append(nn.BatchNorm1d(d_hidden * heads if concat else d_hidden))
        
        self.gnns.append(DenseGATConv(
            d_hidden * heads if concat else d_hidden, d_out,
            heads=heads,
            concat=concat,
            dropout=dropout)
        )

        self.dropout = nn.Dropout(dropout)
        self.non_linear = nn.GELU()
        self.use_batch_norm = use_batch_norm
        self.use_non_linear = use_non_linear

    def reset_parameters(self):
        for gnn in self.gnns:
            gnn.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, X, A):
        Z = self.dropout(X)
        for i in range(len(self.gnns) - 1):
            Z = self.gnns[i](Z, A)
            if self.use_batch_norm:
                Z = Z.transpose(1, 2)
                Z = self.batch_norms[i](Z)
                Z = Z.transpose(1, 2)
            if self.use_non_linear:
                Z = self.non_linear(Z)
            Z = self.dropout(Z)
        Z = self.gnns[-1](Z, A)
        return Z


class GCNModule(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers=3, dropout=0.3, use_non_linear=False, use_batch_norm=False):
        super(GCNModule, self).__init__()
        self.gnns = nn.ModuleList()

        self.gnns.append(DenseGCNConv(d_in, d_hidden))

        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm1d(d_hidden))

        for _ in range(num_layers - 2):
            self.gnns.append(DenseGCNConv(
                d_hidden, d_hidden)
            )
            self.batch_norms.append(nn.BatchNorm1d(d_hidden))
        
        self.gnns.append(DenseGCNConv(
            d_hidden, d_out)
        )

        self.dropout = nn.Dropout(dropout)
        self.non_linear = nn.GELU()
        self.use_batch_norm = use_batch_norm
        self.use_non_linear = use_non_linear

    def reset_parameters(self):
        for gnn in self.gnns:
            gnn.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, X, A):
        Z = self.dropout(X)
        for i in range(len(self.gnns) - 1):
            Z = self.gnns[i](Z, A)
            if self.use_batch_norm:
                Z = Z.transpose(1, 2)
                Z = self.batch_norms[i](Z)
                Z = Z.transpose(1, 2)
            if self.use_non_linear:
                Z = self.non_linear(Z)
            Z = self.dropout(Z)
        Z = self.gnns[-1](Z, A)
        return Z


class SAGEModule(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, num_layers=3, dropout=0.3, use_non_linear=False, use_batch_norm=False):
        super(SAGEModule, self).__init__()
        self.gnns = nn.ModuleList()

        self.gnns.append(DenseSAGEConv(d_in, d_hidden))

        self.batch_norms = nn.ModuleList()
        self.batch_norms.append(nn.BatchNorm1d(d_hidden))

        for _ in range(num_layers - 2):
            self.gnns.append(DenseSAGEConv(
                d_hidden, d_hidden)
            )
            self.batch_norms.append(nn.BatchNorm1d(d_hidden))
        
        self.gnns.append(DenseSAGEConv(
            d_hidden, d_out)
        )

        self.dropout = nn.Dropout(dropout)
        self.non_linear = nn.GELU()
        self.use_batch_norm = use_batch_norm
        self.use_non_linear = use_non_linear

    def reset_parameters(self):
        for gnn in self.gnns:
            gnn.reset_parameters()
        for batch_norm in self.batch_norms:
            batch_norm.reset_parameters()

    def forward(self, X, A):
        Z = self.dropout(X)
        for i in range(len(self.gnns) - 1):
            Z = self.gnns[i](Z, A)
            if self.use_batch_norm:
                Z = Z.transpose(1, 2)
                Z = self.batch_norms[i](Z)
                Z = Z.transpose(1, 2)
            if self.use_non_linear:
                Z = self.non_linear(Z)
            Z = self.dropout(Z)
        Z = self.gnns[-1](Z, A)
        return Z


class GlobalGraphLearner(nn.Module):
    def __init__(self, d_in, num_heads, random=False):
        super(GlobalGraphLearner, self).__init__()
        self.random = random
        if not self.random:
            w = torch.Tensor(num_heads, d_in)
            self.w = Parameter(nn.init.xavier_uniform_(w), requires_grad=True)

    def reset_parameters(self):
        if not self.random:
            self.w = Parameter(nn.init.xavier_uniform_(self.w))

    def forward(self, Z):
        if self.random:
            att_global = torch.randn((Z.size(0), Z.size(1), Z.size(1))).to(Z.device)
        else:
            w_expanded = self.w.unsqueeze(1).unsqueeze(1)
            Z = Z.unsqueeze(0) * w_expanded
            Z = F.normalize(Z, p=2, dim=-1)
            att_global = torch.matmul(Z, Z.transpose(-1, -2)).mean(0)
        mask_global = (att_global > 0).detach().float()
        att_global = att_global * mask_global

        return att_global


class DenseAPPNP(nn.Module):
    def __init__(self, K, alpha):
        super().__init__()
        self.K = K
        self.alpha = alpha

    def forward(self, x, adj_t):
        h = x
        for _ in range(self.K):
            if adj_t.is_sparse:
                x = torch_sparse.spmm(adj_t, x)
            else:
                x = torch.matmul(adj_t, x)
            x = x * (1 - self.alpha)
            x += self.alpha * h
        x /= self.K
        return x


class Dense_APPNP_Net(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.5, K=10, alpha=.1):
        super(Dense_APPNP_Net, self).__init__()
        self.lin1 = nn.Linear(d_in, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_out)
        self.prop1 = DenseAPPNP(K, alpha)
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj_t):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, adj_t)
        return x


class MMGraphLearner(nn.Module):
    def __init__(self, d_in, num_heads, random=False):
        super(MMGraphLearner, self).__init__()
        self.random = random
        if not self.random:
            w = torch.Tensor(num_heads, d_in)
            self.w = Parameter(nn.init.xavier_uniform_(w), requires_grad=True)

            self.fc = nn.Linear(d_in, d_in)

    def reset_parameters(self):
        if not self.random:
            self.fc.reset_parameters()
            self.w = Parameter(nn.init.xavier_uniform_(self.w), requires_grad=True)

    def forward(self, features):
        if self.random:
            att = torch.randn((features.size(0), features.size(1), features.size(1))).to(features.device)
        else:
            features = self.fc(features)
            w_expanded = self.w.unsqueeze(1).unsqueeze(1)
            features = features.unsqueeze(0) * w_expanded
            features = F.normalize(features, p=2, dim=-1)
            att = torch.matmul(features, features.transpose(-1, -2)).mean(0)
        mask = (att > 0).detach().float()
        att = att * mask
        return att
      

class QNetLocal(nn.Module):
    def __init__(self, config):
        super(QNetLocal, self).__init__()
        self.config=config

        self.mm_gnn_modules = nn.ModuleList()
        self.mm_graph_learners_1 = nn.ModuleList()
        self.mm_graph_learners_2 = nn.ModuleList()

        for _ in range(self.config.num_modalities):
            if self.config.gnn_type == 'gat':
                self.mm_gnn_modules.append(GATModule(
                    self.config.d_model, self.config.d_model, self.config.d_model,
                    num_layers=self.config.num_local_gnn_layers,
                    heads=self.config.num_local_gnn_heads,
                    dropout=self.config.local_gnn_dropout,
                    concat=self.config.local_gnn_concat,
                    use_batch_norm=self.config.use_local_gnn_bn,
                    use_non_linear=self.config.use_non_linear
                    )
                )
            elif self.config.gnn_type == 'appnp':
                self.mm_gnn_modules.append(Dense_APPNP_Net(
                    self.config.d_model, self.config.d_model, self.config.d_model, dropout=self.config.local_gnn_dropout,
                    K=self.config.gnn_K, alpha=self.config.gnn_alpha
                    )
                )
            elif self.config.gnn_type == 'gcn':
                self.mm_gnn_modules.append(GCNModule(
                    self.config.d_model, self.config.d_model, self.config.d_model,
                    num_layers=self.config.num_local_gnn_layers,
                    dropout=self.config.local_gnn_dropout,
                    use_batch_norm=self.config.use_local_gnn_bn,
                    use_non_linear=self.config.use_non_linear
                )
            )
            elif self.config.gnn_type == 'sage':
                self.mm_gnn_modules.append(SAGEModule(
                    self.config.d_model, self.config.d_model, self.config.d_model,
                    num_layers=self.config.num_local_gnn_layers,
                    dropout=self.config.local_gnn_dropout,
                    use_batch_norm=self.config.use_local_gnn_bn,
                    use_non_linear=self.config.use_non_linear
                )
            )
            else:
                raise ValueError
            self.mm_graph_learners_1.append(MMGraphLearner(self.config.d_model, self.config.num_local_gr_learner_heads, random=self.config.use_random_graphs))
            self.mm_graph_learners_2.append(MMGraphLearner(self.config.d_model * 2, self.config.num_local_gr_learner_heads, random=self.config.use_random_graphs))

    def reset_parameters(self):
        for i in range(self.config.num_modalities):
            self.mm_gnn_modules[i].reset_parameters()
            self.mm_graph_learners_1[i].reset_parameters()
            self.mm_graph_learners_2[i].reset_parameters()

    def forward(self, features, A_tildes=None):
        mm_Xs = features#  []
        device = features[0].device

        if A_tildes is None:
            A_tildes = []
            for mm_X in mm_Xs:
                A_tildes.append(get_knn_graph(mm_X, self.config.num_nn, device))

        ################# Multi-modal graph learner (upper branch) #################
        A_primes = []
        for i, mm_X in enumerate(mm_Xs):  # iterate over the modalities
            A_primes.append(self.mm_graph_learners_1[i](mm_X))

        # Linear combination of A_primes with A_tildes
        A_primes = [(1 - self.config.init_adj_ratio) * A_prime + self.config.init_adj_ratio * A_tilde for A_prime, A_tilde in zip(A_primes, A_tildes)]

        ################# Multi-modal gnn (upper branch) #################
        Z_primes = []
        for i, (mm_X, A_prime) in enumerate(zip(mm_Xs, A_primes)):
            Z_primes.append(self.mm_gnn_modules[i](mm_X, A_prime))

        ################# Multi-modal gnn (lower branch) #################
        Z_double_primes = []
        for i, (mm_X, A_tilde) in enumerate(zip(mm_Xs, A_tildes)):
            Z_double_primes.append(self.mm_gnn_modules[i](mm_X, A_tilde))

        Z_concats = [torch.cat([Z_1, Z_2], dim=-1) for Z_1, Z_2 in zip(Z_primes, Z_double_primes)]
                
        ################# Multi-modal graph learner (lower branch) #################
        A_double_primes = []
        for i, Z_concat in enumerate(Z_concats):
            A_double_primes.append(self.mm_graph_learners_2[i](Z_concat))

        A_double_primes = [(1 - self.config.init_adj_ratio) * A_double_prime + self.config.init_adj_ratio * A_tilde for A_double_prime, A_tilde in zip(A_double_primes, A_tildes)]

        As = [(1 - self.config.adj_ratio) * A_prime + self.config.adj_ratio * A_double_prime for A_prime, A_double_prime in zip(A_primes, A_double_primes)]

        ################## Average across all multimodal inputs ##################

        Zs = [0.5 * Z1 + 0.5 * Z2 for Z1, Z2 in zip(Z_primes, Z_double_primes)]
        return As, Zs


class PNetLocal(nn.Module):
    def __init__(self, config):
        super(PNetLocal, self).__init__()
        self.config = config
        self.mm_gnn_modules = nn.ModuleList()
        self.mm_mlp_modules = nn.ModuleList()

        self.mm_graph_learners_1 = nn.ModuleList()
        self.mm_graph_learners_2 = nn.ModuleList()

        for _ in range(self.config.num_modalities):
            if self.config.gnn_type == 'gat':
                self.mm_gnn_modules.append(GATModule(
                    self.config.d_model, self.config.d_model, self.config.d_model,
                    num_layers=self.config.num_local_gnn_layers,
                    heads=self.config.num_local_gnn_heads,
                    dropout=self.config.local_gnn_dropout,
                    concat=self.config.local_gnn_concat,
                    use_batch_norm=self.config.use_local_gnn_bn,
                    use_non_linear=self.config.use_non_linear
                    )
                )
            elif self.config.gnn_type == 'appnp':
                self.mm_gnn_modules.append(Dense_APPNP_Net(
                    self.config.d_model, self.config.d_model, self.config.d_model, dropout=self.config.local_gnn_dropout,
                    K=self.config.gnn_K, alpha=self.config.gnn_alpha
                    )
                )
            elif self.config.gnn_type == 'gcn':
                self.mm_gnn_modules.append(GCNModule(
                    self.config.d_model, self.config.d_model, self.config.d_model,
                    num_layers=self.config.num_local_gnn_layers,
                    dropout=self.config.local_gnn_dropout,
                    use_batch_norm=self.config.use_local_gnn_bn,
                    use_non_linear=self.config.use_non_linear
                )
            )
            elif self.config.gnn_type == 'sage':
                self.mm_gnn_modules.append(SAGEModule(
                    self.config.d_model, self.config.d_model, self.config.d_model,
                    num_layers=self.config.num_local_gnn_layers,
                    dropout=self.config.local_gnn_dropout,
                    use_batch_norm=self.config.use_local_gnn_bn,
                    use_non_linear=self.config.use_non_linear
                )
            )
            else:
                raise ValueError
            self.mm_mlp_modules.append(MLPModule(
                self.config.d_model, self.config.d_model, self.config.d_model,
                num_layers=self.config.num_local_fc_layers,
                dropout=self.config.local_fc_dropout,
                use_batch_norm=self.config.use_local_fc_bn,
                use_non_linear=self.config.use_non_linear
            ))

            self.mm_graph_learners_1.append(MMGraphLearner(self.config.d_model, self.config.num_local_gr_learner_heads, random=self.config.use_random_graphs))
            self.mm_graph_learners_2.append(MMGraphLearner(self.config.d_model * 2, self.config.num_local_gr_learner_heads, random=self.config.use_random_graphs))

    def reset_parameters(self):
        for i in range(self.config.num_modalities):
            self.mm_gnn_modules[i].reset_parameters()
            self.mm_mlp_modules[i].reset_parameters()
            self.mm_graph_learners_1[i].reset_parameters()
            self.mm_graph_learners_2[i].reset_parameters()

    def forward(self, features):
        mm_Xs = features
       
        ################# Multi-modal graph learner (upper branch) #################
        A_primes = []
        for i, mm_X in enumerate(mm_Xs):  # iterate over the modalities
            A_primes.append(self.mm_graph_learners_1[i](mm_X))

        ################# Multi-modal gnn (upper branch) #################
        Z_primes = []
        for i, (mm_X, A_prime) in enumerate(zip(mm_Xs, A_primes)):
            Z_primes.append(self.mm_gnn_modules[i](mm_X, A_prime))

        ################# Multi-modal gnn (lower branch) #################
        Z_double_primes = []
        for i, mm_X, in enumerate(mm_Xs):
            Z_double_primes.append(self.mm_mlp_modules[i](mm_X))

        Z_concats = [torch.cat([Z_1, Z_2], dim=-1) for Z_1, Z_2 in zip(Z_primes, Z_double_primes)]
                
        ################# Multi-modal graph learner (lower branch) #################
        A_double_primes = []
        for i, Z_concat in enumerate(Z_concats):
            A_double_primes.append(self.mm_graph_learners_2[i](Z_concat))

        As = [(1 - self.config.adj_ratio) * A_prime + self.config.adj_ratio * A_double_prime for A_prime, A_double_prime in zip(A_primes, A_double_primes)]

        ################## Average across all multimodal inputs ##################

        Zs = [0.5 * Z1 + 0.5 * Z2 for Z1, Z2 in zip(Z_primes, Z_double_primes)]

        return As, Zs


class QNetGlobal(nn.Module):
    def __init__(self, config):
        super(QNetGlobal, self).__init__()
        self.config = config
        if self.config.gnn_type == 'gat':
            self.gnn = GATModule(
                self.config.d_model, self.config.d_model, self.config.d_model,
                num_layers=self.config.num_global_gnn_layers,
                heads=self.config.num_global_gnn_heads,
                dropout=self.config.global_gnn_dropout,
                concat=self.config.global_gnn_concat,
                use_batch_norm=self.config.use_global_gnn_bn,
                use_non_linear=self.config.use_non_linear
            )
        elif self.config.gnn_type == 'appnp':
            self.gnn = Dense_APPNP_Net(
                self.config.d_model, self.config.d_model, self.config.d_model, dropout=self.config.local_gnn_dropout,
                K=self.config.gnn_K, alpha=self.config.gnn_alpha
            )
        elif self.config.gnn_type == 'gcn':
            self.gnn = GCNModule(
                self.config.d_model, self.config.d_model, self.config.d_model,
                num_layers=self.config.num_global_gnn_layers,
                dropout=self.config.global_gnn_dropout,
                use_batch_norm=self.config.use_global_gnn_bn,
                use_non_linear=self.config.use_non_linear
            )

        elif self.config.gnn_type == 'sage':
            self.gnn = SAGEModule(
                self.config.d_model, self.config.d_model, self.config.d_model,
                num_layers=self.config.num_global_gnn_layers,
                dropout=self.config.global_gnn_dropout,
                use_batch_norm=self.config.use_global_gnn_bn,
                use_non_linear=self.config.use_non_linear
            )
        else:
            raise ValueError

        self.graph_learner_1 = GlobalGraphLearner(self.config.d_model, self.config.num_global_gr_learner_heads, self.config.use_random_graphs)
        self.graph_learner_2 = GlobalGraphLearner(self.config.d_model * 2, self.config.num_global_gr_learner_heads, self.config.use_random_graphs)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.graph_learner_1.reset_parameters()
        self.graph_learner_2.reset_parameters()
    
    def forward(self, Z, A):

        ################# Graph learner (upper branch) #################
        A_prime = self.graph_learner_1(Z)
        A_prime = (1-self.config.init_adj_ratio) * A_prime + self.config.init_adj_ratio * A

        ################# Gnn (upper branch) #################
        Z_prime = self.gnn(Z, A_prime)

        ################# Gnn (lower branch) #################
        Z_double_prime = self.gnn(Z, A)
        Z_concat = torch.cat([Z_prime, Z_double_prime], dim=-1)
                
        ################# Graph learner (lower branch) #################
        A_double_prime = self.graph_learner_2(Z_concat)
        A_double_prime = (1-self.config.init_adj_ratio) * A_double_prime + self.config.init_adj_ratio * A

        ################## Average across  branches ##################
        A_global = (1 - self.config.adj_ratio) * A_prime + self.config.adj_ratio * A_double_prime
        Z_global = 0.5 * Z_prime + 0.5 * Z_double_prime 
        return A_global, Z_global


class PNetGlobal(nn.Module):
    def __init__(self, config):
        super(PNetGlobal, self).__init__()
        self.config = config

        if self.config.gnn_type == 'gat':
            self.gnn = GATModule(
                self.config.d_model, self.config.d_model, self.config.d_model,
                num_layers=self.config.num_global_gnn_layers,
                heads=self.config.num_global_gnn_heads,
                dropout=self.config.global_gnn_dropout,
                concat=self.config.global_gnn_concat,
                use_batch_norm=self.config.use_global_gnn_bn,
                use_non_linear=self.config.use_non_linear
            )
        elif self.config.gnn_type == 'appnp':
            self.gnn = Dense_APPNP_Net(
                self.config.d_model, self.config.d_model, self.config.d_model, dropout=self.config.local_gnn_dropout,
                K=self.config.gnn_K, alpha=self.config.gnn_alpha
            )
        elif self.config.gnn_type == 'gcn':
            self.gnn = GCNModule(
                self.config.d_model, self.config.d_model, self.config.d_model,
                num_layers=self.config.num_global_gnn_layers,
                dropout=self.config.global_gnn_dropout,
                use_batch_norm=self.config.use_global_gnn_bn,
                use_non_linear=self.config.use_non_linear
            )
        elif self.config.gnn_type == 'sage':
            self.gnn = SAGEModule(
                self.config.d_model, self.config.d_model, self.config.d_model,
                num_layers=self.config.num_global_gnn_layers,
                dropout=self.config.global_gnn_dropout,
                use_batch_norm=self.config.use_global_gnn_bn,
                use_non_linear=self.config.use_non_linear
            )
        else:
            raise ValueError

        self.mlp = MLPModule(
            self.config.d_model, self.config.d_model, self.config.d_model,
            num_layers=self.config.num_global_fc_layers,
            dropout=self.config.global_fc_dropout,
            use_batch_norm=self.config.use_global_fc_bn,
            use_non_linear=self.config.use_non_linear

        )

        self.graph_learner_1 = GlobalGraphLearner(self.config.d_model, self.config.num_global_gr_learner_heads, random=self.config.use_random_graphs)
        self.graph_learner_2 = GlobalGraphLearner(self.config.d_model * 2, self.config.num_global_gr_learner_heads, random=self.config.use_random_graphs)

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.mlp.reset_parameters()
        self.graph_learner_1.reset_parameters()
        self.graph_learner_2.reset_parameters()

    def forward(self, Z, A):

        ################# Graph learner (upper branch) #################
        A_prime = self.graph_learner_1(Z)

        ################# Gnn (upper branch) #################
        Z_prime = self.gnn(Z, A_prime)

        ################# mlp (lower branch) #################
        Z_double_prime = self.mlp(Z)
        Z_concat = torch.cat([Z_prime, Z_double_prime], dim=-1)
                
        ################# Graph learner (lower branch) #################
        A_double_prime = self.graph_learner_2(Z_concat)
        # A_double_prime = (1-self.config.init_adj_ratio) * A_double_prime + self.config.init_adj_ratio * A

        ################## Average across braches ##################
        A_global = (1 - self.config.adj_ratio) * A_prime + self.config.adj_ratio * A_double_prime
        Z_global = 0.5 * Z_prime + 0.5 * Z_double_prime 
        return A_global, Z_global

