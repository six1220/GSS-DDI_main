import torch
from torch import nn
import torch.nn.functional as F
import math
import copy

class AMDE(nn.Module):

    def __init__(self, args):
        super(AMDE, self).__init__()
        self.args = args
        self.max_d = args.AMDE_max_d
        self.vocab_size = args.AMDE_vocab_size
        self.n_layer = args.AMDE_n_layer
        self.emb_size = args.AMDE_emb_size
        self.dropout_rate = args.AMDE_dropout_rate
        self.out_dim = args.AMDE_out_dim
        # 添加基因表达数据维度
        self.use_gene_expr = hasattr(args, 'use_gene_expr') and args.use_gene_expr
        self.gene_expr_dim = getattr(args, 'gene_expr_out_dim', 978) if self.use_gene_expr else 0

        # encoder
        self.hidden_size = args.AMDE_hidden_size
        self.intermediate_size = args.AMDE_intermediate_size
        self.num_attention_heads = args.AMDE_attention_head
        self.attention_probs_dropout_prob = args.AMDE_attention_dropout
        self.hidden_dropout_prob = args.AMDE_hidden_dropout

        # specialized embedding with positional one
        self.emb = Embeddings(self.vocab_size, self.emb_size, self.max_d, self.dropout_rate)
        self.d_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        self.p_encoder = Encoder_MultipleLayers(self.n_layer, self.hidden_size, self.intermediate_size,
                                                self.num_attention_heads, self.attention_probs_dropout_prob,
                                                self.hidden_dropout_prob)
        # dencoder
        self.decoder_trans_mpnn_cat = nn.Sequential(
            nn.Linear(406, 64),
            nn.ReLU(True),

            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(True),

            # output layer
            nn.Linear(32, 1)
        )
        self.decoder_trans_mpnn_sum = nn.Sequential(
            nn.Linear(203, 32),
            nn.ReLU(True),
            nn.BatchNorm1d(32),
            # output layer
            nn.Linear(32, 1)
        )

        # 更新解码器输入维度，包含基因表达数据
        self.decoder_1 = nn.Sequential(
             nn.Linear(args.AMDE_max_d*384 + self.gene_expr_dim, 512),
             nn.ReLU(True),
             nn.BatchNorm1d(512),

             nn.Linear(512, self.out_dim)
        )

    def forward(self, de_1, de_2, mask_1, mask_2, d1_expr=None, d2_expr=None):

        batch_size = de_1.size(0)

        #Sequence encoder
        ex_d_mask = de_1.unsqueeze(1).unsqueeze(2)  # 16个分子的smile序列每个50个片段
        ex_p_mask = de_2.unsqueeze(1).unsqueeze(2)
        ex_d_mask = (1.0 - ex_d_mask) * -10000.0  # 把不存在的片段编码为负无穷
        ex_p_mask = (1.0 - ex_p_mask) * -10000.0

        d_emb = self.emb(de_1)  # batch_size x seq_length x embed_size
        # print(d_emb.is_cuda,'d_emb')
        p_emb = self.emb(de_2)

        # set output_all_encoded_layers be false, to obtain the last layer hidden states only...
        d_encoded_layers = self.d_encoder(d_emb.float(), ex_d_mask.float())  # 自注意力+残差
        p_encoded_layers = self.d_encoder(p_emb.float(), ex_p_mask.float())
        d1_trans_fts = d_encoded_layers.view(batch_size, -1)  # 相当于把50个片段编码concat到一起成为分子emb
        d2_trans_fts = p_encoded_layers.view(batch_size, -1)

        # 拼接基因表达数据
        if self.use_gene_expr and d1_expr is not None and d2_expr is not None:
            # 确保基因表达特征在正确的设备上
            if d1_expr.device != d1_trans_fts.device:
                d1_expr = d1_expr.to(d1_trans_fts.device)
            if d2_expr.device != d2_trans_fts.device:
                d2_expr = d2_expr.to(d2_trans_fts.device)
            
            # 拼接基因表达数据
            d1_trans_fts = torch.cat([d1_trans_fts, d1_expr], dim=1)
            d2_trans_fts = torch.cat([d2_trans_fts, d2_expr], dim=1)

        d1_seq_fts_layer1 = self.decoder_1(d1_trans_fts)  # 将分子emb压缩成128维 得到1D smile的输出
        d2_seq_fts_layer1 = self.decoder_1(d2_trans_fts)
        return d1_seq_fts_layer1, d2_seq_fts_layer1


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)  # (x - u)^2 方差 求每个分子的方差
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # 依照正态分布进行标准化
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Construct the embeddings from protein/target, position embeddings.
    """

    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        b = torch.LongTensor(1, 2).to(input_ids.device)
        # print(input_ids.is_cuda)
        input_ids = input_ids.type_as(b)
        # print(input_ids.is_cuda)

        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # 【1.。。50】

        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        # print(words_embeddings.is_cuda)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)  # +注意力
        attention_output = self.output(self_output, input_tensor)  # +残差
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)  # 给向量加了残差和注意力机制
        intermediate_output = self.intermediate(attention_output)  # 给向量拉长
        layer_output = self.output(intermediate_output, attention_output)  # 把向量带着残差压缩回去

        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                 hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob,
                        hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)

        return hidden_states