import torch
from torch import nn
import torch.nn.functional as F


class Cross_MultiAttention(nn.Module):
    def __init__(self, Q_dim, KV_dim, emb_dim, num_heads):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.Wq = nn.Linear(Q_dim, emb_dim)
        self.Wk = nn.Linear(KV_dim, emb_dim)
        self.Wv = nn.Linear(KV_dim, emb_dim)

        # self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, q_candidate, kv_candidate, pad_mask=None):
        """
        :param q_candidate: [batch_size, c, h, w]
        :param kv_candidate: [batch_size, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        """
        batch_size = q_candidate.shape[0]

        Q = self.Wq(q_candidate)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(kv_candidate)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(kv_candidate)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1).to(torch.bool)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)  # [batch_size, h*w, emb_dim]

        # out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)  # [batch_size, c, h, w]
        # out = self.proj_out(out)  # [batch_size, c, h, w]

        return out, att_weights