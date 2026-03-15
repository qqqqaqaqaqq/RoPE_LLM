import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, **kwargs):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # head 합친 후 출력

        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)  # attention 후
        self.norm2 = nn.LayerNorm(d_model)  # FFN 후

        self.dropout = nn.Dropout(dropout)

    # RoPE 이론
    # Q, K 회전
    def apply_rope(self, x, T, device):
        # 캐싱 적용 메모리 절약
        if not hasattr(self, '_rope_cache') or self._rope_cache[0] != T:
            d_k = x.shape[-1]
            theta = 1.0 / (10000 ** (torch.arange(0, d_k, 2, device=device) / d_k))
            positions = torch.arange(T, device=device)
            angles = positions.unsqueeze(1) * theta.unsqueeze(0)
            self._rope_cache = (T, angles.cos(), angles.sin())
        _, cos, sin = self._rope_cache

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x:torch.Tensor, src_mask=None, src_key_padding_mask=None):
        B, T, D = x.shape
        N = self.nhead
        d_k = D // N

        Q = self.Wq(x).view(B, T, N, d_k).transpose(1, 2)
        K = self.Wk(x).view(B, T, N, d_k).transpose(1, 2)
        V = self.Wv(x).view(B, T, N, d_k).transpose(1, 2)

        Q = self.apply_rope(Q, T, x.device)
        K = self.apply_rope(K, T, x.device)

        # pad_mask 캐싱
        if src_key_padding_mask is not None:
            pad_mask = src_key_padding_mask.float().masked_fill(src_key_padding_mask, float('-inf'))
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2).to(dtype=Q.dtype)
        else:
            pad_mask = None

        out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=pad_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )

        # contiguous() 제거 - view → reshape
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)
        x = self.norm1(x + self.dropout(out))
        ffn = self.ff2(F.gelu(self.ff1(x)))
        x = self.norm2(x + self.dropout(ffn))
        return x