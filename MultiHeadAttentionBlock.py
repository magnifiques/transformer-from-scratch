import torch
from torch import nn
import math

class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, d_model: int, heads: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.heads = heads
    assert d_model % heads == 0, "d_model is not divisible by heads"

    self.d_k = self.d_model // self.heads

    # Weights
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    self.w_o = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)

  @staticmethod
  def attention(query, key, value, MASK, dropout: nn.Dropout):

    d_k = query.shape[-1]

    # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

    if MASK is not None:
      attention_scores.masked_fill_(MASK == 0, -1e9)
    attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)

    if dropout is not None:
      attention_scores = dropout(attention_scores)

    return (attention_scores @ value), attention_scores

  def forward(self, q, k, v, MASK):
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)

    # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.heads, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.heads, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.heads, self.d_k).transpose(1, 2)

    x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, MASK, self.dropout)

    # (batch, heads, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.d_k)

    return self.w_o(x)
