import torch
from torch import nn
import math

class InputEmbedding(nn.Module):
  def __init__(self, dimension_model: int, vocab_size: int):
    super().__init__()
    self.dimension_model = dimension_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size, dimension_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.dimension_model)

class PositionalEmbedding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    # Create a matrix of shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)

    # Create a matrix of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    # Apply Sin to even positions
    pe[:, 0::2] = torch.sin(position * div_term)

    # Apply Cos to odd positions
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0) # (1, seq_len, d_model)

    self.register_buffer('pe', pe)

  def forward(self, x):
    # x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False)
    x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
    return self.dropout(x)