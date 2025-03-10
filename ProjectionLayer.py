import torch
from torch import nn

class ProjectionLayer(nn.Module):

  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.proj = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
    return torch.log_softmax(self.proj(x), dim = -1)