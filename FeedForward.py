import torch
from torch import nn

class FeedForwardBlock(nn.Module):

  def __init__(self, d_model: int, d_ff: int, dropout:float) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

  def forward(self, x):
    # (Batch. Seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))