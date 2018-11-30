#!/usr/bin/env python
import torch


class PositionwiseFeedForward(torch.nn.Module):
  """Implements FFN equation."""

  def __init__(self, d_model, d_ff, dropout=0.1):
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = torch.nn.Linear(d_model, d_ff)
    self.w_2 = torch.nn.Linear(d_ff, d_model)
    self.dropout = torch.nn.Dropout(dropout)

  def forward(self, x):
    return self.w_2(self.dropout(torch.nn.functional.relu(self.w_1(x))))
