#!/usr/bin/env python
# source: http://nlp.seas.harvard.edu/2018/04/03/attention.html
from __future__ import print_function
from __future__ import division
import torch
import math


class PositionalEncoding(torch.nn.Module):
  """Implement the PE function."""

  def __init__(self, d_model, dropout, max_len=5000):
    super(PositionalEncoding, self).__init__()
    self.dropout = torch.nn.Dropout(p=dropout)

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], requires_grad=False)
    return self.dropout(x)
