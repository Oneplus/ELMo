from __future__ import absolute_import
from __future__ import unicode_literals
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .highway import Highway


class MLPbiLm(nn.Module):
  def __init__(self, config, use_cuda=False):
    super(MLPbiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim'] * width
    hidden_size = config['encoder']['projection_dim']
    num_layers = config['encoder']['n_layers']

    if use_cuda:
       self.left_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
    else:
       self.left_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))

    self.left_project = nn.Linear(input_size, hidden_size)
    self.right_project = nn.Linear(input_size, hidden_size)

    self.left_highway = Highway(hidden_size, num_layers)
    self.right_highway = Highway(hidden_size, num_layers)

    self.input_size = input_size
    self.num_layers = num_layers
    self.width = width

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    new_inputs = torch.cat([self.left_padding.repeat(batch_size, 1, 1),
                            inputs,
                            self.right_padding.repeat(batch_size, 1, 1)], dim=1)

    outs = []
    for start in range(sequence_len):
      end = start + self.width
      left_inp = new_inputs.narrow(1, start, self.width).contiguous().view(batch_size, -1)
      right_inp = new_inputs.narrow(1, end + 1, self.width).contiguous().view(batch_size, -1)

      left_out = self.left_highway(
        F.dropout(self.left_project(left_inp), self.config['dropout'], self.training))
      right_out = self.right_highway(
        F.dropout(self.right_project(right_inp), self.config['dropout'], self.training))

      out = torch.cat([left_out, right_out], dim=1)
      outs.append(out)

    return torch.stack(outs, dim=1)
