from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
from .highway import Highway
from .positional_encoding import PositionalEncoding


class LBLbiLm(nn.Module):
  def __init__(self, config, use_cuda=False):
    super(LBLbiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']
    num_layers = config['encoder']['n_layers']

    if use_cuda:
       self.left_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
       self.left_weights = torch.autograd.Variable(torch.cuda.FloatTensor(width))
       self.right_weights = torch.autograd.Variable(torch.cuda.FloatTensor(width))
    else:
       self.left_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))
       self.left_weights = torch.autograd.Variable(torch.FloatTensor(width))
       self.right_weights = torch.autograd.Variable(torch.FloatTensor(width))

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
      left_inp = self.left_weights.repeat(batch_size, 1).view(batch_size, 1, -1).bmm(
        new_inputs.narrow(1, start, self.width)).view(batch_size, -1)
      right_inp = self.right_weights.repeat(batch_size, 1).view(batch_size, 1, -1).bmm(
        new_inputs.narrow(1, end + 1, self.width)).view(batch_size, -1)

      left_out = self.left_highway(left_inp)
      right_out = self.right_highway(right_inp)

      out = torch.cat([left_out, right_out], dim=1)
      outs.append(out)

    return torch.stack(outs, dim=1)


class LBLwithPositionBiLm(nn.Module):
  def __init__(self, config, use_cuda=False):
    super(LBLwithPositionBiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']
    num_layers = config['encoder']['n_layers']

    if use_cuda:
       self.left_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
       self.left_weights = torch.autograd.Variable(torch.cuda.FloatTensor(width))
       self.right_weights = torch.autograd.Variable(torch.cuda.FloatTensor(width))
    else:
       self.left_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))
       self.left_weights = torch.autograd.Variable(torch.FloatTensor(width))
       self.right_weights = torch.autograd.Variable(torch.FloatTensor(width))

    self.position = PositionalEncoding(config['encoder']['projection_dim'], self.config['dropout'])

    self.left_highway = Highway(hidden_size, num_layers)
    self.right_highway = Highway(hidden_size, num_layers)

    self.input_size = input_size
    self.num_layers = num_layers
    self.width = width

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    new_inputs = torch.cat([self.left_padding.repeat(batch_size, 1, 1),
                            self.position(inputs),
                            self.right_padding.repeat(batch_size, 1, 1)], dim=1)

    outs = []
    for start in range(sequence_len):
      end = start + self.width
      left_inp = self.left_weights.repeat(batch_size, 1).view(batch_size, 1, -1).bmm(
        new_inputs.narrow(1, start, self.width)).view(batch_size, -1)
      right_inp = self.right_weights.repeat(batch_size, 1).view(batch_size, 1, -1).bmm(
        new_inputs.narrow(1, end + 1, self.width)).view(batch_size, -1)

      left_out = self.left_highway(left_inp)
      right_out = self.right_highway(right_inp)

      out = torch.cat([left_out, right_out], dim=1)
      outs.append(out)

    return torch.stack(outs, dim=1)
