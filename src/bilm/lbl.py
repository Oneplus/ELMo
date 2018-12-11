from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import numpy as np
from modules.highway import Highway
from modules.positional_encoding import PositionalEncoding
from modules.sublayer_connection import SublayerConnection
from modules.positionwise_feedforward import PositionwiseFeedForward


class LBLHighwayBiLm(torch.nn.Module):
  def __init__(self, config, use_cuda=False):
    super(LBLHighwayBiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.use_position = config['encoder'].get('position', False)
    self.n_layers = n_layers = config['encoder']['n_layers']
    self.n_highway = n_highway = config['encoder']['n_highway']

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    self.width = width = config['encoder']['width']
    self.input_size = input_size = config['encoder']['projection_dim']
    self.hidden_size = hidden_size = config['encoder']['projection_dim']

    forward_paddings, backward_paddings = [], []
    forward_weights, backward_weights = [], []
    forward_blocks, backward_blocks = [], []

    for _ in range(n_layers):
      forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
      backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))

      forward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))
      backward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))

      forward_blocks.append(Highway(hidden_size, num_layers=n_highway))
      backward_blocks.append(Highway(hidden_size, num_layers=n_highway))

    self.forward_paddings = torch.nn.ParameterList(forward_paddings)
    self.backward_paddings = torch.nn.ParameterList(backward_paddings)
    self.forward_weights = torch.nn.ParameterList(forward_weights)
    self.backward_weights = torch.nn.ParameterList(backward_weights)
    self.forward_blocks = torch.nn.ModuleList(forward_blocks)
    self.backward_blocks = torch.nn.ModuleList(backward_blocks)

    if self.use_position:
      self.position = PositionalEncoding(hidden_size, self.config['dropout'])

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    all_layers_along_steps = []

    last_forward_inputs = inputs
    last_backward_inputs = inputs
    for i in range(self.n_layers):
      if self.use_position:
        last_forward_inputs = self.position(last_forward_inputs)
        last_backward_inputs = self.position(last_backward_inputs)

      padded_last_forward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                              last_forward_inputs,
                                              self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)
      padded_last_backward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                               last_backward_inputs,
                                               self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)

      forward_steps, backward_steps = [], []
      for start in range(sequence_len):
        end = start + self.width
        forward_input = padded_last_forward_inputs.narrow(1, start, self.width + 1)
        forward_output = forward_input.transpose(-2, -1).matmul(self.forward_weights[i])
        forward_output = self.forward_blocks[i](forward_output)

        backward_input = padded_last_backward_inputs.narrow(1, end, self.width + 1)
        backward_output = backward_input.transpose(-2, -1).matmul(self.backward_weights[i])
        backward_output = self.backward_blocks[i](backward_output)

        forward_steps.append(forward_output)
        backward_steps.append(backward_output)

      last_forward_inputs = torch.stack(forward_steps, dim=1)
      last_backward_inputs = torch.stack(backward_steps, dim=1)

      all_layers_along_steps.append(torch.cat([last_forward_inputs, last_backward_inputs], dim=-1))

    return torch.stack(all_layers_along_steps, dim=0)


class LBLResNetBiLm(torch.nn.Module):
  def __init__(self, config, use_cuda=False):
    super(LBLResNetBiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.use_position = config['encoder'].get('position', False)

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    self.width = width = config['encoder']['width']
    self.input_size = input_size = config['encoder']['projection_dim']
    self.hidden_size = hidden_size = config['encoder']['projection_dim']
    self.n_layers = n_layers = config['encoder']['n_layers']

    forward_paddings, backward_paddings = [], []
    forward_weights, backward_weights = [], []
    for _ in range(self.n_layers):
      forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
      backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
      forward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))
      backward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))

    self.forward_paddings = torch.nn.ParameterList(forward_paddings)
    self.backward_paddings = torch.nn.ParameterList(backward_paddings)
    self.forward_weights = torch.nn.Parameter(forward_weights)
    self.backward_weights = torch.nn.Parameter(backward_weights)

    if self.use_position:
      self.position = PositionalEncoding(hidden_size, self.config['dropout'])

    self.forward_linears = torch.nn.ModuleList(
      [PositionwiseFeedForward(hidden_size, hidden_size, self.config['dropout'])
       for _ in range(n_layers)])
    self.backward_linears = torch.nn.ModuleList(
      [PositionwiseFeedForward(hidden_size, hidden_size, self.config['dropout'])
       for _ in range(n_layers)])

    self.forward_blocks = torch.nn.ModuleList(
      [SublayerConnection(hidden_size, self.config['dropout']) for _ in range(n_layers)])
    self.backward_blocks = torch.nn.ModuleList(
      [SublayerConnection(hidden_size, self.config['dropout']) for _ in range(n_layers)])

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    all_layers_along_steps = []

    last_forward_inputs = inputs
    last_backward_inputs = inputs
    for i in range(self.n_layers):
      if self.use_position:
        last_forward_inputs = self.position(last_forward_inputs)
        last_backward_inputs = self.position(last_backward_inputs)

      padded_last_forward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                              last_forward_inputs,
                                              self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)
      padded_last_backward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                               last_backward_inputs,
                                               self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)

      forward_steps, backward_steps = [], []
      for start in range(sequence_len):
        end = start + self.width
        forward_input = padded_last_forward_inputs.narrow(1, start, self.width + 1)
        forward_output = forward_input.transpose(-2, -1).matmul(self.forward_weights[i])
        forward_output = self.forward_blocks[i](forward_output, self.forward_linears[i])

        backward_input = padded_last_backward_inputs.narrow(1, end, self.width + 1)
        backward_output = backward_input.transpose(-2, -1).matmul(self.backward_weights[i])
        backward_output = self.backward_blocks[i](backward_output, self.backward_linears[i])

        forward_steps.append(forward_output)
        backward_steps.append(backward_output)

      last_forward_inputs = torch.stack(forward_steps, dim=1)
      last_backward_inputs = torch.stack(backward_steps, dim=1)

      all_layers_along_steps.append(torch.cat([last_forward_inputs, last_backward_inputs], dim=-1))

    return torch.stack(all_layers_along_steps, dim=0)
