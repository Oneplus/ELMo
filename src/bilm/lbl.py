from __future__ import absolute_import
from __future__ import unicode_literals
import torch
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
    self.num_layers = config['encoder']['n_layers']

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']

    left_padding = torch.FloatTensor(width, hidden_size)
    right_padding = torch.FloatTensor(width, hidden_size)
    left_weights = torch.FloatTensor(width + 1).fill_(1. / (width + 1))
    right_weights = torch.FloatTensor(width + 1).fill_(1. / (width + 1))

    self.left_padding = torch.nn.Parameter(left_padding)
    self.right_padding = torch.nn.Parameter(right_padding)
    self.left_weights = torch.nn.Parameter(left_weights)
    self.right_weights = torch.nn.Parameter(right_weights)

    if self.use_position:
      self.position = PositionalEncoding(config['encoder']['projection_dim'], self.config['dropout'])

    self.left_block = Highway(hidden_size, num_layers=self.num_layers)
    self.right_block = Highway(hidden_size, num_layers=self.num_layers)

    self.input_size = input_size
    self.width = width

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    if self.use_position:
      inputs = self.position(inputs)
    new_inputs = torch.cat([self.left_padding.expand(batch_size, -1, -1),
                            inputs,
                            self.right_padding.expand(batch_size, -1, -1)], dim=1)

    all_layers_along_steps, last_layer_along_steps = [], []
    for start in range(sequence_len):
      end = start + self.width
      left_inp = new_inputs.narrow(1, start, self.width + 1)
      left_out = left_inp.transpose(-2, -1).matmul(self.left_weights)

      right_inp = new_inputs.narrow(1, end, self.width + 1)
      right_out = right_inp.transpose(-2, -1).matmul(self.right_weights)

      left_out = self.left_block(self.dropout(left_out))
      right_out = self.right_block(self.dropout(right_out))
      out = torch.cat([left_out, right_out], dim=1)

      last_layer_along_steps.append(out)
      all_layers_along_steps.append(out.unsqueeze(0))

    return torch.stack(all_layers_along_steps, dim=2), torch.stack(last_layer_along_steps, dim=1)


class LBLResNetBiLm(torch.nn.Module):
  def __init__(self, config, use_cuda=False):
    super(LBLResNetBiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.use_position = config['encoder'].get('position', False)

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']
    num_layers = config['encoder']['n_layers']

    left_padding = torch.FloatTensor(width, hidden_size)
    right_padding = torch.FloatTensor(width, hidden_size)
    left_weights = torch.FloatTensor(width + 1)
    right_weights = torch.FloatTensor(width + 1)

    self.left_padding = torch.nn.Parameter(left_padding)
    self.right_padding = torch.nn.Parameter(right_padding)
    self.left_weights = torch.nn.Parameter(left_weights)
    self.right_weights = torch.nn.Parameter(right_weights)

    if self.use_position:
      self.position = PositionalEncoding(config['encoder']['projection_dim'], self.config['dropout'])

    self.left_linears = torch.nn.ModuleList(
      [PositionwiseFeedForward(hidden_size, hidden_size, self.config['dropout'])
       for _ in range(num_layers)])
    self.right_linears = torch.nn.ModuleList(
      [PositionwiseFeedForward(hidden_size, hidden_size, self.config['dropout'])
       for _ in range(num_layers)])

    self.left_blocks = torch.nn.ModuleList(
      [SublayerConnection(hidden_size, self.config['dropout']) for _ in range(num_layers)])
    self.right_blocks = torch.nn.ModuleList(
      [SublayerConnection(hidden_size, self.config['dropout']) for _ in range(num_layers)])

    self.input_size = input_size
    self.num_layers = num_layers
    self.width = width

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    if self.use_position:
      inputs = self.position(inputs)
    new_inputs = torch.cat([self.left_padding.expand(batch_size, -1, -1),
                            inputs,
                            self.right_padding.expand(batch_size, -1, -1)], dim=1)

    all_layers_along_steps, last_layer_along_steps = [], []
    for start in range(sequence_len):
      end = start + self.width
      left_inp = new_inputs.narrow(1, start, self.width)
      left_out = left_inp.transpose(-2, -1).matmul(self.left_weights)

      right_inp = new_inputs.narrow(1, end + 1, self.width)
      right_out = right_inp.transpose(-2, -1).matmul(self.right_weights)

      layers = []
      for i in range(self.num_layers):
        left_out = self.left_blocks[i](left_out, self.left_linears[i])
        right_out = self.right_blocks[i](right_out, self.right_linears[i])
        layers.append(torch.cat([left_out, right_out], dim=1))

      last_layer_along_steps.append(layers[-1])
      all_layers_along_steps.append(torch.stack(layers, dim=0))

    return torch.stack(all_layers_along_steps, dim=2), torch.stack(last_layer_along_steps, dim=1)
