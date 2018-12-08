from __future__ import absolute_import
from __future__ import unicode_literals
import torch
from modules.highway import Highway
from modules.positional_encoding import PositionalEncoding
from modules.positionwise_feedforward import PositionwiseFeedForward
from modules.sublayer_connection import SublayerConnection


class Bengio03HighwayBiLm(torch.nn.Module):
  def __init__(self, config, use_cuda=False):
    super(Bengio03HighwayBiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.use_position = config['encoder'].get('position', False)
    self.num_layers = config['encoder']['n_layers']

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim'] * (width + 1)
    hidden_size = config['encoder']['projection_dim']

    left_padding = torch.FloatTensor(width, hidden_size)
    right_padding = torch.FloatTensor(width, hidden_size)

    self.left_padding = torch.nn.Parameter(left_padding)
    self.right_padding = torch.nn.Parameter(right_padding)

    if self.use_position:
      self.position = PositionalEncoding(config['encoder']['projection_dim'], self.config['dropout'])

    self.left_project = torch.nn.Linear(input_size, hidden_size)
    self.right_project = torch.nn.Linear(input_size, hidden_size)
    self.left_highway = Highway(hidden_size, num_layers=self.num_layers)
    self.right_highway = Highway(hidden_size, num_layers=self.num_layers)

    self.input_size = input_size
    self.width = width

  def forward(self, inputs):
    """

    :param inputs:
    :return:
    """
    batch_size, sequence_len, dim = inputs.size()
    if self.use_position:
      inputs = self.position(inputs)
    new_inputs = torch.cat([self.left_padding.expand(batch_size, -1, -1),
                            inputs,
                            self.right_padding.expand(batch_size, -1, -1)], dim=1)

    all_layers_along_steps, last_layer_along_steps = [], []
    for start in range(sequence_len):
      end = start + self.width
      # left_inp: [32 x 9 x 512]
      left_inp = new_inputs.narrow(1, start, self.width + 1).contiguous().view(batch_size, -1)
      right_inp = new_inputs.narrow(1, end, self.width + 1).contiguous().view(batch_size, -1)

      # left_out: [32 x 512]
      left_out = self.dropout(self.activation(self.left_project(self.dropout(left_inp))))
      right_out = self.dropout(self.activation(self.right_project(self.dropout(right_inp))))

      # left_out: [32 x 512]
      left_out = self.left_highway(left_out)
      right_out = self.right_highway(right_out)

      # out: [32 x 1024]
      out = torch.cat([left_out, right_out], dim=1)

      last_layer_along_steps.append(out)
      # all_layers[-1]: [1 x 32 x 1024]
      all_layers_along_steps.append(out.unsqueeze(0))

    # ret[0]: [1 x 32 x 10 x 1024]
    return torch.stack(all_layers_along_steps, dim=2), torch.stack(last_layer_along_steps, dim=1)


class Bengio03ResNetBiLm(torch.nn.Module):
  def __init__(self, config, use_cuda=False):
    super(Bengio03ResNetBiLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.use_position = config['encoder'].get('position', False)

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim'] * width
    hidden_size = config['encoder']['projection_dim']
    num_layers = config['encoder']['n_layers']

    left_padding = torch.FloatTensor(width, hidden_size)
    right_padding = torch.FloatTensor(width, hidden_size)

    self.left_padding = torch.nn.Parameter(left_padding)
    self.right_padding = torch.nn.Parameter(right_padding)

    if self.use_position:
      self.position = PositionalEncoding(config['encoder']['projection_dim'], self.config['dropout'])

    self.left_project = torch.nn.Linear(input_size, hidden_size)
    self.right_project = torch.nn.Linear(input_size, hidden_size)

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
    """

    :param inputs:
    :return:
    """
    batch_size, sequence_len, dim = inputs.size()
    if self.use_position:
      inputs = self.position(inputs)
    new_inputs = torch.cat([self.left_padding.expand(batch_size, -1, -1),
                            inputs,
                            self.right_padding.expand(batch_size, -1, -1)], dim=1)

    all_layers_along_steps, last_layer_along_steps = [], []
    for start in range(sequence_len):
      end = start + self.width
      # left_inp: [32 x 8 x 512]
      left_inp = new_inputs.narrow(1, start, self.width).contiguous().view(batch_size, -1)
      right_inp = new_inputs.narrow(1, end + 1, self.width).contiguous().view(batch_size, -1)

      # left_out: [32 x 512]
      left_out = self.dropout(self.activation(self.left_project(left_inp)))
      right_out = self.dropout(self.activation(self.right_project(right_inp)))

      layers = []
      for i in range(self.num_layers):
        # left_out: [32 x 512]
        left_out = self.left_blocks[i](left_out, self.left_linears[i])
        right_out = self.right_blocks[i](right_out, self.right_linears[i])
        # layers[-1]: [32 x 1024]
        layers.append(torch.cat([left_out, right_out], dim=1))

      last_layer_along_steps.append(layers[-1])
      # all_layers[-1]: [2 x 32 x 1024]
      all_layers_along_steps.append(torch.stack(layers, dim=0))

    # ret[0]: [2 x 32 x 10 x 1024]
    return torch.stack(all_layers_along_steps, dim=2), torch.stack(last_layer_along_steps, dim=1)
