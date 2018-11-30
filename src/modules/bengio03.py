from __future__ import absolute_import
from __future__ import unicode_literals
import torch
from .positional_encoding import PositionalEncoding
from .positionwise_feedforward import PositionwiseFeedForward
from .sublayer_connection import SublayerConnection


class Bengio03biLm(torch.nn.Module):
  def __init__(self, config, use_cuda=False):
    super(Bengio03biLm, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.use_position = config['encoder'].get('position', False)

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

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

    all_layers, last_layers = [], []
    for start in range(sequence_len):
      end = start + self.width
      left_inp = new_inputs.narrow(1, start, self.width).contiguous().view(batch_size, -1)
      right_inp = new_inputs.narrow(1, end + 1, self.width).contiguous().view(batch_size, -1)

      left_out = self.dropout(self.activation(self.left_project(left_inp)))
      right_out = self.dropout(self.activation(self.right_project(right_inp)))

      layers = []
      for i in range(self.num_layers):
        left_out = self.left_blocks[i](left_out, self.left_linears[i])
        right_out = self.right_blocks[i](right_out, self.right_linears[i])
        layers.append(torch.cat([left_out, right_out], dim=1))

      last_layers.append(layers[-1])
      all_layers.append(torch.stack(layers, dim=0))

    return torch.stack(all_layers, dim=1), torch.stack(last_layers, dim=1)
