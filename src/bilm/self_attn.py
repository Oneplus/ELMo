import torch
import math
import copy
import numpy as np
from modules.highway import Highway
from modules.positional_encoding import PositionalEncoding


def clones(module, N):
  """Produce N identical layers."""
  return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask=None, dropout=None):
  """
  Compute 'Scaled Dot Product Attention'

  :param query: [batch, h, seq_len, d_k]
  :param key: [batch, h, seq_len, d_k]
  :param value: [batch, h, seq_len, d_k]
  :param mask: [1, seq_len, seq_len]
  :param dropout:
  :return:
  """
  d_k = query.size(-1)
  scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    scores = scores.masked_fill(mask == 0, -1e9)
  p_attn = torch.nn.functional.softmax(scores, dim=-1)
  if dropout is not None:
    p_attn = dropout(p_attn)
  return torch.matmul(p_attn, value), p_attn


def subsequent_mask(size):
  """Mask out subsequent positions."""
  attn_shape = (1, size, size)
  mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
  return torch.from_numpy(mask) == 0


def local_mask(size, width, left_to_right=True):
  """Mask out subsequent positions."""
  attn_shape = (1, size, size)
  mask = np.triu(np.ones(attn_shape), k=-width - 1) * (1 - np.triu(np.ones(attn_shape), k=1))
  if not left_to_right:
    mask = np.flip(mask)
  mask = mask.astype('uint8')
  return torch.from_numpy(mask)


class MultiHeadedAttention(torch.nn.Module):
  def __init__(self, h, d_model, dropout=0.1):
    """Take in model size and number of heads."""
    super(MultiHeadedAttention, self).__init__()
    assert d_model % h == 0
    # We assume d_v always equals d_k
    self.d_k = d_model // h
    self.h = h
    self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
    self.attn = None
    self.dropout = torch.nn.Dropout(p=dropout)

  def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
              mask=None) -> torch.Tensor:
    """

    :param query: [batch, seq_len, d_model]
    :param key: [batch, seq_len, d_model]
    :param value: [batch, seq_len, d_model]
    :param mask: [1, seq_len, seq_len]
    :return:
    """
    if mask is not None:
      # Same mask applied to all h heads.
      mask = mask.unsqueeze(1)
    nbatches = query.size(0)

    # 1) Do all the linear projections in batch from d_model => h x d_k
    query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                         for l, x in zip(self.linears, (query, key, value))]

    # 2) Apply attention on all the projected vectors in batch.
    x, self.attn = attention(query, key, value, mask=mask,
                             dropout=self.dropout)

    # 3) "Concat" using a view and apply a final linear.
    x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
    return self.linears[-1](x)


class SelfAttentiveLBLBiLM(torch.nn.Module):
  def __init__(self, config, use_cuda=False):
    super(SelfAttentiveLBLBiLM, self).__init__()
    self.config = config
    self.use_cuda = use_cuda
    self.use_position = config['encoder'].get('position', False)
    self.use_relative_position_weights = config['encoder'].get('relative_position_weights', False)
    self.n_layers = n_layers = config['encoder']['n_layers']
    self.n_highway = n_highway = config['encoder']['n_highway']
    self.n_heads = n_heads = config['encoder']['n_heads']
    self.input_size = input_size = config['encoder']['projection_dim']
    self.width = width = config['encoder']['width']
    self.hidden_size = hidden_size = config['encoder']['projection_dim']

    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    forward_attns, backward_attns = [], []
    forward_paddings, backward_paddings = [], []
    forward_blocks, backward_blocks = [], []
    forward_weights, backward_weights = [], []

    for _ in range(n_layers):
      forward_attns.append(MultiHeadedAttention(n_heads, hidden_size, config['dropout']))
      backward_attns.append(MultiHeadedAttention(n_heads, hidden_size, config['dropout']))

      forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
      backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))

      forward_blocks.append(Highway(hidden_size, n_highway))
      backward_blocks.append(Highway(hidden_size, n_highway))

      if self.use_relative_position_weights:
        forward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))
        backward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))

    self.forward_attns = torch.nn.ModuleList(forward_attns)
    self.backward_attns = torch.nn.ModuleList(backward_attns)

    self.forward_paddings = torch.nn.ParameterList(forward_paddings)
    self.backward_paddings = torch.nn.ParameterList(backward_paddings)

    self.forward_blocks = torch.nn.ModuleList(forward_blocks)
    self.backward_blocks = torch.nn.ModuleList(backward_blocks)

    if self.use_relative_position_weights:
      self.forward_weights = torch.nn.ParameterList(forward_weights)
      self.backward_weights = torch.nn.ParameterList(backward_weights)

    if self.use_position:
      self.position = PositionalEncoding(config['encoder']['projection_dim'], self.config['dropout'])

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    all_layers_along_steps = []

    forward_inputs = inputs
    backward_inputs = inputs

    forward_mask = local_mask(sequence_len + self.width * 2, self.width)
    backward_mask = local_mask(sequence_len + self.width * 2, self.width, left_to_right=False)
    if self.use_cuda:
      forward_mask = forward_mask.cuda()
      backward_mask = backward_mask.cuda()

    for i in range(self.n_layers):
      if self.use_position:
        forward_inputs = self.position(forward_inputs)
        backward_inputs = self.position(backward_inputs)

      forward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                  forward_inputs,
                                  self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)
      backward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                   backward_inputs,
                                   self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)

      forward_inputs = self.forward_attns[i](forward_inputs, forward_inputs,
                                             forward_inputs, forward_mask)
      backward_inputs = self.backward_attns[i](backward_inputs, backward_inputs,
                                               backward_inputs, backward_mask)

      forward_steps, backward_steps = [], []
      for start in range(sequence_len):
        end = start + self.width
        forward_output = forward_inputs[:, end, :]
        backward_output = backward_inputs[:, end, :]

        if self.use_relative_position_weights:
          forward_output = forward_output + forward_inputs.narrow(1, start, self.width + 1).\
            transpose(-2, -1).matmul(self.forward_weights[i])
          backward_output = backward_output + backward_inputs.narrow(1, end, self.width + 1).\
            transpose(-2, -1).matmul(self.backward_weights[i])

        forward_output = self.forward_blocks[i](forward_output)
        backward_output = self.backward_blocks[i](backward_output)

        forward_steps.append(forward_output)
        backward_steps.append(backward_output)

      forward_inputs = torch.stack(forward_steps, dim=1)
      backward_inputs = torch.stack(backward_steps, dim=1)
      all_layers_along_steps.append(torch.cat([forward_inputs, backward_inputs], dim=-1))

    return torch.stack(all_layers_along_steps, dim=0)
