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
    query, key, value = \
      [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
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
    self.num_layers = config['encoder']['n_layers']

    self.left_attn = MultiHeadedAttention(config['encoder']['n_heads'],
                                          config['encoder']['projection_dim'], config['dropout'])
    self.right_attn = MultiHeadedAttention(config['encoder']['n_heads'],
                                           config['encoder']['projection_dim'], config['dropout'])
    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']

    left_padding = torch.randn(width, hidden_size) / np.sqrt(hidden_size)
    right_padding = torch.randn(width, hidden_size) / np.sqrt(hidden_size)
    self.left_padding = torch.nn.Parameter(left_padding, requires_grad=True)
    self.right_padding = torch.nn.Parameter(right_padding, requires_grad=True)

    if self.use_relative_position_weights:
      left_weights = torch.randn(width + 1)
      right_weights = torch.randn(width + 1)
      self.left_weights = torch.nn.Parameter(left_weights, requires_grad=True)
      self.right_weights = torch.nn.Parameter(right_weights, requires_grad=True)

    if self.use_position:
      self.position = PositionalEncoding(config['encoder']['projection_dim'], self.config['dropout'])

    self.left_block = Highway(hidden_size, num_layers=config['encoder']['n_highway'])
    self.right_block = Highway(hidden_size, num_layers=config['encoder']['n_highway'])

    self.input_size = input_size
    self.width = width

  def forward(self, inputs):
    batch_size, sequence_len, dim = inputs.size()
    if self.use_position:
      inputs = self.position(inputs)
    new_inputs = torch.cat([self.left_padding.expand(batch_size, -1, -1),
                            inputs,
                            self.right_padding.expand(batch_size, -1, -1)], dim=1)

    left_mask = local_mask(new_inputs.size(-2), self.width)
    right_mask = local_mask(new_inputs.size(-2), self.width, left_to_right=False)
    if self.use_cuda:
      left_mask = left_mask.cuda()
      right_mask = right_mask.cuda()

    new_left_inputs = self.left_attn(new_inputs, new_inputs, new_inputs, left_mask)
    new_right_inputs = self.right_attn(new_inputs, new_inputs, new_inputs, right_mask)

    all_layers_along_steps = []
    for start in range(sequence_len):
      end = start + self.width
      left_out = new_left_inputs[:, end, :]
      right_out = new_right_inputs[:, end, :]

      if self.use_relative_position_weights:
        left_out = left_out + new_inputs.narrow(1, start, self.width + 1).transpose(-2, -1).matmul(self.left_weights)
        right_out = right_out + new_inputs.narrow(1, end, self.width + 1).transpose(-2, -1).matmul(self.right_weights)

      left_out = self.left_block(left_out)
      right_out = self.right_block(right_out)
      out = torch.cat([left_out, right_out], dim=1)

      all_layers_along_steps.append(out.unsqueeze(0))

    return torch.stack(all_layers_along_steps, dim=2)
