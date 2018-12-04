import torch
import math
import copy
import numpy as np
from modules.highway import Highway


def clones(module, N):
  """Produce N identical layers."""
  return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
  """Compute 'Scaled Dot Product Attention'"""
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

  def forward(self, query, key, value, mask=None):
    """Implements Figure 2"""
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
    self.num_layers = config['encoder']['n_layers']

    self.left_attn = MultiHeadedAttention(64, config['encoder']['projection_dim'], config['dropout'])
    self.right_attn = MultiHeadedAttention(64, config['encoder']['projection_dim'], config['dropout'])
    self.dropout = torch.nn.Dropout(self.config['dropout'])
    self.activation = torch.nn.ReLU()

    width = config['encoder']['width']
    input_size = config['encoder']['projection_dim']
    hidden_size = config['encoder']['projection_dim']

    if use_cuda:
       self.left_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.cuda.FloatTensor(width, hidden_size))
    else:
       self.left_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))
       self.right_padding = torch.autograd.Variable(torch.FloatTensor(width, hidden_size))

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

    left_mask = local_mask(new_inputs.size(-2), self.width)
    right_mask = local_mask(new_inputs.size(-2), self.width, left_to_right=False)

    new_left_inputs = self.left_attn(new_inputs, new_inputs, new_inputs, left_mask)
    new_right_inputs = self.right_attn(new_inputs, new_inputs, new_inputs, right_mask)

    all_layers_along_steps, last_layer_along_steps = [], []
    for start in range(sequence_len):
      end = start + self.width
      left_out = new_left_inputs[:, end, :]
      right_out = new_right_inputs[:, end, :]

      left_out = self.left_block(left_out)
      right_out = self.right_block(right_out)
      out = torch.cat([left_out, right_out], dim=1)

      last_layer_along_steps.append(out)
      all_layers_along_steps.append(out.unsqueeze(0))

    return torch.stack(all_layers_along_steps, dim=2), torch.stack(last_layer_along_steps, dim=1)
