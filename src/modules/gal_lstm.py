#!/usr/bin/env python
import torch
from .locked_dropout import LockedDropout
from .weight_dropout import WeightDrop


class GalLSTM(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, bidirectional=False,
               num_layers=1,
               wdrop=0.25, idrop=0.25, batch_first=True):
    super(GalLSTM, self).__init__()
    # Modified LockedDropout that support batch first arrangement
    self.lockdrop = LockedDropout(dropout=idrop, batch_first=batch_first)
    self.idrop = idrop
    self.wdrop = wdrop
    self.n_layers = num_layers
    self.rnns = [
      torch.nn.LSTM(input_dim if l == 0 else hidden_dim * 2, hidden_dim, num_layers=1,
                    batch_first=batch_first, dropout=0, bidirectional=bidirectional)
      for l in range(num_layers)
    ]
    if wdrop:
      self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop, variational=True) for rnn in self.rnns]
    self.rnns = torch.nn.ModuleList(self.rnns)

  def forward(self, x):
    raw_output = self.lockdrop(x)
    new_hidden, new_cell_state = [], []
    for l, rnn in enumerate(self.rnns):
      raw_output, (new_h, new_c) = rnn(raw_output)
      new_hidden.append(new_h)
      new_cell_state.append(new_c)
      raw_output = self.lockdrop(raw_output)

    hidden = torch.cat(new_hidden, 0)
    cell_state = torch.cat(new_cell_state, 0)
    return raw_output, (hidden, cell_state)
