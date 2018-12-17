#!/usr/bin/env python
import torch
from seqlabel.crf_layer import CRFLayer


class PartialCRFLayer(CRFLayer):
  uncertain = 1

  def __init__(self, n_in, num_tags, use_cuda=False):
    super(PartialCRFLayer, self).__init__(n_in, num_tags, use_cuda)

  def forward(self, x, y):
    emissions = self.hidden2tag(x)
    new_emissions = emissions.permute(1, 0, 2).contiguous()

    if self.training:
      new_y = y.permute(1, 0).contiguous()
      numerator = self._compute_log_constrained_partition_function(new_emissions, torch.nn.Variable(new_y))
      denominator = self._compute_log_partition_function(new_emissions)
      llh = denominator - numerator
      return None, torch.sum(llh)
    else:
      path = self._viterbi_decode(new_emissions)
      path = path.permute(1, 0)
      return path, None

  def _compute_log_constrained_partition_function(self, emissions, tags):
    seq_length, batch_size = tags.size(0), tags.size(1)
    mask = torch.ones(seq_length, batch_size, self.num_tags).float()

    # create mask
    tags_data = tags.data
    for i in range(seq_length):
      for j in range(batch_size):
        if tags_data[i][j] != self.uncertain:
          mask[i][j].zero_()
          mask[i][j][tags_data[i][j]] = 1
        else:
          mask[i][j][0] = 0
          mask[i][j][self.uncertain] = 0

    mask = Variable(mask)
    log_prob = emissions[0] * mask[0] + (1 - mask[0]) * self.ninf

    for i in range(1, seq_length):
      prev_mask, cur_mask = mask[i - 1], mask[i]
      transition_mask = torch.bmm(prev_mask.unsqueeze(2), cur_mask.unsqueeze(1))
      # (batch_size, num_tags, 1)
      broadcast_log_prob = log_prob.unsqueeze(2) * prev_mask.unsqueeze(2) + (1 - prev_mask.unsqueeze(2)) * self.ninf
      # (batch_size, num_tags, num_tags)
      broadcast_transitions = self.transitions.unsqueeze(0) * transition_mask + (1 - transition_mask) * self.ninf
      # (batch_size, 1, num_tags)
      broadcast_emissions = emissions[i].unsqueeze(1) * cur_mask.unsqueeze(1) + (1 - cur_mask.unsqueeze(1)) * self.ninf

      score = broadcast_log_prob + broadcast_transitions + broadcast_emissions
      log_prob = self._log_sum_exp(score, 1)

    return self._log_sum_exp(log_prob, 1)

  def _viterbi_decode(self, emissions):
    seq_length, batch_size = emissions.size(0), emissions.size(1)
    mask = torch.ones(seq_length, batch_size, self.num_tags).float()

    # create mask
    for i in range(seq_length):
      for j in range(batch_size):
        mask[i][j][0] = 0
        mask[i][j][self.uncertain] = 0

    mask = torch.nn.Variable(mask)
    return self._viterbi_decode_with_mask(emissions, mask)
