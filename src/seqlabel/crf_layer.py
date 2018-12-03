import torch


class CRFLayer(torch.nn.Module):
  ninf = -1e8

  def __init__(self, n_in, num_tags, use_cuda=False):
    super(CRFLayer, self).__init__()
    self.n_in = n_in
    self.num_tags = num_tags
    self.hidden2tag = torch.nn.Linear(n_in, num_tags)
    self.use_cuda = use_cuda

    self.transitions = torch.nn.Parameter(torch.FloatTensor(num_tags, num_tags))
    torch.nn.init.uniform_(self.transitions, -0.1, 0.1)

  def forward(self, x, y):
    emissions = self.hidden2tag(x)
    new_emissions = emissions.permute(1, 0, 2).contiguous()

    if self.training:
      new_y = y.permute(1, 0).contiguous()
      numerator = self._compute_joint_llh(new_emissions, torch.autograd.Variable(new_y))
      denominator = self._compute_log_partition_function(new_emissions)
      llh = denominator - numerator
      return None, torch.sum(llh)
    else:
      path = self._viterbi_decode(new_emissions)
      path = path.permute(1, 0)
      return path, None

  def _compute_joint_llh(self, emissions, tags):
    seq_length = emissions.size(0)
    llh = torch.zeros_like(tags[0]).float()
    for i in range(seq_length - 1):
      cur_tag, next_tag = tags[i], tags[i + 1]
      llh += emissions[i].gather(1, cur_tag.view(-1, 1)).squeeze(1)
      transition_score = self.transitions[cur_tag, next_tag]
      llh += transition_score
    cur_tag = tags[-1]
    llh += emissions[-1].gather(1, cur_tag.view(-1, 1)).squeeze(1)

    return llh

  def _compute_log_partition_function(self, emissions):
    seq_length = emissions.size(0)
    log_prob = torch.zeros_like(emissions[0]) + emissions[0]

    for i in range(1, seq_length):
      broadcast_log_prob = log_prob.unsqueeze(2)  # (batch_size, num_tags, 1)
      broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
      broadcast_emissions = emissions[i].unsqueeze(1)  # (batch_size, 1, num_tags)

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

    mask = torch.autograd.Variable(mask).cuda() if self.use_cuda else torch.autograd.Variable(mask)
    return self._viterbi_decode_with_mask(emissions, mask)

  def _viterbi_decode_with_mask(self, emissions, mask):
    seq_length = emissions.size(0)

    # (batch_size, num_tags)
    viterbi_score = emissions[0] * mask[0] + (1 - mask[0]) * self.ninf
    viterbi_path = []
    for i in range(1, seq_length):
      prev_mask, cur_mask = mask[i - 1], mask[i]
      transition_mask = torch.bmm(prev_mask.unsqueeze(2), cur_mask.unsqueeze(1))

      # (batch_size, num_tags, 1)
      broadcast_score = viterbi_score.unsqueeze(2) * prev_mask.unsqueeze(2) + (1 - prev_mask.unsqueeze(2)) * self.ninf
      # (batch_size, num_tags, num_tags)
      broadcast_transitions = self.transitions.unsqueeze(0) * transition_mask + (1 - transition_mask) * self.ninf
      # (batch_size, 1, num_tags)
      broadcast_emission = emissions[i].unsqueeze(1) * cur_mask.unsqueeze(1) + (1 - cur_mask.unsqueeze(1)) * self.ninf

      score = broadcast_score + broadcast_transitions + broadcast_emission
      # (batch_size, num_tags), (batch_size, num_tags)
      best_score, best_path = score.max(1)
      viterbi_score = best_score
      viterbi_path.append(best_path)

    # _, (batch_size, )
    _, best_last_tag = viterbi_score.max(1)
    best_tags = [best_last_tag.view(-1, 1)]
    for path in reversed(viterbi_path):
      # indexing
      best_last_tag = path.gather(1, best_tags[-1])
      best_tags.append(best_last_tag)

    best_tags.reverse()
    return torch.stack(best_tags).squeeze(2)

  @staticmethod
  def _log_sum_exp(tensor, dim):
    # Find the max value along `dim`
    offset, _ = tensor.max(dim)
    # Make offset broadcastable
    broadcast_offset = offset.unsqueeze(dim)
    # Perform log-sum-exp safely
    safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
    # Add offset back
    return offset + safe_log_sum_exp
