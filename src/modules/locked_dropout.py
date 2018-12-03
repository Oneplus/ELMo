import torch


class LockedDropout(torch.nn.Module):
  def __init__(self, dropout, batch_first=False):
    super(LockedDropout, self).__init__()
    self.dropout = dropout
    self.batch_first = batch_first

  def forward(self, x):
    if not self.training or not self.dropout:
      return x
    if self.batch_first:
      m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
    else:
      m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
    mask = torch.autograd.Variable(m, requires_grad=False) / (1 - self.dropout)
    mask = mask.expand_as(x)
    return mask * x
