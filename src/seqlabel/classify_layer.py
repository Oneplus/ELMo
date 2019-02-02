#!/usr/bin/env python
import torch


class ClassifyLayer(torch.nn.Module):
  def __init__(self, n_in, num_tags, use_cuda=False):
    super(ClassifyLayer, self).__init__()
    self.hidden2tag = torch.nn.Linear(n_in, num_tags)
    self.num_tags = num_tags
    self.use_cuda = use_cuda
    self.logsoftmax = torch.nn.LogSoftmax(dim=2)
    weights = torch.ones(num_tags)
    weights[0] = 0
    self.criterion = torch.nn.NLLLoss(weights, size_average=False)

  def forward(self, x, y):
    """
    :param x: torch.Tensor (batch_size, seq_len, n_in)
    :param y: torch.Tensor (batch_size, seq_len)
    :return:
    """
    tag_scores = self.hidden2tag(x)
    if self.training:
      tag_scores = self.logsoftmax(tag_scores)

    _, tag_result = torch.max(tag_scores[:, :, 1:], 2)
    tag_result.add_(1)
    if self.training:
      return tag_result, self.criterion(tag_scores.view(-1, self.num_tags),
                                        torch.autograd.Variable(y).view(-1))
    else:
      return tag_result, torch.FloatTensor([0.0])
