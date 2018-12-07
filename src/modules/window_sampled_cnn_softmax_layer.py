#!/usr/bin/env python
import torch
import math


class WindowSampledCNNSoftmaxLayer(torch.nn.Module):
  def __init__(self, token_embedder, output_dim, n_class, n_samples, corr_dim, use_cuda):
    super(WindowSampledCNNSoftmaxLayer, self).__init__()
    self.token_embedder = token_embedder
    self.n_samples = n_samples
    self.use_cuda = use_cuda
    self.criterion = torch.nn.CrossEntropyLoss(size_average=False)
    self.negative_samples = []
    self.word_to_column = {0: 0}

    self.all_word = []
    self.all_word_to_column = {0: 0}

    self.M = torch.nn.Parameter(torch.Tensor(output_dim, corr_dim))
    stdv = 1. / math.sqrt(self.M.size(1))
    self.M.data.uniform_(-stdv, stdv)

    self.corr = torch.nn.Embedding(n_class, corr_dim)
    self.corr.weight.data.uniform_(-0.25, 0.25)

    self.oov_column = torch.nn.Parameter(torch.Tensor(output_dim, 1))
    self.oov_column.data.uniform_(-0.25, 0.25)

  def forward(self, x, y):
    if self.training:
      for i in range(y.size(0)):
        y[i] = self.word_to_column.get(y[i].tolist())
      samples = torch.LongTensor(len(self.word_to_column)).fill_(0)
      for package in self.negative_samples:
        samples[self.word_to_column[package[0]]] = package[0]
    else:
      for i in range(y.size(0)):
        y[i] = self.all_word_to_column.get(y[i].tolist(), 0)
      samples = torch.LongTensor(len(self.all_word_to_column)).fill_(0)
      for package in self.all_word:
        samples[self.all_word_to_column[package[0]]] = package[0]

    if self.use_cuda:
      samples = samples.cuda()

    tag_scores = (x.matmul(self.embedding_matrix)).view(y.size(0), -1) + \
                 (x.matmul(self.M).matmul(self.corr.forward(samples).transpose(0, 1))).view(y.size(0), -1)
    return self.criterion(tag_scores, y)

  def update_embedding_matrix(self):
    batch_size = 2048
    word_inp, chars_inp = [], []
    if self.training:
      sub_matrices = [self.oov_column]
      samples = self.negative_samples
      id2pack = {}
      for i, package in enumerate(samples):
        id2pack[self.word_to_column[package[0]]] = i
    else:
      sub_matrices = [self.oov_column]
      samples = self.all_word
      id2pack = {}
      for i, package in enumerate(samples):
        id2pack[self.all_word_to_column[package[0]]] = i

    for i in range(len(samples)):
      # [n_samples, 1], [n_samples, 1, x], [n_samples, 1]
      word_inp.append(samples[id2pack[i + 1]][0])
      chars_inp.append(samples[id2pack[i + 1]][1])
      if len(word_inp) == batch_size or i == len(samples) - 1:
        sub_matrices.append(self.token_embedder.forward(torch.LongTensor(word_inp).view(len(word_inp), 1),
                                                        None if chars_inp[0] is None else torch.LongTensor(chars_inp).view(len(word_inp), 1, len(package[1])),
                                                        (len(word_inp), 1)).squeeze(1).transpose(0, 1))
        if not self.training:
          sub_matrices[-1] = sub_matrices[-1].detach()
        word_inp, chars_inp = [], []

    sum = 0
    for mat in sub_matrices:
      sum += mat.size(1)
    # print(sum, len(self.word_to_column))
    self.embedding_matrix = torch.cat(sub_matrices, dim=1)

  def update_negative_samples(self, word_inp, chars_inp, mask):
    batch_size, seq_len = word_inp.size(0), word_inp.size(1)
    in_batch = set()
    for i in range(batch_size):
      for j in range(seq_len):
        if mask[i][j] == 0:
          continue
        word = word_inp[i][j].tolist()
        in_batch.add(word)
    for i in range(batch_size):
      for j in range(seq_len):
        if mask[i][j] == 0:
          continue
        package = (word_inp[i][j].tolist(), None if chars_inp is None else chars_inp[i][j].tolist())
        if package[0] not in self.all_word_to_column:
          self.all_word.append(package)
          self.all_word_to_column[package[0]] = len(self.all_word_to_column)

        if package[0] not in self.word_to_column:
          if len(self.negative_samples) < self.n_samples:
            self.negative_samples.append(package)
            self.word_to_column[package[0]] = len(self.word_to_column)
          else:
            while self.negative_samples[0][0] in in_batch:
              self.negative_samples = self.negative_samples[1:] + [self.negative_samples[0]]
            self.word_to_column[package[0]] = self.word_to_column.pop(self.negative_samples[0][0])
            self.negative_samples = self.negative_samples[1:] + [package]
