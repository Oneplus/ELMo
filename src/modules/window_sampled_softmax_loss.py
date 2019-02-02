#!/usr/bin/env python
import torch
import numpy as np


class WindowSampledSoftmaxLoss(torch.nn.Module):
    def __init__(self,
                 embedding_dim: int,
                 n_class: int,
                 n_samples: int,
                 use_cuda: bool):

        super(WindowSampledSoftmaxLoss, self).__init__()
        self.n_samples = n_samples
        self.n_class = n_class
        self.use_cuda = use_cuda
        self.criterion = torch.nn.CrossEntropyLoss(size_average=False)

        # indexing of negative samples word to id
        self.negative_samples = []
        self.word_to_column = {0: 0}

        # indexing of word to id
        self.all_word = []
        self.all_word_to_column = {0: 0}

        self.softmax_w = torch.nn.Embedding(n_class, embedding_dim)
        self.softmax_w.weight.data.normal_(mean=0.0, std=1.0 / np.sqrt(embedding_dim))

        self.softmax_b = torch.nn.Embedding(n_class, 1)
        self.softmax_b.weight.data.fill_(0.0)

        self.current_embed_matrix = None

    def forward(self,
                embeddings: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        batch_size = targets.size(0)

        if self.training:
            for i in range(batch_size):
                targets[i] = self.word_to_column.get(targets[i].tolist())
            word_to_column = self.word_to_column
            words = self.negative_samples
        else:
            for i in range(batch_size):
                targets[i] = self.all_word_to_column.get(targets[i].tolist(), 0)
            word_to_column = self.all_word_to_column
            words = self.all_word

        samples = embeddings.new_zeros(len(word_to_column))
        for word in words:
            samples[word_to_column[word]] = word

        tag_scores = (embeddings.matmul(self.current_embed_matrix)).view(batch_size, -1) + \
                     (self.softmax_b.forward(samples)).view(1, -1)
        return self.criterion(tag_scores, targets)

    def update_embedding_matrix(self):
        if self.training:
            words = self.negative_samples
            word_to_column = self.word_to_column
        else:
            words = self.all_word
            word_to_column = self.all_word_to_column

        columns = torch.LongTensor(len(words) + 1)
        for i, word in enumerate(words):
            columns[word_to_column[word]] = word
        columns[0] = 0

        if self.use_cuda:
            columns = columns.cuda()

        self.current_embed_matrix = self.softmax_w.forward(columns).transpose(0, 1)

    def update_negative_samples(self,
                                word_inp: torch.Tensor,
                                chars_inp: torch.Tensor,
                                mask: torch.Tensor):
        batch_size, seq_len = mask.size()
        words_in_batch = set()
        # put all the words in the batch as `words_in_batch`
        for i in range(batch_size):
            for j in range(seq_len):
                if mask[i][j] == 0:
                    continue
                word = word_inp[i][j].tolist()
                words_in_batch.add(word)

        for word in words_in_batch:
            # update word indexing
            if word not in self.all_word_to_column:
                self.all_word.append(word)
                self.all_word_to_column[word] = len(self.all_word_to_column)

            # update negative samples word indexing
            if word not in self.word_to_column:
                # if the current negative samples don't reach the limit
                if len(self.negative_samples) < self.n_samples:
                    self.negative_samples.append(word)
                    self.word_to_column[word] = len(self.word_to_column)
                # shift all the samples in the words_in_batch to the last one
                else:
                    while self.negative_samples[0] in words_in_batch:
                        self.negative_samples = self.negative_samples[1:] + [self.negative_samples[0]]
                    self.word_to_column[word] = self.word_to_column.pop(self.negative_samples[0])
                    self.negative_samples = self.negative_samples[1:] + [word]
