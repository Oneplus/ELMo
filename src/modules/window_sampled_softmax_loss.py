#!/usr/bin/env python
from typing import List
import torch
import numpy as np
from collections import Counter


class WindowSampledSoftmaxLoss(torch.nn.Module):
    def __init__(self,
                 num_words: int,
                 embedding_dim: int,
                 num_samples: int,
                 sparse: bool = False,
                 unk_id: int = None,
                 use_character_inputs: bool = True):

        super(WindowSampledSoftmaxLoss, self).__init__()
        self.tie_embeddings = False

        assert num_samples < num_words

        if sparse:
            # create our own sparse embedding
            self.softmax_w = torch.nn.Embedding(num_words, embedding_dim, sparse=True)
            self.softmax_w.weight.data.normal_(mean=0.0, std=1.0 / np.sqrt(embedding_dim))
            self.softmax_b = torch.nn.Embedding(num_words, 1, sparse=True)
            self.softmax_b.weight.data.fill_(0.0)
        else:
            # just create tensors to use as the embeddings
            # Glorit init (std=(1.0 / sqrt(fan_in))
            self.softmax_w = torch.nn.Parameter(torch.randn(num_words, embedding_dim) / np.sqrt(embedding_dim))
            self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))

        self.sparse = sparse
        self.use_character_inputs = use_character_inputs

        if use_character_inputs:
            self._unk_id = unk_id

        self._criterion = torch.nn.CrossEntropyLoss(size_average=False)
        self._negative_samples = []
        self._negative_samples_counter = Counter()
        self._sampled_indexing = np.zeros(num_words, dtype=np.int64)
        self._num_samples = num_samples
        self._embedding_dim = embedding_dim
        self._num_words = num_words

    def forward(self,
                embeddings: torch.Tensor,
                targets: torch.Tensor,
                target_token_embedding: torch.Tensor = None) -> torch.Tensor:

        if embeddings.shape[0] == 0:
            # empty batch
            return torch.tensor(0.0).to(embeddings.device)  # pylint: disable=not-callable

        if not self.training:
            return self._forward_eval(embeddings, targets)
        else:
            return self._forward_train(embeddings, targets, target_token_embedding)

    def _forward_train(self,
                       embeddings: torch.Tensor,
                       targets: torch.Tensor,
                       target_token_embedding: torch.Tensor) -> torch.Tensor:
        assert len(self._negative_samples) > 0

        all_ids = targets.new_zeros(len(self._negative_samples_counter))
        for i, negative_sample in enumerate(self._negative_samples_counter):
            all_ids[i] = negative_sample
            self._sampled_indexing[negative_sample] = i

        new_targets = torch.from_numpy(self._sampled_indexing[targets.cpu().detach().numpy()]).to(targets.device)

        all_ids.requires_grad_(False)
        new_targets.requires_grad_(False)

        if self.sparse:
            all_ids_1 = all_ids.unsqueeze(1)
            all_w = self.softmax_w(all_ids_1).squeeze(1)
            all_b = self.softmax_b(all_ids_1).squeeze(2).squeeze(1)
        else:
            all_w = torch.nn.functional.embedding(all_ids, self.softmax_w)
            # the unsqueeze / squeeze works around an issue with 1 dim
            # embeddings
            all_b = torch.nn.functional.embedding(all_ids, self.softmax_b.unsqueeze(1)).squeeze(1)

        scores = embeddings.matmul(all_w.t()) + all_b.unsqueeze(0)
        return self._criterion(scores, new_targets)

    def _forward_eval(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sparse:
            w = self.softmax_w.weight
            b = self.softmax_b.weight.squeeze(1)
        else:
            w = self.softmax_w
            b = self.softmax_b

        log_softmax = torch.nn.functional.log_softmax(torch.matmul(embeddings, w.t()) + b, dim=-1)
        if self.tie_embeddings and not self.use_character_inputs:
            targets_ = targets + 1
        else:
            targets_ = targets
        return torch.nn.functional.nll_loss(log_softmax, targets_.long(),
                                            reduction="sum")

    def update_negative_samples(self,
                                targets: List[str]):
        # put all the words in the batch as `words_in_batch`
        self._negative_samples.extend(targets)
        self._negative_samples_counter.update(targets)

        num_samples_to_delete = len(self._negative_samples_counter) - self._num_samples
        if num_samples_to_delete <= 0:
            return

        tail = 0
        while num_samples_to_delete > 0:
            target = self._negative_samples[tail]
            tail += 1
            self._negative_samples_counter[target] -= 1
            if self._negative_samples_counter[target] == 0:
                num_samples_to_delete -= 1
                self._negative_samples_counter.pop(target)

        self._negative_samples = self._negative_samples[tail:]
