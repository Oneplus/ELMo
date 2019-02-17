#!/usr/bin/env python
from typing import Tuple
import torch
import logging
from .input_embed_base import InputEmbedderBase
from modules.embeddings import Embeddings
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_softmax
logger = logging.getLogger(__name__)


class LstmTokenEmbedder(InputEmbedderBase):
    def __init__(self, output_dim: int,
                 embeddings: Embeddings,
                 dropout: float,
                 use_cuda: bool,
                 input_field_name: str = None):
        super(LstmTokenEmbedder, self).__init__(input_field_name)
        self.embeddings = embeddings
        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.encoder_ = torch.nn.LSTM(embeddings.n_d, embeddings.n_d,
                                      num_layers=1, bidirectional=False,
                                      batch_first=True, dropout=dropout)
        self.attention = torch.nn.Linear(embeddings.n_d, 1, bias=False)
        self.projection = torch.nn.Linear(embeddings.n_d, output_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.attention.weight)
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.constant_(self.projection.bias, 0.)

    def forward(self, input_: Tuple[torch.Tensor, torch.Tensor]):
        chars, lengths = input_
        batch_size, seq_len, max_chars = chars.size()

        chars = chars.view(batch_size * seq_len, -1)
        lengths = lengths.view(batch_size * seq_len)
        mask = get_mask_from_sequence_lengths(lengths, max_chars)
        chars = torch.autograd.Variable(chars, requires_grad=False)

        embeded_chars = self.embeddings(chars)
        output, _ = self.encoder_(embeded_chars)
        attentions = masked_softmax(self.attention(output).squeeze(-1), mask, dim=-1)
        output = torch.bmm(output.permute(0, 2, 1), attentions.unsqueeze(-1))

        return self.projection(output.view(batch_size, seq_len, -1))

    def get_embed_dim(self):
        return self.output_dim

    def get_output_dim(self):
        return self.output_dim
