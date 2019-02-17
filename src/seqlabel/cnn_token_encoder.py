#!/usr/bin/env python
from typing import List, Tuple
import torch
from .input_embed_base import InputEmbedderBase
from modules.embeddings import Embeddings
from allennlp.modules.highway import Highway
from allennlp.nn.activations import Activation


class ConvTokenEmbedder(InputEmbedderBase):
    def __init__(self, output_dim: int,
                 embeddings: Embeddings,
                 filters: List[Tuple[int, int]],
                 n_highway: int,
                 activation: str,
                 use_cuda: bool,
                 input_field_name: str = None):
        super(ConvTokenEmbedder, self).__init__(input_field_name)
        self.embeddings = embeddings
        self.output_dim = output_dim
        self.use_cuda = use_cuda
        self.filters = filters

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(in_channels=embeddings.n_d,
                                   out_channels=num,
                                   kernel_size=width,
                                   bias=True)
            convolutions.append(conv)

        self.convolutions = torch.nn.ModuleList(convolutions)

        self.n_filters = sum(f[1] for f in filters)
        self.n_highway = n_highway

        self.highways = Highway(self.n_filters, self.n_highway, activation=torch.nn.functional.relu)

        self.activation = Activation.by_name(activation)()
        self.projection = torch.nn.Linear(self.n_filters, output_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.projection.weight.data)
        self.projection.bias.data.fill_(0.)

    def forward(self, input_: Tuple[torch.Tensor, torch.Tensor]):
        chars, lengths = input_
        batch_size, seq_len, max_chars = chars.size()

        chars = chars.view(batch_size * seq_len, -1)
        chars = torch.autograd.Variable(chars, requires_grad=False)

        embeded_chars = self.embeddings(chars)
        embeded_chars = torch.transpose(embeded_chars, 1, 2)

        convs = []
        for i in range(len(self.convolutions)):
            convolved = self.convolutions[i](embeded_chars)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = self.activation(convolved)
            convs.append(convolved)

        output = torch.cat(convs, dim=-1)
        output = self.highways(output)
        output = output.view(batch_size, seq_len, -1)

        return self.projection(output)

    def get_embed_dim(self):
        return self.output_dim

    def get_output_dim(self):
        return self.output_dim
