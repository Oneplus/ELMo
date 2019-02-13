#!/usr/bin/env python
import torch
from .encoder_base import EncoderBase
from allennlp.modules.input_variational_dropout import InputVariationalDropout


class ProjectedEncoder(EncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,
                 dropout: float = 0.0):
        super(ProjectedEncoder, self).__init__()
        self.encoder = torch.nn.Linear(inp_dim, hidden_dim, bias=False)
        self.dropout = InputVariationalDropout(dropout)
        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)
        self.encoder.bias.data.fill_(0.)

    def forward(self, x: torch.Tensor, *args):
        return self.dropout(self.encoder(x))

    def get_output_dim(self):
        return self.hidden_dim
