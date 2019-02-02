#!/usr/bin/env python
import torch
from .input_encoder_base import InputEncoderBase


class ProjectedInputEncoder(InputEncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,):
        super(ProjectedInputEncoder, self).__init__()
        self.encoder_ = torch.nn.Linear(inp_dim, hidden_dim, bias=False)
        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.encoder_.weight.data)
        self.encoder_.bias.data.fill_(0.)

    def forward(self, x):
        return self.encoder_(x)

    def encoding_dim(self):
        return self.hidden_dim
