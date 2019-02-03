#!/usr/bin/env python
import torch
from .input_encoder_base import InputEncoderBase


class ProjectedInputEncoder(InputEncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,):
        super(ProjectedInputEncoder, self).__init__()
        self.encoder = torch.nn.Linear(inp_dim, hidden_dim, bias=False)
        self.hidden_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.encoder.weight.data)
        self.encoder.bias.data.fill_(0.)

    def forward(self, x: torch.Tensor, *args):
        return self.encoder(x)

    def get_output_dim(self):
        return self.hidden_dim
