#!/usr/bin/env python
import torch


class EncoderBase(torch.nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()

    def get_output_dim(self):
        raise NotImplementedError()
