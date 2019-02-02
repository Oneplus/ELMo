#!/usr/bin/env python
import torch


class InputEncoderBase(torch.nn.Module):
    def __init__(self):
        super(InputEncoderBase, self).__init__()

    def encoding_dim(self):
        raise NotImplementedError()
