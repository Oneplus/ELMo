#!/usr/bin/env python
import torch


class InputEncoderBase(torch.nn.Module):
    def __init__(self):
        super(InputEncoderBase, self).__init__()

    def get_output_dim(self):
        raise NotImplementedError()
