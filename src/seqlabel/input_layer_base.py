#!/usr/bin/env python
import torch


class InputLayerBase(torch.nn.Module):
    def __init__(self, input_field_name: str):
        super(InputLayerBase, self).__init__()
        self.input_field_name = input_field_name

    def encoding_dim(self) -> int:
        raise NotImplementedError()
