#!/usr/bin/env python
import torch


class InputEmbedderBase(torch.nn.Module):
    def __init__(self, input_field_name: str):
        super(InputEmbedderBase, self).__init__()
        self.input_field_name = input_field_name

    def get_output_dim(self) -> int:
        raise NotImplementedError()
