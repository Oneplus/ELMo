#!/usr/bin/env python
from typing import List
import torch


class InputEncoderBase(torch.nn.Module):
    def __init__(self, use_cuda: bool):
        super(InputEncoderBase, self).__init__()
        self.use_cuda = use_cuda

    def get_output_dim(self) -> int:
        raise NotImplementedError()

    def get_ordered_names(self) -> List[str]:
        raise NotImplementedError()
