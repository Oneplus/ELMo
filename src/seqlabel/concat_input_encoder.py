#!/usr/bin/env python
from typing import Dict, List
import torch
from .input_encoder_base import InputEncoderBase


class ConcatenateInputEncoder(InputEncoderBase):
    def __init__(self, input_info_: Dict[str, int],
                 use_cuda: bool):
        super(ConcatenateInputEncoder, self).__init__(use_cuda)
        self.input_info_ = input_info_
        self.output_dim_ = sum([dim for (name, dim) in input_info_.items()])
        self.ordered_names = [name for (name, dim) in input_info_.items()]

    def forward(self, inputs_: Dict[str, torch.Tensor]) -> torch.Tensor:
        output_ = [inputs_[name] for name in self.ordered_names]
        if len(output_) == 1:
            return output_[0]
        return torch.cat(output_, dim=-1)

    def get_output_dim(self) -> int:
        return self.output_dim_

    def get_ordered_names(self) -> List[str]:
        return self.ordered_names
