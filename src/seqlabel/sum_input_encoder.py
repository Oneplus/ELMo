#!/usr/bin/env python
from typing import Dict, List
import torch
from .input_encoder_base import InputEncoderBase


class AffineTransformInputEncoder(InputEncoderBase):
    def __init__(self, input_info_: Dict[str, int],
                 output_dim_: int,
                 use_cuda: bool):
        super(AffineTransformInputEncoder, self).__init__(use_cuda)
        self.input_info_ = input_info_
        self.output_dim_ = output_dim_

        projections = {}
        self.ordered_names = []
        for i, (name, dim) in enumerate(input_info_.items()):
            if i == 0:
                projections[name] = torch.nn.Linear(dim, output_dim_)
            else:
                projections[name] = torch.nn.Linear(dim, output_dim_, bias=False)
            self.ordered_names.append(name)
        self.projections = torch.nn.ModuleDict(projections)

        self.reset_parameters()

    def reset_parameters(self):
        for i, name in enumerate(self.ordered_names):
            torch.nn.init.xavier_uniform_(self.projections[name].weight.data)
            if i == 0:
                self.projections[name].bias.data.fill_(0.)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        arbitrary_input_ = next(iter(inputs.values()))
        batch_size, seq_len, _ = arbitrary_input_.size()
        output = arbitrary_input_.new_zeros([batch_size, seq_len, self.output_dim_])

        for name, projection in self.projections.items():
            output.add_(projection(inputs[name]))

        return output

    def get_output_dim(self) -> int:
        return self.output_dim_

    def get_ordered_names(self) -> List[str]:
        return self.ordered_names


class SummationInputEncoder(InputEncoderBase):
    def __init__(self, input_info_: Dict[str, int],
                 use_cuda: bool):
        super(SummationInputEncoder, self).__init__(use_cuda)
        self.input_info_ = input_info_
        self.output_dim_ = list(input_info_.values())[0]

        self.ordered_names = []
        for i, (name, dim) in enumerate(input_info_.items()):
            assert dim == self.output_dim_, 'Expected output dim ({0}) and' \
                                            ' input dim ({0}) not equal.'.format(self.output_dim_, dim)
            self.ordered_names.append(name)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        arbitrary_input_ = next(iter(inputs.values()))
        batch_size, seq_len, _ = arbitrary_input_.size()
        output = arbitrary_input_.new_zeros([batch_size, seq_len, self.output_dim_])

        for name in self.ordered_names:
            output.add_(inputs[name])

        return output

    def get_output_dim(self) -> int:
        return self.output_dim_

    def get_ordered_names(self) -> List[str]:
        return self.ordered_names
