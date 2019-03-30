from __future__ import absolute_import
from __future__ import unicode_literals
from typing import Sequence
import torch
from allennlp.modules.seq2seq_encoders.gated_cnn_encoder import GatedCnnEncoder


class GatedCnnLm(torch.nn.Module):
    def __init__(self, input_size: int,
                 layers: Sequence[Sequence[Sequence[int]]],
                 dropout: float):
        super(GatedCnnLm, self).__init__()

        self._module = GatedCnnEncoder(input_dim=input_size, layers=layers,
                                       dropout=dropout, return_all_layers=True)

    def forward(self, inputs: torch.Tensor,
                mask: torch.Tensor):

        sequence_outputs = self._module(inputs, mask)
        stacked_sequence_outputs = torch.stack(sequence_outputs, dim=0)

        return stacked_sequence_outputs
