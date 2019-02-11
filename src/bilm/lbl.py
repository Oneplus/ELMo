from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import numpy as np
from allennlp.modules.highway import Highway
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionalEncoding
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionwiseFeedForward
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import SublayerConnection


class LBLHighwayBiLm(torch.nn.Module):
    def __init__(self, width: int,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 n_highway: int,
                 use_position: bool = False):
        super(LBLHighwayBiLm, self).__init__()
        self.use_position = use_position
        self.n_layers = n_layers = n_layers
        self.n_highway = n_highway = n_highway

        self.width = width
        self.input_size = input_size
        self.hidden_size = hidden_size

        forward_paddings, backward_paddings = [], []
        forward_weights, backward_weights = [], []
        forward_blocks, backward_blocks = [], []

        for _ in range(n_layers):
            forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
            backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))

            forward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))
            backward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))

            forward_blocks.append(Highway(hidden_size, num_layers=n_highway))
            backward_blocks.append(Highway(hidden_size, num_layers=n_highway))

        self.forward_paddings = torch.nn.ParameterList(forward_paddings)
        self.backward_paddings = torch.nn.ParameterList(backward_paddings)
        self.forward_weights = torch.nn.ParameterList(forward_weights)
        self.backward_weights = torch.nn.ParameterList(backward_weights)
        self.forward_blocks = torch.nn.ModuleList(forward_blocks)
        self.backward_blocks = torch.nn.ModuleList(backward_blocks)

        if self.use_position:
            self.position = PositionalEncoding(hidden_size)

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        batch_size, sequence_len, dim = inputs.size()
        all_layers_along_steps = []

        last_forward_inputs = inputs
        last_backward_inputs = inputs
        for layer_index in range(self.n_layers):
            if self.use_position:
                last_forward_inputs = self.position(last_forward_inputs)
                last_backward_inputs = self.position(last_backward_inputs)

            padded_last_forward_inputs = torch.cat([self.forward_paddings[layer_index].expand(batch_size, -1, -1),
                                                    last_forward_inputs,
                                                    self.backward_paddings[layer_index].expand(batch_size, -1, -1)], dim=1)
            padded_last_backward_inputs = torch.cat([self.forward_paddings[layer_index].expand(batch_size, -1, -1),
                                                     last_backward_inputs,
                                                     self.backward_paddings[layer_index].expand(batch_size, -1, -1)], dim=1)

            forward_steps, backward_steps = [], []
            for start in range(sequence_len):
                end = start + self.width
                forward_input = padded_last_forward_inputs.narrow(1, start, self.width + 1)
                forward_output = forward_input.transpose(-2, -1).matmul(self.forward_weights[layer_index])
                forward_output = self.forward_blocks[layer_index](forward_output)

                backward_input = padded_last_backward_inputs.narrow(1, end, self.width + 1)
                backward_output = backward_input.transpose(-2, -1).matmul(self.backward_weights[layer_index])
                backward_output = self.backward_blocks[layer_index](backward_output)

                forward_steps.append(forward_output)
                backward_steps.append(backward_output)

            last_forward_inputs = torch.stack(forward_steps, dim=1)
            last_backward_inputs = torch.stack(backward_steps, dim=1)

            all_layers_along_steps.append(torch.cat([last_forward_inputs, last_backward_inputs], dim=-1))

        return torch.stack(all_layers_along_steps, dim=0)


class LBLHighwayBiLmV2(torch.nn.Module):
    def __init__(self, width: int,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 n_highway: int,
                 use_position: bool = False):
        super(LBLHighwayBiLmV2, self).__init__()
        self.use_position = use_position
        self.n_layers = n_layers = n_layers
        self.n_highway = n_highway = n_highway

        self.width = width
        self.input_size = input_size
        self.hidden_size = hidden_size

        forward_weights, backward_weights = [], []
        forward_blocks, backward_blocks = [], []

        for _ in range(n_layers):
            forward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))
            backward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))

            forward_blocks.append(Highway(hidden_size, num_layers=n_highway))
            backward_blocks.append(Highway(hidden_size, num_layers=n_highway))

        self.forward_weights = torch.nn.ParameterList(forward_weights)
        self.backward_weights = torch.nn.ParameterList(backward_weights)
        self.forward_blocks = torch.nn.ModuleList(forward_blocks)
        self.backward_blocks = torch.nn.ModuleList(backward_blocks)

        if self.use_position:
            self.position = PositionalEncoding(hidden_size)

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        batch_size, sequence_len, dim = inputs.size()
        sequence_outputs = []

        forward_output_sequence = inputs
        backward_output_sequence = inputs

        for layer_index in range(self.n_layers):
            if self.use_position:
                forward_output_sequence = self.position(forward_output_sequence)
                backward_output_sequence = self.position(backward_output_sequence)

            forward_mask = forward_output_sequence.new_zeros(sequence_len, sequence_len)
            backward_mask = backward_output_sequence.new_zeros(sequence_len, sequence_len)

            for current in range(sequence_len):
                start = max(current - self.width, 0)
                length = current + 1 - start
                forward_mask[start: current + 1, current] = self.forward_weights[layer_index][-length:]
            for current in range(sequence_len):
                end = min(sequence_len, current + self.width + 1)
                length = end - current
                backward_mask[current: end, current] = self.backward_weights[layer_index][: length]
            # print('forward', forward_mask != 0)
            # print('backward', backward_mask != 0)
            forward_mask = forward_mask.unsqueeze(0).unsqueeze(-1)
            backward_mask = backward_mask.unsqueeze(0).unsqueeze(-1)

            forward_output_sequence = (forward_mask * forward_output_sequence.unsqueeze(-2)).sum(1)
            backward_output_sequence = (backward_mask * backward_output_sequence.unsqueeze(-2)).sum(1)

            forward_output_sequence = self.forward_blocks[layer_index](forward_output_sequence)
            backward_output_sequence = self.backward_blocks[layer_index](backward_output_sequence)

            sequence_outputs.append(torch.cat([forward_output_sequence, backward_output_sequence], dim=-1))

        stacked_sequence_outputs = torch.stack(sequence_outputs, dim=0)
        return stacked_sequence_outputs


class LBLResNetBiLm(torch.nn.Module):
    def __init__(self, width: int,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 use_position: bool = False,
                 dropout: float = 0.0):
        super(LBLResNetBiLm, self).__init__()
        self.use_position = use_position

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.ReLU()

        self.width = width
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        forward_paddings, backward_paddings = [], []
        forward_weights, backward_weights = [], []
        for _ in range(self.n_layers):
            forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
            backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
            forward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))
            backward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))

        self.forward_paddings = torch.nn.ParameterList(forward_paddings)
        self.backward_paddings = torch.nn.ParameterList(backward_paddings)
        self.forward_weights = torch.nn.Parameter(forward_weights)
        self.backward_weights = torch.nn.Parameter(backward_weights)

        if self.use_position:
            self.position = PositionalEncoding(hidden_size)

        self.forward_linears = torch.nn.ModuleList(
            [PositionwiseFeedForward(hidden_size, hidden_size, dropout)
             for _ in range(n_layers)])
        self.backward_linears = torch.nn.ModuleList(
            [PositionwiseFeedForward(hidden_size, hidden_size, dropout)
             for _ in range(n_layers)])

        self.forward_blocks = torch.nn.ModuleList(
            [SublayerConnection(hidden_size, dropout) for _ in range(n_layers)])
        self.backward_blocks = torch.nn.ModuleList(
            [SublayerConnection(hidden_size, dropout) for _ in range(n_layers)])

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        batch_size, sequence_len, dim = inputs.size()
        all_layers_along_steps = []

        last_forward_inputs = inputs
        last_backward_inputs = inputs
        for i in range(self.n_layers):
            if self.use_position:
                last_forward_inputs = self.position(last_forward_inputs)
                last_backward_inputs = self.position(last_backward_inputs)

            padded_last_forward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                                    last_forward_inputs,
                                                    self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)
            padded_last_backward_inputs = torch.cat([self.forward_paddings[i].expand(batch_size, -1, -1),
                                                     last_backward_inputs,
                                                     self.backward_paddings[i].expand(batch_size, -1, -1)], dim=1)

            forward_steps, backward_steps = [], []
            for start in range(sequence_len):
                end = start + self.width
                forward_input = padded_last_forward_inputs.narrow(1, start, self.width + 1)
                forward_output = forward_input.transpose(-2, -1).matmul(self.forward_weights[i])
                forward_output = self.forward_blocks[i](forward_output, self.forward_linears[i])

                backward_input = padded_last_backward_inputs.narrow(1, end, self.width + 1)
                backward_output = backward_input.transpose(-2, -1).matmul(self.backward_weights[i])
                backward_output = self.backward_blocks[i](backward_output, self.backward_linears[i])

                forward_steps.append(forward_output)
                backward_steps.append(backward_output)

            last_forward_inputs = torch.stack(forward_steps, dim=1)
            last_backward_inputs = torch.stack(backward_steps, dim=1)

            all_layers_along_steps.append(torch.cat([last_forward_inputs, last_backward_inputs], dim=-1))

        return torch.stack(all_layers_along_steps, dim=0)
