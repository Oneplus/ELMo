from __future__ import absolute_import
from __future__ import unicode_literals
import torch
import numpy as np
from allennlp.modules.highway import Highway
from allennlp.modules.seq2seq_encoders.bidirectional_transformer_encoder import PositionalEncoding
from allennlp.modules.seq2seq_encoders.bidirectional_transformer_encoder import PositionwiseFeedForward
from allennlp.modules.seq2seq_encoders.bidirectional_transformer_encoder import SublayerConnection


class Bengio03HighwayBiLm(torch.nn.Module):
    def __init__(self, width: int,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 n_highway: int,
                 use_position: bool = False,
                 dropout: float = 0.0):
        super(Bengio03HighwayBiLm, self).__init__()
        self.use_position = use_position
        self.n_layers = n_layers
        self.n_highway = n_highway

        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

        self.width = width
        self.input_size = input_size
        self.context_input_size = input_size * (width + 1)
        self.hidden_size = hidden_size

        forward_paddings, backward_paddings = [], []
        forward_blocks, backward_blocks = [], []
        forward_projects, backward_projects = [], []
        for i in range(n_layers):
            forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size)))
            backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size)))

            forward_blocks.append(Highway(hidden_size, num_layers=n_highway))
            backward_blocks.append(Highway(hidden_size, num_layers=n_highway))

            forward_projects.append(torch.nn.Linear(self.context_input_size, hidden_size))
            backward_projects.append(torch.nn.Linear(self.context_input_size, hidden_size))

        self.forward_projects = torch.nn.ModuleList(forward_projects)
        self.backward_projects = torch.nn.ModuleList(backward_projects)
        self.forward_paddings = torch.nn.ParameterList(forward_paddings)
        self.backward_paddings = torch.nn.ParameterList(backward_paddings)
        self.forward_blocks = torch.nn.ModuleList(forward_blocks)
        self.backward_blocks = torch.nn.ModuleList(backward_blocks)

        if self.use_position:
            self.position = PositionalEncoding(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        for layer_index in range(self.n_layers):
            torch.nn.init.xavier_uniform_(self.forward_projects[layer_index].weight)
            torch.nn.init.xavier_uniform_(self.backward_projects[layer_index].weight)

            torch.nn.init.constant_(self.forward_projects[layer_index].bias, 0.)
            torch.nn.init.constant_(self.backward_projects[layer_index].bias, 0.)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
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
                forward_input = padded_last_forward_inputs.narrow(1, start, self.width + 1).contiguous().view(
                    batch_size, -1)
                forward_output = self.activation(self.dropout(self.forward_projects[i](forward_input)))
                forward_output = self.forward_blocks[i](forward_output)

                backward_input = padded_last_backward_inputs.narrow(1, end, self.width + 1).contiguous().view(
                    batch_size, -1)
                backward_output = self.activation(self.dropout(self.backward_projects[i](backward_input)))
                backward_output = self.backward_blocks[i](backward_output)

                forward_steps.append(forward_output)
                backward_steps.append(backward_output)

            last_forward_inputs = torch.stack(forward_steps, dim=1)
            last_backward_inputs = torch.stack(backward_steps, dim=1)

            all_layers_along_steps.append(torch.cat([last_forward_inputs, last_backward_inputs], dim=-1))

        return torch.stack(all_layers_along_steps, dim=0)


class Bengio03ResNetBiLm(torch.nn.Module):
    def __init__(self, width: int,
                 input_size: int,
                 hidden_size: int,
                 n_layers: int,
                 use_position: bool = False,
                 dropout: float = 0.0):
        super(Bengio03ResNetBiLm, self).__init__()
        self.use_position = use_position
        self.n_layers = n_layers

        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()

        self.width = width
        self.input_size = input_size
        self.context_input_size = input_size * (width + 1)
        self.hidden_size = hidden_size

        forward_paddings, backward_paddings = [], []
        forward_projects, backward_projects = [], []
        for i in range(n_layers):
            forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
            backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))

            forward_projects.append(torch.nn.Linear(self.context_input_size, hidden_size))
            backward_projects.append(torch.nn.Linear(self.context_input_size, hidden_size))

        self.forward_projects = torch.nn.ModuleList(forward_projects)
        self.backward_projects = torch.nn.ModuleList(backward_projects)

        self.forward_paddings = torch.nn.ParameterList(forward_paddings)
        self.backward_paddings = torch.nn.ParameterList(backward_paddings)

        self.left_linears = torch.nn.ModuleList(
            [PositionwiseFeedForward(hidden_size, hidden_size, self.config['dropout']) for _ in range(n_layers)])
        self.right_linears = torch.nn.ModuleList(
            [PositionwiseFeedForward(hidden_size, hidden_size, self.config['dropout']) for _ in range(n_layers)])

        self.left_blocks = torch.nn.ModuleList(
            [SublayerConnection(hidden_size, self.config['dropout']) for _ in range(n_layers)])
        self.right_blocks = torch.nn.ModuleList(
            [SublayerConnection(hidden_size, self.config['dropout']) for _ in range(n_layers)])

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor):
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
                forward_input = padded_last_forward_inputs.narrow(1, start, self.width + 1).contiguous().view(
                    batch_size, -1)
                forward_output = self.activation(self.dropout(self.forward_projects[i](forward_input)))
                forward_output = self.forward_blocks[i](forward_output, self.forward_linears[i])

                backward_input = padded_last_backward_inputs.narrow(1, end, self.width + 1).contiguous().view(
                    batch_size, -1)
                backward_output = self.activation(self.dropout(self.backward_projects[i](backward_input)))
                backward_output = self.backward_blocks[i](backward_output, self.backwarrd_linears[i])

                forward_steps.append(forward_output)
                backward_steps.append(backward_output)

            last_forward_inputs = torch.stack(forward_steps, dim=1)
            last_backward_inputs = torch.stack(backward_steps, dim=1)

            all_layers_along_steps.append(torch.cat([last_forward_inputs, last_backward_inputs], dim=-1))

        return torch.stack(all_layers_along_steps, dim=0)
