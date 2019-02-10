import torch
import numpy as np
from allennlp.modules.highway import Highway
from allennlp.modules.seq2seq_encoders.bidirectional_transformer_encoder import PositionalEncoding
from allennlp.modules.seq2seq_encoders.bidirectional_transformer_encoder import MultiHeadedAttention


def local_mask(size, width, device, left_to_right=True):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=-width - 1) * (1 - np.triu(np.ones(attn_shape), k=1))
    if not left_to_right:
        mask = np.flip(mask)
    mask = mask.astype('uint8')
    return torch.from_numpy(mask).to(device)


class SelfAttentiveLBLBiLM(torch.nn.Module):
    def __init__(self, width: int,
                 input_size: int,
                 hidden_size: int,
                 n_heads: int,
                 n_layers: int,
                 n_highway: int,
                 use_position: bool = False,
                 use_relative_position: bool = False,
                 dropout: float = 0.0):
        super(SelfAttentiveLBLBiLM, self).__init__()
        self.use_position = use_position
        self.use_relative_position_weights = use_relative_position
        self.n_layers = n_layers
        self.n_highway = n_highway
        self.n_heads = n_heads
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        forward_attns, backward_attns = [], []
        forward_paddings, backward_paddings = [], []
        forward_blocks, backward_blocks = [], []
        forward_weights, backward_weights = [], []

        for _ in range(n_layers):
            forward_attns.append(MultiHeadedAttention(n_heads, hidden_size, dropout))
            backward_attns.append(MultiHeadedAttention(n_heads, hidden_size, dropout))

            forward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))
            backward_paddings.append(torch.nn.Parameter(torch.randn(width, hidden_size) / np.sqrt(hidden_size)))

            forward_blocks.append(Highway(hidden_size, n_highway))
            backward_blocks.append(Highway(hidden_size, n_highway))

            if self.use_relative_position_weights:
                forward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))
                backward_weights.append(torch.nn.Parameter(torch.randn(width + 1)))

        self.forward_attns = torch.nn.ModuleList(forward_attns)
        self.backward_attns = torch.nn.ModuleList(backward_attns)

        self.forward_paddings = torch.nn.ParameterList(forward_paddings)
        self.backward_paddings = torch.nn.ParameterList(backward_paddings)

        self.forward_blocks = torch.nn.ModuleList(forward_blocks)
        self.backward_blocks = torch.nn.ModuleList(backward_blocks)

        if self.use_relative_position_weights:
            self.forward_weights = torch.nn.ParameterList(forward_weights)
            self.backward_weights = torch.nn.ParameterList(backward_weights)

        if self.use_position:
            self.position = PositionalEncoding(hidden_size)

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor):
        batch_size, sequence_len, dim = inputs.size()
        all_layers_along_steps = []

        forward_inputs = inputs
        backward_inputs = inputs

        forward_mask = local_mask(sequence_len + self.width * 2, self.width, inputs.device)
        backward_mask = local_mask(sequence_len + self.width * 2, self.width, inputs.device,
                                   left_to_right=False)

        for layer_index in range(self.n_layers):
            if self.use_position:
                forward_inputs = self.position(forward_inputs)
                backward_inputs = self.position(backward_inputs)

            forward_inputs = torch.cat([self.forward_paddings[layer_index].expand(batch_size, -1, -1),
                                        forward_inputs,
                                        self.backward_paddings[layer_index].expand(batch_size, -1, -1)], dim=1)
            backward_inputs = torch.cat([self.forward_paddings[layer_index].expand(batch_size, -1, -1),
                                         backward_inputs,
                                         self.backward_paddings[layer_index].expand(batch_size, -1, -1)], dim=1)

            forward_inputs = self.forward_attns[layer_index](forward_inputs, forward_inputs,
                                                             forward_inputs, forward_mask)
            backward_inputs = self.backward_attns[layer_index](backward_inputs, backward_inputs,
                                                               backward_inputs, backward_mask)
            print(self.forward_weights[layer_index].size())

            forward_steps, backward_steps = [], []
            for start in range(sequence_len):
                end = start + self.width
                forward_output = forward_inputs[:, end, :]
                backward_output = backward_inputs[:, end, :]

                if self.use_relative_position_weights:
                    forward_output = forward_output + forward_inputs.narrow(1, start, self.width + 1). \
                        transpose(-2, -1).matmul(self.forward_weights[layer_index])
                    backward_output = backward_output + backward_inputs.narrow(1, end, self.width + 1). \
                        transpose(-2, -1).matmul(self.backward_weights[layer_index])

                forward_output = self.forward_blocks[layer_index](forward_output)
                backward_output = self.backward_blocks[layer_index](backward_output)

                forward_steps.append(forward_output)
                backward_steps.append(backward_output)

            forward_inputs = torch.stack(forward_steps, dim=1)
            backward_inputs = torch.stack(backward_steps, dim=1)
            all_layers_along_steps.append(torch.cat([forward_inputs, backward_inputs], dim=-1))

        return torch.stack(all_layers_along_steps, dim=0)
