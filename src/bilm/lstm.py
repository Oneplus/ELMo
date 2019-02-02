from __future__ import absolute_import
from __future__ import unicode_literals
from typing import Optional, Tuple, List
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from allennlp.nn.util import ConfigurationError
from allennlp.modules.augmented_lstm import AugmentedLstm
from allennlp.modules.encoder_base import _EncoderBase


class LstmbiLm(_EncoderBase):
    def __init__(self, input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 dropout: float):
        super(LstmbiLm, self).__init__(stateful=True)
        self.hidden_size = hidden_size
        self.cell_size = hidden_size

        forward_layers = []
        backward_layers = []

        lstm_input_size = input_size
        for layer_index in range(num_layers):
            forward_layer = AugmentedLstm(input_size=lstm_input_size,
                                          hidden_size=hidden_size,
                                          go_forward=True,
                                          recurrent_dropout_probability=dropout)
            backward_layer = AugmentedLstm(input_size=lstm_input_size,
                                           hidden_size=hidden_size,
                                           go_forward=False,
                                           recurrent_dropout_probability=dropout)

            lstm_input_size = hidden_size
            self.add_module('forward_layer_{}'.format(layer_index), forward_layer)
            self.add_module('backward_layer_{}'.format(layer_index), backward_layer)
            forward_layers.append(forward_layer)
            backward_layers.append(backward_layer)
        self.forward_layers = forward_layers
        self.backward_layers = backward_layers

    def forward(self, inputs: torch.Tensor,
                mask: torch.Tensor):
        batch_size, total_sequence_length = mask.size()
        stacked_sequence_output, final_states, restoration_indices = \
            self.sort_and_run_forward(self._lstm_forward, inputs, mask)

        num_layers, num_valid, returned_timesteps, encoder_dim = stacked_sequence_output.size()
        # Add back invalid rows which were removed in the call to sort_and_run_forward.
        if num_valid < batch_size:
            zeros = stacked_sequence_output.new_zeros(num_layers,
                                                      batch_size - num_valid,
                                                      returned_timesteps,
                                                      encoder_dim)
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 1)

            # The states also need to have invalid rows added back.
            new_states = []
            for state in final_states:
                state_dim = state.size(-1)
                zeros = state.new_zeros(num_layers, batch_size - num_valid, state_dim)
                new_states.append(torch.cat([state, zeros], 1))
            final_states = new_states

        # It's possible to need to pass sequences which are padded to longer than the
        # max length of the sequence to a Seq2StackEncoder. However, packing and unpacking
        # the sequences mean that the returned tensor won't include these dimensions, because
        # the RNN did not need to process them. We add them back on in the form of zeros here.
        sequence_length_difference = total_sequence_length - returned_timesteps
        if sequence_length_difference > 0:
            zeros = stacked_sequence_output.new_zeros(num_layers,
                                                      batch_size,
                                                      sequence_length_difference,
                                                      stacked_sequence_output[0].size(-1))
            stacked_sequence_output = torch.cat([stacked_sequence_output, zeros], 2)

        self._update_states(final_states, restoration_indices)

        # Restore the original indices and return the sequence.
        # Has shape (num_layers, batch_size, sequence_length, hidden_size)
        return stacked_sequence_output.index_select(1, restoration_indices)

    def _lstm_forward(self,
                      inputs: PackedSequence,
                      initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> \
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if initial_state is None:
            hidden_states: List[Optional[Tuple[torch.Tensor,
                                               torch.Tensor]]] = [None] * len(self.forward_layers)
        elif initial_state[0].size()[0] != len(self.forward_layers):
            raise ConfigurationError("Initial states were passed to forward() but the number of "
                                     "initial states does not match the number of layers.")
        else:
            hidden_states = list(zip(initial_state[0].split(1, 0), initial_state[1].split(1, 0)))

        forward_output_sequence = inputs
        backward_output_sequence = inputs

        final_states = []
        sequence_outputs = []
        for layer_index, state in enumerate(hidden_states):
            forward_layer = getattr(self, 'forward_layer_{}'.format(layer_index))
            backward_layer = getattr(self, 'backward_layer_{}'.format(layer_index))

            if state is not None:
                forward_hidden_state, backward_hidden_state = state[0].split(self.hidden_size, 2)
                forward_memory_state, backward_memory_state = state[1].split(self.cell_size, 2)
                forward_state = (forward_hidden_state, forward_memory_state)
                backward_state = (backward_hidden_state, backward_memory_state)
            else:
                forward_state = None
                backward_state = None

            forward_output_sequence, forward_state = forward_layer(forward_output_sequence,
                                                                   forward_state)
            backward_output_sequence, backward_state = backward_layer(backward_output_sequence,
                                                                      backward_state)

            unpacked_forward_output_sequence, _ = pad_packed_sequence(forward_output_sequence, batch_first=True)
            unpacked_backward_output_sequence, _ = pad_packed_sequence(backward_output_sequence, batch_first=True)

            sequence_outputs.append(torch.cat([unpacked_forward_output_sequence,
                                               unpacked_backward_output_sequence], -1))
            # Append the state tuples in a list, so that we can return
            # the final states for all the layers.
            final_states.append((torch.cat([forward_state[0], backward_state[0]], -1),
                                 torch.cat([forward_state[1], backward_state[1]], -1)))

        stacked_sequence_outputs: torch.FloatTensor = torch.stack(sequence_outputs)
        # Stack the hidden state and memory for each layer into 2 tensors of shape
        # (num_layers, batch_size, hidden_size) and (num_layers, batch_size, cell_size)
        # respectively.
        final_hidden_states, final_memory_states = zip(*final_states)
        final_state_tuple: Tuple[torch.FloatTensor,
                                 torch.FloatTensor] = (torch.cat(final_hidden_states, 0),
                                                       torch.cat(final_memory_states, 0))
        return stacked_sequence_outputs, final_state_tuple
