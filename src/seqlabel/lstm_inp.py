#!/usr/bin/env python
import torch
from .input_encoder_base import InputEncoderBase
from allennlp.common.params import Params
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder


class GalLSTMInputEncoder(InputEncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,
                 n_layer: int,
                 dropout: float):
        super(GalLSTMInputEncoder, self).__init__()
        self.encoder_ = Seq2SeqEncoder.from_params(Params({
            "type": "stacked_bidirectional_lstm",
            "num_layers": n_layer,
            "input_size": inp_dim,
            "hidden_size": hidden_dim,
            "recurrent_dropout_probability": dropout,
            "use_highway": True}))

        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.size()
        raw_output, _ = self.encoder_(x)
        # raw_output: (batch_size, seq_len, hidden_dim)
        return raw_output

    def encoding_dim(self):
        return self.hidden_dim * 2


class LSTMInputEncoder(InputEncoderBase):
    def __init__(self, inp_dim: int,
                 hidden_dim: int,
                 n_layer: int,
                 dropout: float):
        super(LSTMInputEncoder, self).__init__()
        self.encoder_ = torch.nn.LSTM(inp_dim, hidden_dim,
                                      num_layers=n_layer, dropout=dropout,
                                      bidirectional=True, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch_size, seq_len, dim)
        batch_size, seq_len, _ = x.size()
        raw_output, _ = self.encoder_(x)
        # raw_output: (batch_size, seq_len, hidden_dim)
        return raw_output

    def encoding_dim(self):
        return self.hidden_dim * 2
