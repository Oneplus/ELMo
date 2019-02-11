from __future__ import absolute_import
from __future__ import unicode_literals
from typing import Tuple, List
import torch
from modules.embeddings import Embeddings
from allennlp.modules.highway import Highway
from allennlp.nn.util import get_mask_from_sequence_lengths, masked_softmax
from allennlp.nn.activations import Activation


class TokenEmbedderBase(torch.nn.Module):
    def __init__(self, output_dim: int,
                 word_embedder: Embeddings,
                 char_embedder: Embeddings):
        super(TokenEmbedderBase, self).__init__()
        self.output_dim = output_dim
        self.word_embedder = word_embedder
        self.char_embedder = char_embedder

    def get_output_dim(self):
        return self.output_dim


class LstmTokenEmbedder(TokenEmbedderBase):
    # Single directional LSTM + self-attention seems to be
    # a good solution of character encoding.
    def __init__(self, output_dim: int,
                 word_embedder: Embeddings,
                 char_embedder: Embeddings,
                 dropout: float):
        super(LstmTokenEmbedder, self).__init__(output_dim, word_embedder, char_embedder)
        emb_dim = 0
        if word_embedder is not None:
            emb_dim += word_embedder.n_d

        if char_embedder is not None:
            emb_dim += char_embedder.n_d
            self.char_encoder = torch.nn.LSTM(char_embedder.n_d, char_embedder.n_d, num_layers=1,
                                              bidirectional=False,
                                              batch_first=True, dropout=dropout)
            self.char_attention = torch.nn.Linear(char_embedder.n_d, 1, bias=False)

        self.projection = torch.nn.Linear(emb_dim, output_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.constant_(self.projection.bias, 0.)

    def forward(self, word_inputs: torch.Tensor,
                char_inputs: torch.Tensor):
        embs = []
        if self.word_embedder is not None:
            word_inputs = torch.autograd.Variable(word_inputs, requires_grad=False)
            if self.use_cuda:
                word_inputs = word_inputs.cuda()
            word_emb = self.word_embedder(word_inputs)
            embs.append(word_emb)

        if self.char_embedder is not None:
            char_inputs, char_lengths = char_inputs
            batch_size, seq_len = char_lengths.size()

            char_inputs = char_inputs.view(batch_size * seq_len, -1)
            char_lengths = char_lengths.view(-1)
            char_mask = get_mask_from_sequence_lengths(char_lengths, char_lengths.max())

            embeded_char_inputs = self.char_embedder(char_inputs)
            encoded_char_outputs, _ = self.char_encoder(embeded_char_inputs)
            char_attentions = masked_softmax(self.char_attention(encoded_char_outputs).squeeze(-1), char_mask, dim=-1)
            encoded_char_outputs = char_attentions.unsqueeze(-1).mul(encoded_char_outputs).sum(dim=1)
            encoded_char_outputs = encoded_char_outputs.view(batch_size, seq_len, -1)
            embs.append(encoded_char_outputs)

        token_embedding = torch.cat(embs, dim=2)

        return self.projection(token_embedding)


class ConvTokenEmbedder(TokenEmbedderBase):
    def __init__(self, output_dim: int,
                 word_embedder: Embeddings,
                 char_embedder: Embeddings,
                 filters: List[Tuple[int, int]],
                 n_highway: int,
                 activation: str):
        super(ConvTokenEmbedder, self).__init__(output_dim, word_embedder, char_embedder)

        self.emb_dim = 0
        if word_embedder is not None:
            self.emb_dim += word_embedder.n_d

        if char_embedder is not None:
            self.convolutions = []
            char_embed_dim = char_embedder.n_d

            for i, (width, num) in enumerate(filters):
                conv = torch.nn.Conv1d(in_channels=char_embed_dim,
                                       out_channels=num,
                                       kernel_size=width,
                                       bias=True)
                self.convolutions.append(conv)

            self.convolutions = torch.nn.ModuleList(self.convolutions)

            self.n_filters = sum(f[1] for f in filters)
            self.n_highway = n_highway

            self.highways = Highway(self.n_filters, self.n_highway, activation=Activation.by_name("relu")())
            self.emb_dim += self.n_filters
            self.activation = Activation.by_name(activation)()

        self.projection = torch.nn.Linear(self.emb_dim, self.output_dim, bias=True)

    def forward(self, word_inputs: torch.Tensor,
                char_inputs: torch.Tensor):
        embs = []
        if self.word_embedder is not None:
            word_inputs = torch.autograd.Variable(word_inputs, requires_grad=False)
            embed_words = self.word_embedder(word_inputs)
            embs.append(embed_words)

        if self.char_embedder is not None:
            char_inputs, char_lengths = char_inputs
            batch_size, seq_len = char_lengths.size()[:2]
            char_inputs = char_inputs.view(batch_size * seq_len, -1)
            char_inputs = torch.autograd.Variable(char_inputs, requires_grad=False)

            embeded_chars = self.char_embedder(char_inputs)
            embeded_chars = torch.transpose(embeded_chars, 1, 2)

            convs = []
            for i in range(len(self.convolutions)):
                convolved = self.convolutions[i](embeded_chars)
                # (batch_size * sequence_length, n_filters for this width)
                convolved, _ = torch.max(convolved, dim=-1)
                convolved = self.activation(convolved)
                convs.append(convolved)
            char_emb = torch.cat(convs, dim=-1)
            char_emb = self.highways(char_emb)

            embs.append(char_emb.view(batch_size, -1, self.n_filters))

        token_embedding = torch.cat(embs, dim=2)

        return self.projection(token_embedding)
