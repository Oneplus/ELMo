#!/usr/bin/env python
from typing import Dict
import torch
import math
from .batch import WordBatch, CharacterBatch
from .token_embedder import ConvTokenEmbedder, LstmTokenEmbedder
from .lstm import LstmbiLm
from .bengio03 import Bengio03HighwayBiLm, Bengio03ResNetBiLm
from .lbl import LBLHighwayBiLm, LBLResNetBiLm
from .self_attn import SelfAttentiveLBLBiLM
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn.util import get_mask_from_sequence_lengths
from modules.embeddings import Embeddings


class BiLMBase(torch.nn.Module):
    def __init__(self, conf: Dict,
                 word_batch: WordBatch,
                 char_batch: CharacterBatch):
        super(BiLMBase, self).__init__()
        self.conf = conf
        self.dropout = InputVariationalDropout(p=conf['dropout'])

        c = conf['token_embedder']
        if word_batch is not None:
            word_embedder = Embeddings(c['word_dim'], word_batch.mapping, None, False, normalize=False)
        else:
            word_embedder = None

        if char_batch is not None:
            char_embedder = Embeddings(c['char_dim'], char_batch.mapping, None, False, normalize=False)
        else:
            char_embedder = None

        token_embedder_name = c['name'].lower()
        if token_embedder_name == 'cnn':
            self.token_embedder = ConvTokenEmbedder(output_dim=conf['encoder']['projection_dim'],
                                                    word_embedder=word_embedder,
                                                    char_embedder=char_embedder,
                                                    filters=c['filters'],
                                                    n_highway=c['n_highway'],
                                                    activation=c['activation'])
        elif token_embedder_name == 'lstm':
            self.token_embedder = LstmTokenEmbedder(output_dim=conf['encoder']['projection_dim'],
                                                    word_embedder=word_embedder,
                                                    char_embedder=char_embedder,
                                                    dropout=conf['dropout'])
        else:
            raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))

        self.add_sentence_boundary = c['add_sentence_boundary']
        if self.add_sentence_boundary:
            dim = self.token_embedder.get_output_dim()
            scale = 1. / math.sqrt(dim)
            self.bos_embeddings = torch.nn.Parameter(torch.randn(dim) * scale)
            self.eos_embeddings = torch.nn.Parameter(torch.randn(dim) * scale)

        c = conf['encoder']
        encoder_name = c['name'].lower()
        if encoder_name == 'elmo':
            self.encoder = ElmoLstm(input_size=c['projection_dim'],
                                    hidden_size=c['projection_dim'],
                                    cell_size=c['dim'],
                                    requires_grad=True,
                                    num_layers=c['n_layers'],
                                    recurrent_dropout_probability=conf['dropout'],
                                    memory_cell_clip_value=c['cell_clip'],
                                    state_projection_clip_value=c['proj_clip'])
        elif encoder_name == 'lstm':
            self.encoder = LstmbiLm(input_size=c['projection_dim'],
                                    hidden_size=c['projection_dim'],
                                    num_layers=c['n_layers'],
                                    dropout=conf['dropout'])
        elif encoder_name == 'bengio03highway':
            self.encoder = Bengio03HighwayBiLm(width=c['width'],
                                               input_size=c['projection_dim'],
                                               hidden_size=c['projection_dim'],
                                               n_layers=c['n_layers'],
                                               n_highway=c['n_highway'],
                                               use_position=c.get('position', False),
                                               dropout=conf['dropout'])
        elif encoder_name == 'bengio03resnet':
            self.encoder = Bengio03ResNetBiLm(width=c['width'],
                                              input_size=c['projection_dim'],
                                              hidden_size=c['projection_dim'],
                                              n_layers=c['n_layers'],
                                              use_position=c.get('position', False),
                                              dropout=conf['dropout'])
        elif encoder_name == 'lblhighway':
            self.encoder = LBLHighwayBiLm(width=c['width'],
                                          input_size=c['projection_dim'],
                                          hidden_size=c['projection_dim'],
                                          n_layers=c['n_layers'],
                                          n_highway=c['n_highway'],
                                          use_position=c.get('position', False),
                                          dropout=conf['dropout'])
        elif encoder_name == 'lblresnet':
            self.encoder = LBLResNetBiLm(width=c['width'],
                                         input_size=c['projection_dim'],
                                         hidden_size=c['projection_dim'],
                                         n_layers=c['n_layers'],
                                         use_position=c.get('position', False),
                                         dropout=conf['dropout'])
        elif encoder_name == 'selfattn':
            self.encoder = SelfAttentiveLBLBiLM(width=c['width'],
                                                input_size=c['projection_dim'],
                                                hidden_size=c['projection_dim'],
                                                n_heads=c['n_heads'],
                                                n_layers=c['n_layers'],
                                                n_highway=c['n_highway'],
                                                use_position=c.get('position', False),
                                                use_relative_position=c.get('relative_position_weights', False),
                                                dropout=conf['dropout'])
        else:
            raise ValueError('Unknown encoder name: {}'.format(encoder_name))

        self.output_dim = conf['encoder']['projection_dim']

    def _encoding(self, word_inputs: torch.Tensor,
                chars_inputs: torch.Tensor,
                lengths: torch.Tensor,):

        embedded_tokens = self.token_embedder(word_inputs, chars_inputs)
        embedded_tokens = self.dropout(embedded_tokens)

        mask = get_mask_from_sequence_lengths(lengths, lengths.max())

        if self.add_sentence_boundary:
            embedded_tokens_with_boundary, mask_with_boundary = \
                self._add_sentence_boundary(embedded_tokens, mask)
            encoded_tokens = self.encoder(embedded_tokens_with_boundary,
                                          mask_with_boundary)
            return encoded_tokens[:, :, 1:-1, :], embedded_tokens, mask
        else:
            encoded_tokens = self.encoder(embedded_tokens, mask)
            return encoded_tokens, embedded_tokens, mask

    def _add_sentence_boundary(self, tensor: torch.Tensor,
                               mask: torch.Tensor):
        sequence_lengths = mask.sum(dim=1).detach().cpu().numpy()
        tensor_shape = list(tensor.data.shape)
        new_shape = list(tensor_shape)
        new_shape[1] = tensor_shape[1] + 2
        tensor_with_boundary_embeddings = tensor.new_zeros(*new_shape)
        new_mask = mask.new_zeros(*new_shape[:-1])

        tensor_with_boundary_embeddings[:, 1:-1, :] = tensor
        new_mask[:, 1:-1] = mask
        for i, j in enumerate(sequence_lengths):
            tensor_with_boundary_embeddings[i, 0, :] = self.bos_embeddings
            tensor_with_boundary_embeddings[i, j + 1, :] = self.eos_embeddings
            new_mask[i, 0] = 1
            new_mask[i, j + 1] = 1

        return tensor_with_boundary_embeddings, new_mask

    def forward(self, *input):
        raise NotImplementedError()
