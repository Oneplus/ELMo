#!/usr/bin/env python
from typing import Dict
import torch
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
            self.token_embedder = ConvTokenEmbedder(conf, word_embedder, char_embedder)
        elif token_embedder_name == 'lstm':
            self.token_embedder = LstmTokenEmbedder(conf, word_embedder, char_embedder)
        else:
            raise ValueError('Unknown token embedder name: {}'.format(token_embedder_name))

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

        embeded_tokens = self.token_embedder(word_inputs, chars_inputs)
        embeded_tokens = self.dropout(embeded_tokens)

        mask = get_mask_from_sequence_lengths(lengths, lengths.max())
        encoded_tokens = self.encoder(embeded_tokens, mask)
        return encoded_tokens, embeded_tokens, mask

    def forward(self, *input):
        raise NotImplementedError()
