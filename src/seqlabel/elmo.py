#!/usr/bin/env python
from typing import List
import torch
import h5py
import logging
from .input_layer_base import InputLayerBase
logger = logging.getLogger(__name__)


class ContextualizedWordEmbeddings(InputLayerBase):
    def __init__(self, input_field_name, lexicon_path, use_cuda):
        super(ContextualizedWordEmbeddings, self).__init__(input_field_name)
        self.use_cuda = use_cuda
        self.lexicon = h5py.File(lexicon_path, 'r')
        self.dim = self.lexicon['#info'][0].item()
        self.n_layers = self.lexicon['#info'][1].item()

        logger.info('dim: {}'.format(self.dim))
        logger.info('number of layers: {}'.format(self.n_layers))

        weights = torch.randn(self.n_layers)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)

    def forward(self, input_: List[List[str]]) -> torch.Tensor:
        # input_: (batch_size, seq_len, dim)
        batch_size = len(input_)
        seq_len = max([len(seq) for seq in input_])

        encoding_ = torch.FloatTensor(batch_size, seq_len, self.dim).fill_(0.)
        if self.use_cuda:
            encoding_ = encoding_.cuda()

        for i, one_input_ in enumerate(input_):
            sentence_key = '\t'.join(one_input_).replace('.', '$period$').replace('/', '$backslash$')
            one_seq_len_ = len(one_input_)
            data_ = torch.from_numpy(self.lexicon[sentence_key][()]).transpose(0, 1)
            if self.use_cuda:
                data_ = data_.cuda()
            data_ = torch.autograd.Variable(data_, requires_grad=False)
            encoding_[i, :one_seq_len_, :] = data_.transpose(-2, -1).matmul(self.weights)

        # output_: (batch_size, seq_len, max_seg_len, dim)
        return encoding_

    def get_output_dim(self):
        return self.dim


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG)

    seq_len = 5
    dim = 7
    batch_size = 2
    max_seg_len = 3

    encoder = ContextualizedWordEmbeddings(sys.argv[1], False)
    print(encoder)
    print(encoder.get_output_dim())

    input_ = [
        ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
        ['Peter', 'Blackburn'],
        ['BRUSSELS', '1996-08-22']]
    print(input_)
    print(encoder.forward(input_))
