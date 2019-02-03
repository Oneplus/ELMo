#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict
import os
import sys
import codecs
import argparse
import logging
import json
import torch
import numpy as np
import h5py
from bilm.bilm_base import BiLMBase
from bilm.batch import Batcher, WordBatch, CharacterBatch, VocabBatch
from bilm.io_util import dict2namedtuple, read_corpus_with_original_text, read_conll_corpus_with_original_text

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(BiLMBase):
    def __init__(self, config: Dict,
                 word_batch: WordBatch,
                 char_batch: CharacterBatch):
        super(Model, self).__init__(config, word_batch, char_batch)

    def forward(self, word_inputs, char_inputs, lengths):
        encoded_tokens, embedded_tokens, mask = self._encoding(word_inputs, char_inputs, lengths)
        embedded_tokens = torch.cat([embedded_tokens, embedded_tokens], dim=-1).unsqueeze(0)
        output = torch.cat([embedded_tokens, encoded_tokens], dim=0)
        return output

    def load_model(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'),
                                                       map_location=lambda storage, loc: storage))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'),
                                                map_location=lambda storage, loc: storage))


def test_main():
    # Configurations
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--input_format', default='plain', choices=('plain', 'conll'),
                     help='the input format.')
    cmd.add_argument("--input", help="the path to the raw text file.")
    cmd.add_argument("--output_format", default='hdf5', help='the output format. Supported format includes (hdf5, txt).'
                                                             ' Use comma to separate the format identifiers,'
                                                             ' like \'--output_format=hdf5,plain\'')
    cmd.add_argument("--output_prefix", help='the prefix of the output file. The output file is in the format of '
                                             '<output_prefix>.<output_layer>.<output_format>')
    cmd.add_argument("--output_layer", required=True,
                     help='the target layer to output. 0 for the word encoder, 1 for the first LSTM '
                          'hidden layer, 2 for the second LSTM hidden layer, -1 for an average '
                          'of 3 layers.')
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument("--batch_size", "--batch", type=int, default=1, help='the batch size.')
    args = cmd.parse_args(sys.argv[1:])

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    # load the model configurations
    args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))

    with open(args2.config_path, 'r') as fin:
        conf = json.load(fin)

    c = conf['token_embedder']
    if c['char_dim'] > 0:
        char_batch = CharacterBatch('<oov>', '<pad>', '<eow>', not c.get('char_cased', True), use_cuda)
        with codecs.open(os.path.join(args.model, 'char.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                char_batch.mapping[token] = int(i)
    else:
        char_batch = None

    if c['word_dim'] > 0:
        word_batch = WordBatch(c.get('word_min_cut', 0), '<oov>', '<pad>',
                               not c.get('word_cased', True), use_cuda)
        with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                word_batch.mapping[token] = int(i)
    else:
        word_batch = None

    # instantiate the model
    model = Model(conf, word_batch, char_batch)
    if use_cuda:
        model.cuda()

    logging.info(str(model))
    model.load_model(args.model)

    # read test data according to input format
    read_function = read_corpus_with_original_text if args.input_format == 'plain'\
        else read_conll_corpus_with_original_text

    raw_test_data, original_raw_test_data = read_function(args.input, 10000,
                                                          conf['token_embedder'].get('max_characters_per_token', None))

    # create test batches from the input data.
    test_batcher = Batcher(raw_test_data, word_batch, char_batch, None,
                           args.batch_size, sorting=False, shuffle=False,
                           original_raw_dataset=original_raw_test_data)

    # configure the model to evaluation mode.
    model.eval()

    sent_set = set()
    cnt = 0

    output_formats = args.output_format.split(',')
    output_layers = list(map(int, args.output_layer.split(',')))
    if -1 in output_layers:
        assert len(output_layers) == 1

    handlers = {}
    for output_format in output_formats:
        if output_format not in ('hdf5', 'txt'):
            print('Unknown output_format: {0}'.format(output_format))
            continue

        filename = '{0}.ly{1}.{2}'.format(args.output_prefix, args.output_layer, output_format)
        fout = h5py.File(filename, 'w') if output_format == 'hdf5' else open(filename, 'w')

        handlers[output_format] = fout
        dim = conf['encoder']['projection_dim'] * 2
        n_layers = len(output_layers)
        if output_format == 'hdf5':
            info = np.asarray([dim, n_layers])
            fout.create_dataset('#info', info.shape, dtype='int', data=info)
        else:
            print('#projection_dim: {}'.format(dim), file=fout)
            print('#n_layers: {}'.format(n_layers), file=fout)

    for word_inputs, char_inputs, lengths, texts, _ in test_batcher.get():
        output = model.forward(word_inputs, char_inputs, lengths)
        for i, text in enumerate(texts):
            sent = '\t'.join(text)
            sent = sent.replace('.', '$period$')
            sent = sent.replace('/', '$backslash$')
            if sent in sent_set:
                continue
            sent_set.add(sent)

            data = output[:, i, :lengths[i], :].data
            if use_cuda:
                data = data.cpu()
            data = data.numpy()

            for output_format in output_formats:
                fout = handlers[output_format]
                if output_layers[0] == -1:
                    payload = np.average(data, axis=0)
                else:
                    payload = data[output_layers, :, :]

                if output_format == 'hdf5':
                    fout.create_dataset(sent, payload.shape, dtype='float32', data=payload)
                else:
                    for word, row in zip(text, payload):
                        print('{0}\t{1}'.format(word, '\t'.join(['{0:.8f}'.format(elem) for elem in row])), file=fout)
                    print('', file=fout)

            cnt += 1
            if cnt % 1000 == 0:
                logging.info('Finished {0} sentences.'.format(cnt))
    for _, handler in handlers.items():
        handler.close()


if __name__ == "__main__":
    test_main()
