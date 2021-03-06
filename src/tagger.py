#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List, Union
import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import tempfile
import collections
import torch
import subprocess
import shutil
import numpy as np
from seqlabel.batch import InputBatchBase, Batcher, BucketBatcher, BatcherBase
from seqlabel.batch import TagBatch, LengthBatch, TextBatch, WordBatch, CharacterBatch
from seqlabel.elmo import InputEmbedderBase, ContextualizedWordEmbeddings
from seqlabel.lstm_token_encoder import LstmTokenEmbedder
from seqlabel.cnn_token_encoder import ConvTokenEmbedder
from seqlabel.sum_input_encoder import AffineTransformInputEncoder, SummationInputEncoder
from seqlabel.concat_input_encoder import ConcatenateInputEncoder
from seqlabel.dummy_inp import DummyEncoder
from seqlabel.project_inp import ProjectedEncoder
from seqlabel.crf_layer import CRFLayer
from seqlabel.classify_layer import ClassifyLayer
from modules.embeddings import Embeddings
from modules.embeddings import load_embedding_txt
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.stacked_bidirectional_lstm import StackedBidirectionalLstm
from allennlp.modules.input_variational_dropout import InputVariationalDropout
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.training.optimizers import DenseSparseAdam
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def dict2namedtuple(dic: Dict):
    return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_conllu_dataset(path: str):
    dataset = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for data in fin.read().strip().split('\n\n'):
            lines = data.splitlines()
            items = []
            for line in lines:
                if line.startswith('#'):
                    continue
                fields = tuple(line.strip().split('\t'))
                if '.' in fields[0] or '-' in fields[0]:
                    continue
                items.append(fields)
            dataset.append(items)
    return dataset


def read_tag_dataset(path: str):
    dataset = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for data in fin.read().strip().split('\n\n'):
            items = tuple([line.split() for line in data.splitlines()])
            dataset.append(items)
    return dataset


def read_segment_corpus(path: str):
    dataset = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            fields = line.strip().rsplit('|||')
            fields = [field.split() for field in fields]
            for field in fields:
                assert len(field) == len(fields[0])
            items = []
            n_fields, n_items = len(fields), len(fields[0])
            for i in range(n_items):
                items.append(tuple([fields[i][j] for j in range(n_fields)]))
            dataset.append(items)
    return dataset


class SeqLabelModel(torch.nn.Module):
    def __init__(self, conf: Dict,
                 input_batchers: Dict[str, Union[WordBatch, CharacterBatch]],
                 n_class: int,
                 use_cuda: bool):
        super(SeqLabelModel, self).__init__()
        self.n_class = n_class
        self.use_cuda = use_cuda
        self.input_dropout = torch.nn.Dropout2d(p=conf["dropout"])
        self.dropout = InputVariationalDropout(p=conf['dropout'])

        input_layers = {}
        for i, c in enumerate(conf['input']):
            if c['type'] == 'embeddings':
                if 'pretrained' in c:
                    embs = load_embedding_txt(c['pretrained'], c['has_header'])
                    logger.info('loaded {0} embedding entries.'.format(len(embs[0])))
                else:
                    embs = None
                name = c['name']
                mapping = input_batchers[name].mapping
                layer = Embeddings(c['dim'], mapping, fix_emb=c['fixed'],
                                   embs=embs, normalize=c.get('normalize', False), input_field_name=name)
                logger.info('embedding for field {0} '
                            'created with {1} x {2}.'.format(c['field'], layer.n_V, layer.n_d))
                input_layers[name] = layer

            elif c['type'] == 'cnn_encoder' or c['type'] == 'lstm_encoder':
                name = c['name']
                mapping = input_batchers[name].mapping
                embeddings = Embeddings(c['dim'], mapping, fix_emb=False, embs=None, normalize=False,
                                        input_field_name='{0}_ch_emb'.format(name))
                logger.info('character embedding for field {0} '
                            'created with {1} x {2}.'.format(c['field'], embeddings.n_V, embeddings.n_d))
                if c['type'] == 'lstm_encoder':
                    layer = LstmTokenEmbedder(c['dim'], embeddings, conf['dropout'], use_cuda, input_field_name=name)
                elif c['type'] == 'cnn_encoder':
                    layer = ConvTokenEmbedder(c['dim'], embeddings, c['filters'], c.get('n_highway', 1),
                                              c.get('activation', 'relu'), use_cuda, input_field_name=name)
                else:
                    raise ValueError('Unknown type: {}'.format(c['type']))
                input_layers[name] = layer

            elif c['type'] == 'elmo':
                name = c['name']
                layer = ContextualizedWordEmbeddings(name, c['path'], use_cuda)
                input_layers[name] = layer

            else:
                raise ValueError('{} unknown input layer'.format(c['type']))

        self.input_layers = torch.nn.ModuleDict(input_layers)
        input_encoders = []
        input_dim = 0
        for i, c in enumerate(conf['input_encoder']):
            input_info = {name: self.input_layers[name].get_output_dim() for name in c['input']}

            if c['type'] == 'affine':
                input_encoder = AffineTransformInputEncoder(input_info, c['dim'], use_cuda)
            elif c['type'] == 'sum':
                input_encoder = SummationInputEncoder(input_info, use_cuda)
            elif c['type'] == 'concat':
                input_encoder = ConcatenateInputEncoder(input_info, use_cuda)
            else:
                raise ValueError('{} unknown input encoder'.format(c['type']))

            input_dim += input_encoder.get_output_dim()
            input_encoders.append(input_encoder)

        self.input_encoders = torch.nn.ModuleList(input_encoders)

        encoder_name = conf['encoder']['type'].lower()
        if encoder_name == 'stacked_bidirectional_lstm':
            lstm = StackedBidirectionalLstm(input_size=input_dim,
                                            hidden_size=conf['encoder']['hidden_dim'],
                                            num_layers=conf['encoder']['n_layers'],
                                            recurrent_dropout_probability=conf['dropout'],
                                            layer_dropout_probability=conf['dropout'],
                                            use_highway=conf['encoder'].get('use_highway', True))
            self.encoder = PytorchSeq2SeqWrapper(lstm, stateful=False)
            encoded_input_dim = self.encoder.get_output_dim()
        elif encoder_name == 'project':
            self.encoder = ProjectedEncoder(input_dim,
                                            conf['encoder']['hidden_dim'],
                                            dropout=conf['dropout'])
            encoded_input_dim = self.encoder.get_output_dim()
        elif encoder_name == 'dummy':
            self.encoder = DummyEncoder()
            encoded_input_dim = input_dim
        else:
            raise ValueError('Unknown input encoder: {}'.format(encoder_name))

        if conf["classifier"]["type"].lower() == 'crf':
            self.classify_layer = CRFLayer(encoded_input_dim, n_class, use_cuda)
        else:
            self.classify_layer = ClassifyLayer(encoded_input_dim, n_class, use_cuda)

        self.encode_time = 0
        self.emb_time = 0
        self.classify_time = 0

    def forward(self, inputs: Dict[str, torch.Tensor],
                targets: torch.Tensor):
        # input_: (batch_size, seq_len)
        embedded_input = {}
        for name, fn in self.input_layers.items():
            input_ = inputs[name]
            embedded_input[name] = fn(input_)

        encoded_inputs = []
        for encoder_ in self.input_encoders:
            ordered_names = encoder_.get_ordered_names()
            args_ = {name: embedded_input[name] for name in ordered_names}
            encoded_inputs.append(self.input_dropout(encoder_(args_)))

        encoded_inputs = torch.cat(encoded_inputs, dim=-1)

        lengths = inputs['length']
        mask = get_mask_from_sequence_lengths(lengths, lengths.max())

        encoded_inputs = self.encoder(encoded_inputs, mask)
        # encoded_input_: (batch_size, seq_len, dim)

        encoded_inputs = self.dropout(encoded_inputs)

        output, loss = self.classify_layer(encoded_inputs, targets)

        return output, loss


def eval_model(model: torch.nn.Module,
               batcher: BatcherBase,
               ix2label: Dict[int, str],
               args,
               gold_path: str):
    if args.output is not None:
        path = args.output
        fpo = codecs.open(path, 'w', encoding='utf-8')
    else:
        descriptor, path = tempfile.mkstemp(suffix='.tmp')
        fpo = codecs.getwriter('utf-8')(os.fdopen(descriptor, 'w'))

    model.eval()
    orders, results, sentences = [], [], []
    for inputs, targets, order in batcher.get():
        output, _ = model.forward(inputs, targets)
        for bid in range(len(inputs['text'])):
            length = inputs['length'][bid].item()
            words = inputs['text'][bid]
            tags = [ix2label[output[bid][k].item()] for k in range(length)]
            sentences.append(words)
            results.append(tags)
        orders.extend(order)

    for o in sorted(range(len(results)), key=lambda p: orders[p]):
        words, result = sentences[o], results[o]
        if args.format == 'conllu':
            for i, tag in enumerate(result):
                print('{0}\t{1}\t_\t{2}\t_\t_\t_\t_\t_\t_'.format(i + 1, words[i], tag), file=fpo)
            print(file=fpo)
        elif args.format == 'tag':
            for i, tag in enumerate(result):
                print('{0}\t{1}'.format(tag, words[i]), file=fpo)
            print(file=fpo)
        else:
            print('{0} ||| {1}'.format(' '.join(words), ' '.join(result)))
    fpo.close()
    model.train()
    p = subprocess.Popen([args.script, gold_path, path], stdout=subprocess.PIPE)
    p.wait()
    f = 0
    for line in p.stdout.readlines():
        f = line.strip().split()[-1]
    # os.remove(path)
    return float(f)


def train_model(epoch: int,
                conf: Dict,
                opt: argparse.Namespace,
                model: SeqLabelModel,
                optimizer: torch.optim.Optimizer,
                train_batch: BatcherBase,
                valid_batch: BatcherBase,
                test_batch: BatcherBase,
                ix2label: Dict,
                best_valid: float,
                test_result: float):
    model.train()

    witnessed_improved_valid_result = False
    total_loss, total_tag = 0., 0.
    cnt = 0
    start_time = time.time()

    for inputs, targets, _ in train_batch.get():
        cnt += 1
        model.zero_grad()
        _, loss = model.forward(inputs, targets)

        total_loss += loss.item()
        n_tags = inputs['length'].sum().item()
        total_tag += n_tags
        loss.backward()

        if 'clip_grad' in conf['optimizer']:
            torch.nn.utils.clip_grad_norm_(model.parameters(), conf['optimizer']['clip_grad'])

        optimizer.step()

        if cnt % opt.report_steps == 0:
            logging_str = "| epoch {:3d} | step {:>6d} | lr {:.3g} | " \
                          "ms/batch {:5.2f} | loss {:.4f} |".format(epoch, cnt, optimizer.param_groups[0]['lr'],
                                                                    1000 * (time.time() - start_time) / opt.report_steps,
                                                                    total_loss / total_tag)

            logger.info(logging_str)
            start_time = time.time()

        if cnt % opt.eval_steps == 0:
            valid_result = eval_model(model, valid_batch, ix2label, opt, opt.gold_valid_path)
            logging_str = "| epoch {:3d} | step {:>6d} | lr {:.3g} | loss    {:.4f} | dev  {:.4f} |".format(
                epoch, cnt, optimizer.param_groups[0]['lr'], total_loss / total_tag, valid_result)

            if valid_result > best_valid:
                logging_str = logging_str + ' NEW |'

            logger.info(logging_str)

            if valid_result > best_valid:
                witnessed_improved_valid_result = True
                torch.save(model.state_dict(), os.path.join(opt.model, 'model.pkl'))
                best_valid = valid_result
                if test_batch is not None:
                    test_result = eval_model(model, test_batch, ix2label, opt, opt.gold_test_path)
                    logger.info("| epoch {:3d} | step {:>6d} | lr {:.3g} |                | test {:.4f} |".format(
                        epoch, cnt, optimizer.param_groups[0]['lr'], test_result))

    return best_valid, test_result, witnessed_improved_valid_result


def train():
    cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument('--config', required=True, help='the config file.')
    cmd.add_argument('--format', choices=('conllu', 'tag', 'segment'), default='tag', help='the input format')
    cmd.add_argument('--train_path', required=True, help='the path to the training file.')
    cmd.add_argument('--valid_path', required=True, help='the path to the validation file.')
    cmd.add_argument('--test_path', required=False, help='the path to the testing file.')
    cmd.add_argument('--gold_valid_path', type=str, help='the path to the validation file.')
    cmd.add_argument('--gold_test_path', type=str, help='the path to the testing file.')
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument('--override', help='override embeddings')
    cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
    cmd.add_argument("--report_steps", type=int, default=1024, help='eval every x batches')
    cmd.add_argument("--eval_steps", type=int, help='eval every x batches')
    cmd.add_argument('--output', help='The path to the output file.')
    cmd.add_argument("--script", required=True, help="The path to the evaluation script")

    opt = cmd.parse_args(sys.argv[2:])
    print(opt)

    # setup random
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    conf = json.load(open(opt.config, 'r'))
    if opt.override:
        name, key, value = opt.override.split(':', 2)
        c = conf['input']
        for data in c:
            if data['name'] == name:
                data[key] = value

    if opt.gold_valid_path is None:
        opt.gold_valid_path = opt.valid_path

    if opt.gold_test_path is None and opt.test_path is not None:
        opt.gold_test_path = opt.test_path

    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

    if opt.format == 'conllu':
        read_corpus_fn = read_conllu_dataset
    elif opt.format == 'tag':
        read_corpus_fn = read_tag_dataset
    else:
        read_corpus_fn = read_segment_corpus

    # load raw data
    raw_training_data = read_corpus_fn(opt.train_path)
    raw_valid_data = read_corpus_fn(opt.valid_path)
    if opt.test_path is not None:
        raw_test_data = read_corpus_fn(opt.test_path)
    else:
        raw_test_data = []

    logger.info('we have {0} fields'.format(len(raw_training_data[0][0])))
    logger.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
        len(raw_training_data), len(raw_valid_data), len(raw_test_data)))
    logger.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
        sum([len(seq) for seq in raw_training_data]),
        sum([len(seq) for seq in raw_valid_data]),
        sum([len(seq) for seq in raw_test_data])))

    # create batcher
    input_batches = {}
    for c in conf['input']:
        if c['type'] == 'embeddings':
            batch = WordBatch(c['name'], c['field'], c['min_cut'], not c.get('cased', True), use_cuda)
            if 'pretrained' in c:
                batch.create_dict_from_file(c['pretrained'])
            if c['fixed']:
                if 'pretrained' not in c:
                    logger.warning('it is un-reasonable to use fix embedding without pretraining.')
            else:
                batch.create_dict_from_dataset(raw_training_data)
            input_batches[c['name']] = batch
        elif c['type'] == 'cnn_encoder' or c['type'] == 'lstm_encoder':
            min_char = 1 if c['type'] == 'lstm_encoder' else max([w for w, n in c['filters']])
            batch = CharacterBatch(min_char, c['name'], c['field'],
                                   lower=not c.get('cased', True), use_cuda=use_cuda)
            batch.create_dict_from_dataset(raw_training_data)
            input_batches[c['name']] = batch

    # till now, lexicon is fixed, but embeddings was not
    # keep the order of [textbatcher, lengthbatcher]
    input_batches['text'] = TextBatch(conf['text_field'], use_cuda)
    input_batches['length'] = LengthBatch(use_cuda)

    target_batch = TagBatch(conf['tag_field'], use_cuda)
    target_batch.create_dict_from_dataset(raw_training_data)
    logger.info('tags: {0}'.format(target_batch.mapping))

    n_tags = target_batch.n_tags
    id2label = {ix: label for label, ix in target_batch.mapping.items()}

    training_batcher = BucketBatcher(raw_training_data,
                                     input_batches, target_batch,
                                     opt.batch_size, use_cuda=use_cuda)

    if opt.eval_steps is None or opt.eval_steps > len(raw_training_data):
        opt.eval_steps = training_batcher.num_batches()

    valid_batcher = Batcher(raw_valid_data,
                            input_batches, target_batch, opt.batch_size,
                            shuffle=False, sorting=True, keep_full=True,
                            use_cuda=use_cuda)

    if opt.test_path is not None:
        test_batcher = Batcher(raw_test_data,
                               input_batches, target_batch, opt.batch_size,
                               shuffle=False, sorting=True, keep_full=True,
                               use_cuda=use_cuda)
    else:
        test_batcher = None

    model = SeqLabelModel(conf, input_batches, n_tags, use_cuda)

    logger.info(str(model))
    if use_cuda:
        model = model.cuda()

    c = conf['optimizer']
    optimizer_name = c['type'].lower()
    params = filter(lambda param: param.requires_grad, model.parameters())
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=c.get('lr', 1e-3), betas=c.get('betas', (0.9, 0.999)),
                                     eps=c.get('eps', 1e-8))
    elif optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(params, lr=c.get('lr', 2e-3), betas=c.get('betas', (0.9, 0.999)),
                                       eps=c.get('eps', 1e-8))
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=c.get('lr', 0.01), momentum=c.get('momentum', 0),
                                    nesterov=c.get('nesterov', False))
    elif optimizer_name == 'dense_sparse_adam':
        optimizer = DenseSparseAdam(params, lr=c.get('lr', 1e-3), betas=c.get('betas', (0.9, 0.999)),
                                    eps=c.get('eps', 1e-8))
    else:
        raise ValueError('Unknown optimizer name: {0}'.format(optimizer_name))

    try:
        os.makedirs(opt.model)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    for name, input_batcher in input_batches.items():
        if name == 'text' or name == 'length':
            continue
        with codecs.open(os.path.join(opt.model, '{0}.dic'.format(input_batcher.namespace)), 'w',
                         encoding='utf-8') as fpo:
            for w, i in input_batcher.mapping.items():
                print('{0}\t{1}'.format(w, i), file=fpo)

    with codecs.open(os.path.join(opt.model, 'label.dic'), 'w', encoding='utf-8') as fpo:
        for label, i in target_batch.mapping.items():
            print('{0}\t{1}'.format(label, i), file=fpo)

    new_config_path = os.path.join(opt.model, os.path.basename(opt.config))
    shutil.copy(opt.config, new_config_path)
    opt.config = new_config_path
    json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))

    best_valid, test_result = -1e8, -1e8
    max_decay_times = c.get('max_decay_times', 5)
    max_patience = c.get('max_patience', 10)
    patience = 0
    decay_times = 0
    decay_rate = c.get('decay_rate', 0.5)
    for epoch in range(opt.max_epoch):
        best_valid, test_result, improved = train_model(
            epoch, conf, opt, model, optimizer,
            training_batcher, valid_batcher, test_batcher,
            id2label, best_valid, test_result)

        if not improved:
            patience += 1
            if patience == max_patience:
                decay_times += 1
                if decay_times == max_decay_times:
                    break

                optimizer.param_groups[0]['lr'] *= decay_rate
                patience = 0
                logger.info('Max patience is reached, decay learning rate to '
                            '{0}'.format(optimizer.param_groups[0]['lr']))
        else:
            patience = 0

    logger.info("best_valid_acc: {:.4f}".format(best_valid))
    logger.info("test_acc: {:.4f}".format(test_result))


def test():
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument("--input", help="the path to the test file.")
    cmd.add_argument('--format', choices=('conllu', 'tag', 'segment'), default='tag', help='the input format')
    cmd.add_argument('--output', help='the path to the output file.')
    cmd.add_argument("--model", required=True, help="path to save model")

    opt = cmd.parse_args(sys.argv[2:])
    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

    model_path = opt.model

    model_cmd_opt = dict2namedtuple(json.load(codecs.open(os.path.join(model_path, 'config.json'), 'r',
                                                          encoding='utf-8')))
    conf = json.load(open(model_cmd_opt.config, 'r'))

    torch.manual_seed(model_cmd_opt.seed)
    random.seed(model_cmd_opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        torch.cuda.manual_seed(model_cmd_opt.seed)
        use_cuda = True

    input_batches = {}
    for c in conf['input']:
        if c['type'] == 'embeddings':
            batch = WordBatch(c['name'], c['field'], c['min_cut'], not c.get('cased', True), use_cuda)
            with open(os.path.join(model_path, '{0}.dic'.format(c['name'])), 'r') as fpi:
                mapping = batch.mapping
                for line in fpi:
                    token, i = line.strip().split('\t')
                    mapping[token] = int(i)
            input_batches[c['name']] = batch
        elif c['type'] == 'cnn_encoder' or c['type'] == 'lstm_encoder':
            min_char = 1 if c['type'] == 'lstm_encoder' else max([w for w, n in c['filters']])
            batch = CharacterBatch(min_char, c['name'], c['field'],
                                   lower=not c.get('cased', True), use_cuda=use_cuda)
            with open(os.path.join(model_path, '{0}.dic'.format(c['name'])), 'r') as fpi:
                mapping = batch.mapping
                for line in fpi:
                    token, i = line.strip().split('\t')
                    mapping[token] = int(i)
            input_batches[c['name']] = batch

    input_batches['text'] = TextBatch(conf['text_field'], use_cuda)
    input_batches['length'] = LengthBatch(use_cuda)

    target_batch = TagBatch(conf['tag_field'], use_cuda)

    id2label = {}
    with codecs.open(os.path.join(model_path, 'label.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            token, i = line.strip().split('\t')
            target_batch.mapping[token] = int(i)
            id2label[int(i)] = token
    logger.info('tags: {0}'.format(target_batch.mapping))

    n_tags = len(id2label)
    model = SeqLabelModel(conf, input_batches, n_tags, use_cuda)

    model.load_state_dict(torch.load(os.path.join(model_path, 'model.pkl'), map_location=lambda storage, loc: storage))
    if use_cuda:
        model = model.cuda()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('# of params: {0}'.format(params))

    if opt.format == 'conllu':
        read_corpus_fn = read_conllu_dataset
    elif opt.format == 'tag':
        read_corpus_fn = read_tag_dataset
    else:
        read_corpus_fn = read_segment_corpus

    raw_test_data = read_corpus_fn(opt.input)

    batcher = Batcher(raw_test_data,
                      input_batches, target_batch,
                      model_cmd_opt.batch_size,
                      shuffle=False, sorting=True, keep_full=True,
                      use_cuda=use_cuda)

    if opt.output is not None:
        fpo = codecs.open(opt.output, 'w', encoding='utf-8')
    else:
        fpo = codecs.getwriter('utf-8')(sys.stdout)

    model.eval()
    orders, results, sentences = [], [], []
    for inputs, targets, order in batcher.get():
        output, _ = model.forward(inputs, targets)
        for bid in range(len(inputs['text'])):
            length = inputs['length'][bid].item()
            words = inputs['text'][bid]
            tags = [id2label[output[bid][k].item()] for k in range(length)]
            sentences.append(words)
            results.append(tags)
        orders.extend(order)

    for o in sorted(range(len(results)), key=lambda p: orders[p]):
        words, result = sentences[o], results[o]
        if opt.format == 'conllu':
            for i, tag in enumerate(result):
                print('{0}\t{1}\t_\t{2}\t_\t_\t_\t_\t_\t_'.format(i + 1, words[i], tag), file=fpo)
            print(file=fpo)
        elif opt.format == 'tag':
            for i, tag in enumerate(result):
                print('{0}\t{1}'.format(tag, words[i]), file=fpo)
            print(file=fpo)
        else:
            print('{0} ||| {1}'.format(' '.join(words), ' '.join(result)))
    fpo.close()

    logger.info('Total encoder time: {:.2f}s'.format(model.encode_time))
    logger.info('Total embedding time: {:.2f}s'.format(model.emb_time))
    logger.info('Total classify time: {:.2f}s'.format(model.classify_time))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
