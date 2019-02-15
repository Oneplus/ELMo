#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, List, Tuple
import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import torch
import shutil
import numpy as np
from bilm.bilm_base import BiLMBase
from bilm.io_util import split_train_and_valid, count_tokens, read_corpus
from bilm.io_util import dict2namedtuple
from bilm.batch import BatcherBase, Batcher, BucketBatcher, WordBatch, CharacterBatch, VocabBatch
from modules.softmax_loss import SoftmaxLoss
from modules.window_sampled_softmax_loss import WindowSampledSoftmaxLoss
from modules.window_sampled_cnn_softmax_loss import WindowSampledCNNSoftmaxLoss
from allennlp.modules.sampled_softmax_loss import SampledSoftmaxLoss
from allennlp.training.optimizers import DenseSparseAdam
from allennlp.training.learning_rate_schedulers import NoamLR, CosineWithRestarts, LearningRateScheduler
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Model(BiLMBase):
    def __init__(self, conf: Dict,
                 word_batch: WordBatch,
                 char_batch: CharacterBatch,
                 n_class: int):
        super(Model, self).__init__(conf, word_batch, char_batch)
        self.dropout = torch.nn.Dropout(conf['dropout'])

        c = conf['classifier']
        classify_layer_name = c['name'].lower()
        if classify_layer_name == 'softmax':
            self.classify_layer = SoftmaxLoss(self.output_dim, n_class)
        elif classify_layer_name == 'cnn_softmax':
            raise NotImplementedError('cnn_softmax is not ready.')
        elif classify_layer_name == 'sampled_softmax':
            sparse = conf['optimizer']['type'].lower() in ('sgd', 'adam', 'dense_sparse_adam')
            self.classify_layer = SampledSoftmaxLoss(n_class, self.output_dim, c['n_samples'], sparse=sparse)
        elif classify_layer_name == 'window_sampled_softmax':
            sparse = conf['optimizer']['type'].lower() in ('sgd', 'adam', 'dense_sparse_adam')
            self.classify_layer = WindowSampledSoftmaxLoss(n_class, self.output_dim, c['n_samples'], sparse=sparse)
        else:
            raise ValueError('Unknown classify_layer: {}'.format(classify_layer_name))

    def forward(self, word_inputs: torch.Tensor,
                chars_inputs: torch.Tensor,
                lengths: torch.Tensor,
                targets: Tuple[torch.Tensor, torch.Tensor]):

        encoded_tokens, _, mask = self._encoding(word_inputs, chars_inputs, lengths)

        n_layers = encoded_tokens.size(0)
        # In both the bilm-tf and allennlp.bidirectional_language_model, there
        # is a dropout layer before the final classification.
        encoded_tokens = self.dropout(encoded_tokens[n_layers - 1])

        forward, backward = encoded_tokens.split(self.output_dim, 2)

        byte_mask = mask.byte()

        selected_forward = forward.masked_select(byte_mask.unsqueeze(-1)).view(-1, self.output_dim)
        selected_backward = backward.masked_select(byte_mask.unsqueeze(-1)).view(-1, self.output_dim)

        selected_forward_targets = targets[0].masked_select(byte_mask)
        selected_backward_targets = targets[1].masked_select(byte_mask)

        return self.classify_layer(selected_forward, selected_forward_targets), \
               self.classify_layer(selected_backward, selected_backward_targets)

    def save_model(self, path, save_classify_layer):
        torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pkl'))
        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pkl'))
        if save_classify_layer:
            torch.save(self.classify_layer.state_dict(), os.path.join(path, 'classifier.pkl'))

    def load_model(self, path):
        self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
        self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'classifier.pkl')))


def eval_model(model: Model,
               valid_batch: BatcherBase):
    model.eval()
    total_forward_loss, total_backward_loss, total_tag = 0.0, 0.0, 0.0
    add_sentence_boundary_ids = model.conf['token_embedder'].get('add_sentence_boundary_ids', False)
    for word_inputs, char_inputs, lengths, text, targets in valid_batch.get():
        forward_loss, backward_loss = model.forward(word_inputs, char_inputs, lengths, targets)
        total_forward_loss += forward_loss.item()
        total_backward_loss += backward_loss.item()
        n_tags = lengths.sum().item()
        # It's because length counts sentence boundary, but loss doesn't
        if add_sentence_boundary_ids:
            n_tags -= (lengths.size(0) * 2)
        total_tag += n_tags

    total_loss = (total_forward_loss + total_backward_loss) * 0.5
    model.train()
    return np.exp(total_loss / total_tag), np.exp(total_forward_loss / total_tag),\
           np.exp(total_backward_loss / total_tag)


def train_model(epoch: int,
                conf: Dict,
                opt,
                model: Model,
                optimizer: torch.optim.Optimizer,
                scheduler: LearningRateScheduler,
                train_batch: BatcherBase,
                valid_batch: BatcherBase,
                test_batch: BatcherBase,
                best_train: float, best_valid: float, test_ppl: float):
    model.train()

    total_loss, total_fwd_loss, total_bwd_loss, total_tag = 0., 0., 0., 0.
    step = 0
    start_time = time.time()

    improved = False
    warmup_step = conf['optimizer']['warmup_step']
    scheduler_name = conf['optimizer'].get('scheduler', 'cosine')
    add_sentence_boundary_ids = conf['token_embedder'].get('add_sentence_boundary_ids', False)

    for word_inputs, char_inputs, lengths, texts, targets in train_batch.get():
        if conf['classifier']['name'].lower() in ('window_sampled_cnn_softmax', 'window_sampled_softmax'):
            negative_sample_targets = []
            vocab = train_batch.vocab_batch
            mapping = train_batch.vocab_batch.mapping
            for words in texts:
                negative_sample_targets.append(mapping.get(vocab.bos))
                negative_sample_targets.extend([mapping.get(word) for word in words])
                negative_sample_targets.append(mapping.get(vocab.eos))
            model.classify_layer.update_negative_samples(negative_sample_targets)

        model.zero_grad()
        forward_loss, backward_loss = model.forward(word_inputs, char_inputs, lengths, targets)
        loss = 0.5 * (forward_loss + backward_loss)
        n_tags = lengths.sum().item()
        # It's because length counts sentence boundary, but loss doesn't
        if add_sentence_boundary_ids:
            n_tags -= (lengths.size(0) * 2)

        total_fwd_loss += forward_loss.item()
        total_bwd_loss += backward_loss.item()
        total_loss += loss.item()
        total_tag += n_tags

        if conf['optimizer'].get('fp16', False):
            optimizer.backward(loss)
        else:
            loss.backward()

        if 'clip_grad' in conf['optimizer']:
            if conf['optimizer'].get('fp16', False):
                optimizer.clip_master_grads(conf['optimizer']['clip_grad'])
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf['optimizer']['clip_grad'])

        optimizer.step()
        step += 1
        global_step = epoch * train_batch.num_batches() + step

        if scheduler_name in ['cosine', 'constant', 'dev_perf']:
            # linear warmup stage
            if global_step < warmup_step:
                curr_lr = conf['optimizer']['lr'] * global_step / warmup_step
                optimizer.param_groups[0]['lr'] = curr_lr
            if scheduler_name == 'cosine':
                scheduler.step_batch(global_step)
        elif scheduler_name == 'noam':
            scheduler.step_batch(global_step)

        if step % opt.report_steps == 0:
            train_ppl, train_fwd_ppl, train_bwd_ppl = [np.exp(loss / total_tag)
                                                       for loss in (total_loss, total_fwd_loss, total_bwd_loss)]
            log_str = "| epoch {:3d} | step {:>6d} | lr {:.3g} | ms/batch {:5.2f} | ppl {:.2f} ({:.2f} " \
                      "{:.2f}) |".format(epoch, step, optimizer.param_groups[0]['lr'],
                                         1000 * (time.time() - start_time) / opt.report_steps,
                                         train_ppl, train_fwd_ppl, train_bwd_ppl)
            logger.info(log_str)
            start_time = time.time()

        if step % opt.eval_steps == 0 or step % train_batch.num_batches() == 0:
            train_ppl, train_fwd_ppl, train_bwd_ppl = [np.exp(loss / total_tag)
                                                       for loss in (total_loss, total_fwd_loss, total_bwd_loss)]
            log_str = "| epoch {:3d} | step {:>6d} | lr {:.3g} |      ppl {:.2f} ({:.2f} {:.2f}) |".format(
                epoch, step, optimizer.param_groups[0]['lr'],
                train_ppl, train_fwd_ppl, train_bwd_ppl)

            if valid_batch is None:
                if scheduler:
                    scheduler.step(train_ppl, epoch)

                if train_ppl < best_train:
                    best_train = train_ppl
                    log_str += ' NEW |'
                    logger.info(log_str)

                    improved = True
                    model.save_model(opt.model, opt.save_classify_layer)
            else:
                logger.info(log_str)
                if train_ppl < best_train:
                    best_train = train_ppl
                valid_ppl, valid_fwd_ppl, valid_bwd_ppl = eval_model(model, valid_batch)
                log_str = "| epoch {:3d} | step {:>6d} | lr {:.3g} |  dev ppl {:.2f} ({:.2f} {:.2f}) |".format(
                    epoch, step, optimizer.param_groups[0]['lr'], valid_ppl, valid_fwd_ppl, valid_bwd_ppl)

                if scheduler:
                    scheduler.step(valid_ppl, epoch)

                if valid_ppl < best_valid:
                    improved = True
                    model.save_model(opt.model, opt.save_classify_layer)
                    best_valid = valid_ppl
                    log_str += ' NEW |'
                    logger.info(log_str)

                    if test_batch is not None:
                        test_ppl, test_fwd_ppl, test_bwd_ppl = eval_model(model, test_batch)
                        log_str = "| epoch {:3d} | step {:>6d} | lr {:.3g} | test ppl {:.2f} ({:.2f} {:.2f}) |".format(
                            epoch, step, optimizer.param_groups[0]['lr'], test_ppl, test_fwd_ppl, test_bwd_ppl)
                        logger.info(log_str)
                else:
                    logger.info(log_str)
    return best_train, best_valid, test_ppl, improved


def train():
    cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    cmd.add_argument('--seed', default=1, type=int, help='The random seed.')
    cmd.add_argument('--gpu', default=-1, type=int, help='Use id of gpu, -1 if cpu.')
    cmd.add_argument('--train_path', required=True, help='The path to the training file.')
    cmd.add_argument('--vocab_path', required=True, help='The path to the vocabulary.')
    cmd.add_argument('--valid_path', help='The path to the development file.')
    cmd.add_argument('--test_path', help='The path to the testing file.')
    cmd.add_argument('--config_path', required=True, help='the path to the config file.')
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
    cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
    cmd.add_argument('--max_sent_len', type=int, default=20, help='maximum sentence length.')
    cmd.add_argument('--bucket', action='store_true', default=False, help='do bucket batching.')
    cmd.add_argument('--save_classify_layer', default=False, action='store_true',
                     help="whether to save the classify layer")
    cmd.add_argument('--valid_size', type=int, default=0, help="size of validation dataset when there's no valid.")
    cmd.add_argument('--eval_steps', type=int, help='evaluate every xx batches.')
    cmd.add_argument('--report_steps', type=int, default=32, help='report every xx batches.')

    opt = cmd.parse_args(sys.argv[2:])

    with open(opt.config_path, 'r') as fin:
        conf = json.load(fin)

    # Dump configurations
    print(opt)
    print(conf)

    # Set seed.
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    if opt.gpu >= 0:
        torch.cuda.set_device(opt.gpu)
        if opt.seed > 0:
            torch.cuda.manual_seed(opt.seed)

    use_cuda = opt.gpu >= 0 and torch.cuda.is_available()
    use_fp16 = False
    if conf['optimizer'].get('fp16', False):
        use_fp16 = True
        if not use_cuda:
            logger.warning('WARNING: fp16 requires --gpu, ignoring fp16 option')
            conf['optimizer']['fp16'] = False
            use_fp16 = False
        else:
            try:
                from apex.fp16_utils import FP16_Optimizer
            except:
                print('WARNING: apex not installed, ignoring --fp16 option')
                conf['optimizer']['fp16'] = False
                use_fp16 = False

    c = conf['token_embedder']
    token_embedder_max_chars = c.get('max_characters_per_token', None)

    # Load training data.
    raw_training_data = read_corpus(opt.train_path, opt.max_sent_len, token_embedder_max_chars)
    logger.info('training instance: {}, training tokens: {}.'.format(len(raw_training_data),
                                                                     count_tokens(raw_training_data)))

    # Load valid data if path is provided, else use 10% of training data as valid data
    if opt.valid_path is not None:
        raw_valid_data = read_corpus(opt.valid_path, opt.max_sent_len, token_embedder_max_chars)
        logger.info('valid instance: {}, valid tokens: {}.'.format(len(raw_valid_data), count_tokens(raw_valid_data)))
    elif opt.valid_size > 0:
        raw_training_data, raw_valid_data = split_train_and_valid(raw_training_data, opt.valid_size)
        logger.info('training instance: {}, training tokens after division: {}.'.format(
            len(raw_training_data), count_tokens(raw_training_data)))
        logger.info('valid instance: {}, valid tokens: {}.'.format(len(raw_valid_data), count_tokens(raw_valid_data)))
    else:
        raw_valid_data = None

    # Load test data if path is provided.
    if opt.test_path is not None:
        raw_test_data = read_corpus(opt.test_path, opt.max_sent_len, token_embedder_max_chars)
        logger.info('testing instance: {}, testing tokens: {}.'.format(len(raw_test_data), count_tokens(raw_test_data)))
    else:
        raw_test_data = None

    # Initialized vocab_batch
    vocab_batch = VocabBatch(lower=not conf['classifier'].get('vocab_cased', True),
                             normalize_digits=conf['classifier'].get('vocab_digit_normalized', False),
                             use_cuda=use_cuda)
    vocab_batch.create_dict_from_file(opt.vocab_path)

    # Word
    if c.get('word_dim', 0) > 0:
        word_batch = WordBatch(min_cut=c.get('word_min_cut', 0),
                               lower=not c.get('word_cased', True),
                               add_sentence_boundary=c.get('add_sentence_boundary_ids', False),
                               use_cuda=use_cuda)
    else:
        word_batch = None

    # Character
    if c.get('char_dim', 0) > 0:
        min_char = max([w for w, _ in c['filters']]) if c['name'] == 'cnn' else 1
        char_batch = CharacterBatch(min_char=min_char,
                                    lower=not c.get('char_cased', True),
                                    add_sentence_boundary=c.get('add_sentence_boundary_ids', False),
                                    add_word_boundary=c.get('add_word_boundary_ids', False),
                                    use_cuda=use_cuda)
        char_batch.create_dict_from_dataset(raw_training_data)
    else:
        char_batch = None

    # Create training batch
    if opt.bucket:
        training_batcher = BucketBatcher(raw_training_data, word_batch, char_batch, vocab_batch, opt.batch_size)
    else:
        training_batcher = Batcher(raw_training_data, word_batch, char_batch, vocab_batch, opt.batch_size,
                                   keep_full=False, sorting=True, shuffle=True)

    # Set up evaluation steps.
    if opt.eval_steps is None:
        opt.eval_steps = training_batcher.num_batches()
    logger.info('Evaluate every {0} batches.'.format(opt.eval_steps))

    # If there is valid, create valid batch.
    if raw_valid_data is not None:
        valid_batcher = Batcher(raw_valid_data, word_batch, char_batch, vocab_batch,
                                opt.batch_size, keep_full=True, sorting=True, shuffle=False)
    else:
        valid_batcher = None

    # If there is test, create test batch.
    if raw_test_data is not None:
        test_batcher = Batcher(raw_test_data, word_batch, char_batch, vocab_batch,
                               opt.batch_size, keep_full=True, sorting=True, shuffle=False)
    else:
        test_batcher = None

    logger.info('vocab size: {0}'.format(len(vocab_batch.mapping)))
    n_classes = len(vocab_batch.mapping)

    model = Model(conf, word_batch, char_batch, n_classes)

    logger.info(str(model))
    if use_cuda:
        model = model.cuda()
        if use_fp16:
            model = model.half()

    # Save meta data of
    try:
        os.makedirs(opt.model)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

    if c.get('char_dim', 0) > 0:
        with codecs.open(os.path.join(opt.model, 'char.dic'), 'w', encoding='utf-8') as fpo:
            for ch, i in char_batch.mapping.items():
                print('{0}\t{1}'.format(ch, i), file=fpo)

    if c.get('word_dim', 0) > 0:
        with codecs.open(os.path.join(opt.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
            for w, i in word_batch.mapping.items():
                print('{0}\t{1}'.format(w, i), file=fpo)

    with codecs.open(os.path.join(opt.model, 'vocab.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in vocab_batch.mapping.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    new_config_path = os.path.join(opt.model, os.path.basename(opt.config_path))
    shutil.copy(opt.config_path, new_config_path)
    opt.config_path = new_config_path
    json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))

    c = conf['optimizer']

    optimizer_name = c['type'].lower()
    params = filter(lambda param: param.requires_grad, model.parameters())
    if optimizer_name == 'adamax':
        optimizer = torch.optim.Adamax(params, lr=c.get('lr', 2e-3), betas=c.get('betas', (0.9, 0.999)),
                                       eps=c.get('eps', 1e-8))
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(params, lr=c.get('lr', 0.01), momentum=c.get('momentum', 0),
                                    nesterov=c.get('nesterov', False))
    elif optimizer_name == 'dense_sparse_adam' or optimizer_name == 'adam':
        optimizer = DenseSparseAdam(params, lr=c.get('lr', 1e-3), betas=c.get('betas', (0.9, 0.999)),
                                    eps=c.get('eps', 1e-8))
    else:
        raise ValueError('Unknown optimizer name: {0}'.format(optimizer_name))

    if use_fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=1.,
                                   dynamic_loss_scale=True,
                                   dynamic_loss_args={'init_scale': 2 ** 16})

    scheduler_name = c.get('scheduler', 'noam')
    if scheduler_name == 'cosine':
        scheduler = CosineWithRestarts(optimizer, c['max_step'], eta_min=c.get('eta_min', 0.0))
    elif scheduler_name == 'dev_perf':
        scheduler = LearningRateScheduler.by_name('reduce_on_plateau')(optimizer, factor=c.get('decay_rate', 0.5),
                                                                       patience=c.get('patience', 5),
                                                                       min_lr=c.get('lr_min', 1e-6))
    elif scheduler_name == 'noam':
        scheduler = NoamLR(optimizer, model_size=c.get('model_size', 512),
                           warmup_steps=c.get('warmup_step', 6000))
    else:
        scheduler = None

    best_train, best_valid, test_result = 1e8, 1e8, 1e8
    for epoch in range(opt.max_epoch):
        best_train, best_valid, test_result, improved = train_model(
            epoch, conf, opt, model, optimizer, scheduler,
            training_batcher, valid_batcher, test_batcher,
            best_train, best_valid, test_result)

    if raw_valid_data is None:
        logger.info("best train ppl: {:.6f}.".format(best_train))
    elif raw_test_data is None:
        logger.info("best train ppl: {:.6f}, best valid ppl: {:.6f}.".format(best_train, best_valid))
    else:
        logger.info("best train ppl: {:.6f}, best valid ppl: {:.6f}, test ppl: {:.6f}.".format(
            best_train, best_valid, test_result))


def test():
    cmd = argparse.ArgumentParser('The testing components of')
    cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
    cmd.add_argument("--input", help="the path to the raw text file.")
    cmd.add_argument("--model", required=True, help="path to save model")
    cmd.add_argument("--batch_size", "--batch", type=int, default=1, help='the batch size.')
    args = cmd.parse_args(sys.argv[2:])

    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))

    with open(args2.config_path, 'r') as fin:
        conf = json.load(fin)

    vocab_batch = VocabBatch(lower=not conf['classifier'].get('vocab_cased', False),
                             normalize_digits=conf['classifier'].get('vocab_digit_normalized', False),
                             use_cuda=use_cuda)
    with codecs.open(os.path.join(args.model, 'vocab.dic'), 'r', encoding='utf-8') as fpi:
        for line in fpi:
            tokens = line.strip().split('\t')
            if len(tokens) == 1:
                tokens.insert(0, '\u3000')
            token, i = tokens
            vocab_batch.mapping[token] = int(i)

    c = conf['token_embedder']
    if c['char_dim'] > 0:
        min_char = max([w for w, _ in c['filters']]) if c['name'] == 'cnn' else 1
        char_batch = CharacterBatch(min_char=min_char,
                                    lower=not c.get('char_cased', True),
                                    add_sentence_boundary=c.get('add_sentence_boundary_ids', False),
                                    add_word_boundary=c.get('add_word_boundary_ids', False),
                                    use_cuda=use_cuda)
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
        word_batch = WordBatch(min_cut=c.get('word_min_cut', 0),
                               lower=not c.get('word_cased', True),
                               add_sentence_boundary=c.get('add_sentence_boundary_ids', False),
                               use_cuda=use_cuda)
        with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                word_batch.mapping[token] = int(i)
    else:
        word_batch = None

    model = Model(conf, word_batch, char_batch, len(vocab_batch.mapping))
    if use_cuda:
        model.cuda()

    logger.info(str(model))
    model.load_model(args.model)
    raw_test_data = read_corpus(args.input, 10000,
                                conf['token_embedder'].get('max_characters_per_token', None))

    test_batcher = Batcher(raw_test_data, word_batch, char_batch, vocab_batch,
                           args.batch_size, keep_full=True, sorting=True, shuffle=False)

    test_result = eval_model(model, test_batcher)

    logger.info("test_ppl={:.6f}".format(test_result))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        test()
    else:
        print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
