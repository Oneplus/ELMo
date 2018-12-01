#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
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
import h5py
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def dict2namedtuple(dic):
  return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_corpus(path):
  """
  read CoNLL format data.

  :param path:
  :return:
  """
  dataset = []
  labels_dataset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for lines in fin.read().strip().split('\n\n'):
      items, labels = [], []
      for line in lines.splitlines():
        label, item = line.split()
        items.append(item)
        labels.append(label)

      dataset.append(items)
      labels_dataset.append(labels)
  return dataset, labels_dataset


def create_one_batch(dim, n_layers, raw_data, raw_labels, lexicon, word2id,
                     oov='<oov>', pad='<pad>', sort=True, use_cuda=False):
  batch_size = len(raw_data)
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(raw_data[l]))

  sorted_raw_data = [raw_data[i] for i in lst]
  sorted_raw_labels = [raw_labels[i] for i in lst]
  sorted_lens = [len(raw_data[i]) for i in lst]
  max_len = max(sorted_lens)

  oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
  batch_x = torch.LongTensor(batch_size, max_len).fill_(pad_id)
  batch_y = torch.LongTensor(batch_size, max_len).fill_(0)
  batch_p = torch.FloatTensor(batch_size, max_len, n_layers, dim).fill_(0)

  assert oov_id is not None and pad_id is not None
  for i, raw_items in enumerate(sorted_raw_data):
    new_raw_items = raw_items
    sentence_key = '\t'.join(new_raw_items).replace('.', '$period$').replace('/', '$backslash$')
    batch_p[i][: sorted_lens[i]] = torch.from_numpy(lexicon[sentence_key][()]).transpose(0, 1)

    for j, x_ij in enumerate(raw_items):
      batch_x[i][j] = word2id.get(x_ij, oov_id)
      batch_y[i][j] = sorted_raw_labels[i][j]

  if use_cuda:
    batch_x = batch_x.cuda()
    batch_p = batch_p.cuda()
    batch_y = batch_y.cuda()

  return batch_x, batch_p, batch_y, sorted_lens


# shuffle training examples and create mini-batches
def create_batches(dim, n_layers, raw_data, raw_labels, lexicon, word2id, batch_size,
                   perm=None, shuffle=True, sort=True, use_cuda=False):
  lst = perm or list(range(len(raw_data)))
  if shuffle:
    random.shuffle(lst)

  if sort:
    lst.sort(key=lambda l: -len(raw_data[l]))

  sorted_raw_data = [raw_data[i] for i in lst]
  sorted_raw_labels = [raw_labels[i] for i in lst]

  sum_len = 0.0
  batches_x, batches_p, batches_y, batches_lens = [], [], [], []
  size = batch_size
  n_batch = (len(raw_data) - 1) // size + 1

  for i in range(n_batch):
    start_id, end_id = i * size, (i + 1) * size
    bx, bp, by, blens = create_one_batch(dim, n_layers,
                                         sorted_raw_data[start_id: end_id],
                                         sorted_raw_labels[start_id: end_id],
                                         lexicon,
                                         word2id,  sort=sort, use_cuda=use_cuda)
    sum_len += sum(blens)
    batches_x.append(bx)
    batches_p.append(bp)
    batches_y.append(by)
    batches_lens.append(blens)

  if sort:
    perm = list(range(n_batch))
    random.shuffle(perm)
    batches_x = [batches_x[i] for i in perm]
    batches_p = [batches_p[i] for i in perm]
    batches_y = [batches_y[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]

  logging.info("{} batches, avg len: {:.1f}".format(n_batch, sum_len / len(raw_data)))
  return batches_x, batches_p, batches_y, batches_lens


class EmbeddingLayer(torch.nn.Module):
  def __init__(self, n_d, word2id, embs=None, fix_emb=True,
               oov='<oov>', pad='<pad>', normalize=True):
    super(EmbeddingLayer, self).__init__()
    if embs is not None:
      embwords, embvecs = embs
      logging.info("{} pre-trained word embeddings loaded.".format(len(word2id)))
      if n_d != len(embvecs[0]):
        logging.warning("[WARNING] n_d ({}) != word vector size ({}). Use {} for embeddings.".format(
          n_d, len(embvecs[0]), len(embvecs[0])))
        n_d = len(embvecs[0])

    self.word2id = word2id
    self.id2word = {i: word for word, i in word2id.items()}
    self.n_V, self.n_d = max(self.id2word), n_d
    self.oovid = word2id[oov]
    self.padid = word2id[pad]
    self.embedding = torch.nn.Embedding(self.n_V + 1, n_d, padding_idx=self.padid)
    self.embedding.weight.data.uniform_(-0.25, 0.25)

    if embs is not None:
      weight = self.embedding.weight
      weight.data[:len(embwords)].copy_(torch.from_numpy(embvecs))
      logging.info("embedding shape: {}".format(weight.size()))

    if normalize:
      weight = self.embedding.weight
      norms = weight.data.norm(2, 1)
      if norms.dim() == 1:
        norms = norms.unsqueeze(1)
      weight.data.div_(norms.expand_as(weight.data))

    if fix_emb:
      self.embedding.weight.requires_grad = False

  def forward(self, input_):
    return self.embedding(input_)


class ClassifyLayer(torch.nn.Module):
  def __init__(self, in_dim, n_class, use_cuda=False):
    super(ClassifyLayer, self).__init__()
    self.num_tags = n_class
    self.use_cuda = use_cuda

    self.hidden2tag = torch.nn.Linear(in_dim, n_class)
    self.logsoftmax = torch.nn.LogSoftmax(dim=2)
    weights = torch.ones(n_class)
    weights[0] = 0
    self.criterion = torch.nn.NLLLoss(weights, size_average=False)

  def forward(self, x, y):
    tag_scores = self.hidden2tag(x)
    if self.training:
      tag_scores = self.logsoftmax(tag_scores)

    _, tag_result = torch.max(tag_scores[:, :, 1:], 2)
    tag_result.add_(1)
    if self.training:
      return tag_result, self.criterion(tag_scores.view(-1, self.num_tags),
                                        torch.autograd.Variable(y).view(-1))
    else:
      return tag_result, torch.FloatTensor([0.0])


class Model(torch.nn.Module):
  def __init__(self, opt, word_emb_layer, dim, n_layers, n_class, use_cuda):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.opt = opt
    self.word_emb_layer = word_emb_layer
    self.proj = torch.nn.Linear(dim, opt.word_dim)
    self.dropout = torch.nn.Dropout(p=opt.dropout)
    self.activation = torch.nn.ReLU()

    if opt.encoder.lower() == 'lstm':
      self.encoder = torch.nn.LSTM(opt.word_dim, opt.hidden_dim,
                                   num_layers=opt.depth, bidirectional=True,
                                   batch_first=True, dropout=opt.dropout)
    else:
      raise ValueError('Unknown encoder name: {}'.format(opt.encoder.lower()))

    if use_cuda:
      self.weights = torch.cuda.FloatTensor(n_layers).fill_(1./n_layers)
    else:
      self.weights = torch.FloatTensor(n_layers).fill_(1./n_layers)
    self.weights = torch.autograd.Variable(self.weights, requires_grad=True)

    self.classify_layer = ClassifyLayer(opt.hidden_dim * 2, n_class, self.use_cuda)

    self.train_time = 0
    self.eval_time = 0
    self.emb_time = 0
    self.classify_time = 0

  def forward(self, x, p, y):
    p = torch.autograd.Variable(p, requires_grad=False)
    p = p.transpose(-2, -1).matmul(self.weights)
    x = self.word_emb_layer(torch.autograd.Variable(x).cuda() if self.use_cuda
                            else torch.autograd.Variable(x))

    x = x + self.dropout(self.activation(self.proj(p)))

    output, hidden = self.encoder(x)
    start_time = time.time()

    output, loss = self.classify_layer.forward(output, y)

    if not self.training:
      self.classify_time += time.time() - start_time
    if self.training:
      loss += self.opt.l2 * self.classify_layer.hidden2tag.weight.data.norm(2)
      # loss += self.opt.l2 * self.merge_inputs.weight.data.norm(2)
    return output, loss


def eval_model(model, valid_payload, ix2label, args, gold_path):
  if args.output is not None:
    path = args.output
    fpo = codecs.open(path, 'w', encoding='utf-8')
  else:
    descriptor, path = tempfile.mkstemp(suffix='.tmp')
    fpo = codecs.getwriter('utf-8')(os.fdopen(descriptor, 'w'))

  valid_x, valid_p, valid_y, valid_lens = valid_payload

  model.eval()
  for x, p, y, lens in zip(valid_x, valid_p, valid_y, valid_lens):
    output, loss = model.forward(x, p, y)
    output_data = output.data
    for bid in range(len(x)):
      for k in range(lens[bid]):
        tag = ix2label[int(output_data[bid][k])]
        print(tag, file=fpo)
      print(file=fpo)
  fpo.close()

  model.train()
  p = subprocess.Popen([args.script, gold_path, path], stdout=subprocess.PIPE)
  p.wait()
  f = 0
  for line in p.stdout.readlines():
    f = line.strip().split()[-1]
  os.remove(path)
  return float(f)


def train_model(epoch, model, optimizer,
                train_payload, valid_payload, test_payload,
                ix2label, best_valid, test_result):
  model.train()
  opt = model.opt

  total_loss, total_tag = 0.0, 0
  cnt = 0
  start_time = time.time()

  train_x, train_p, train_y, train_lens = train_payload

  lst = list(range(len(train_x)))
  random.shuffle(lst)
  train_x = [train_x[l] for l in lst]
  train_p = [train_p[l] for l in lst]
  train_y = [train_y[l] for l in lst]
  train_lens = [train_lens[l] for l in lst]

  for x, p, y, lens in zip(train_x, train_p, train_y, train_lens):
    cnt += 1
    model.zero_grad()
    _, loss = model.forward(x, p, y)
    total_loss += loss.item()
    n_tags = sum(lens)
    total_tag += n_tags
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip_grad)
    optimizer.step()

    if cnt * opt.batch_size % 1024 == 0:
      logging.info("Epoch={} iter={} lr={:.6f} train_ave_loss={:.6f} time={:.2f}s".format(
        epoch, cnt, optimizer.param_groups[0]['lr'],
        1.0 * loss.data[0] / n_tags, time.time() - start_time
      ))
      start_time = time.time()

    if cnt % opt.eval_steps == 0:
      valid_result = eval_model(model, valid_payload, ix2label, opt, opt.gold_valid_path)
      logging.info("Epoch={} iter={} lr={:.6f} train_loss={:.6f} valid_acc={:.6f}".format(
        epoch, cnt, optimizer.param_groups[0]['lr'], total_loss, valid_result))

      if valid_result > best_valid:
        torch.save(model.state_dict(), os.path.join(opt.model, 'model.pkl'))
        logging.info("New record achieved!")
        best_valid = valid_result
        if test is not None:
          test_result = eval_model(model, test_payload, ix2label, opt, opt.gold_test_path)
          logging.info("Epoch={} iter={} lr={:.6f} test_acc={:.6f}".format(
            epoch, cnt, optimizer.param_groups[0]['lr'], test_result))

  return best_valid, test_result


def label_to_index(y, label_to_ix, incremental=True):
  for i in range(len(y)):
    for j in range(len(y[i])):
      if y[i][j] not in label_to_ix and incremental:
        label = label_to_ix[y[i][j]] = len(label_to_ix)
      else:
        label = label_to_ix.get(y[i][j], 0)
      y[i][j] = label


def train():
  cmd = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
  cmd.add_argument('--seed', default=1, type=int, help='the random seed.')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument('--encoder', default='lstm', choices=['lstm'],
                   help='the type of encoder: valid options=[lstm]')
  cmd.add_argument('--classifier', default='vanilla', choices=['vanilla', 'crf'],
                   help='The type of classifier: valid options=[vanilla, crf]')
  cmd.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'],
                   help='the type of optimizer: valid options=[sgd, adam]')
  cmd.add_argument('--train_path', required=True, help='the path to the training file.')
  cmd.add_argument('--valid_path', required=True, help='the path to the validation file.')
  cmd.add_argument('--test_path', required=False, help='the path to the testing file.')
  cmd.add_argument('--lexicon', required=True, help='the path to the hdf5 file.')

  cmd.add_argument('--gold_valid_path', type=str, help='the path to the validation file.')
  cmd.add_argument('--gold_test_path', type=str, help='the path to the testing file.')

  cmd.add_argument('--use_elmo', action='store_true', help='whether to use elmo.')
  cmd.add_argument('--train_elmo_path', type=str, help='the path to the train elmo.')
  cmd.add_argument('--valid_elmo_path', type=str, help='the path to the validation elmo.')
  cmd.add_argument('--test_elmo_path', type=str, help='the path to the testing elmo.')

  cmd.add_argument('--use_cluster', action='store_true', help='whether to use brown cluster.')
  cmd.add_argument('--cluster_path', type=str, help='the path to the brown cluster.')

  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
  cmd.add_argument("--hidden_dim", "--hidden", type=int, default=128, help='the hidden dimension.')
  cmd.add_argument("--max_epoch", type=int, default=100, help='the maximum number of iteration.')
  cmd.add_argument("--word_dim", type=int, default=128, help='the input dimension.')
  cmd.add_argument("--dropout", type=float, default=0.0, help='the dropout rate')
  cmd.add_argument("--depth", type=int, default=2, help='the depth of lstm')

  cmd.add_argument("--eval_steps", type=int, help='eval every x batches')
  cmd.add_argument("--l2", type=float, default=0.00001, help='the l2 decay rate.')
  cmd.add_argument("--lr", type=float, default=0.01, help='the learning rate.')
  cmd.add_argument("--lr_decay", type=float, default=0, help='the learning rate decay.')
  cmd.add_argument("--clip_grad", type=float, default=5, help='the tense of clipped grad.')
  cmd.add_argument('--output', help='The path to the output file.')
  cmd.add_argument("--script", required=True, help="The path to the evaluation script")

  opt = cmd.parse_args(sys.argv[2:])

  print(opt)
  torch.manual_seed(opt.seed)
  random.seed(opt.seed)
  if opt.gpu >= 0:
    torch.cuda.set_device(opt.gpu)
    if opt.seed > 0:
      torch.cuda.manual_seed(opt.seed)

  if opt.gold_valid_path is None:
    opt.gold_valid_path = opt.valid_path

  if opt.gold_test_path is None and opt.test_path is not None:
    opt.gold_test_path = opt.test_path

  use_cuda = opt.gpu >= 0 and torch.cuda.is_available()

  lexicon = h5py.File(opt.lexicon, 'r')
  dim, n_layers = lexicon['#info'][0].item(), lexicon['#info'][1].item()
  logging.info('dim: {}'.format(dim))
  logging.info('n_layers: {}'.format(n_layers))

  raw_training_data, raw_training_labels = read_corpus(opt.train_path)
  raw_valid_data, raw_valid_labels = read_corpus(opt.valid_path)
  if opt.test_path is not None:
    raw_test_data, raw_test_labels = read_corpus(opt.test_path)
  else:
    raw_test_data, raw_test_labels = [], []

  logging.info('training instance: {}, validation instance: {}, test instance: {}.'.format(
    len(raw_training_labels), len(raw_valid_labels), len(raw_test_labels)))
  logging.info('training tokens: {}, validation tokens: {}, test tokens: {}.'.format(
    sum([len(seq) for seq in raw_training_labels]),
    sum([len(seq) for seq in raw_valid_labels]),
    sum([len(seq) for seq in raw_test_labels])))

  label_to_ix = {'<pad>': 0}
  label_to_index(raw_training_labels, label_to_ix)
  label_to_index(raw_valid_labels, label_to_ix, incremental=False)
  label_to_index(raw_test_labels, label_to_ix, incremental=False)

  logging.info('number of tags: {0}'.format(len(label_to_ix)))

  word_lexicon = {}
  for x in raw_training_data:
    for w in x:
      if w not in word_lexicon:
        word_lexicon[w] = len(word_lexicon)

  for special_word in ['<oov>', '<pad>']:
    if special_word not in word_lexicon:
      word_lexicon[special_word] = len(word_lexicon)
  logging.info('training vocab size: {}'.format(len(word_lexicon)))

  word_emb_layer = EmbeddingLayer(opt.word_dim, word_lexicon, fix_emb=False, embs=None)
  logging.info('Word embedding size: {0}'.format(len(word_emb_layer.word2id)))

  n_classes = len(label_to_ix)
  ix2label = {ix: label for label, ix in label_to_ix.items()}

  word2id = word_emb_layer.word2id

  training_payload = create_batches(dim, n_layers, raw_training_data, raw_training_labels,
                                    lexicon, word2id, opt.batch_size, use_cuda=use_cuda)

  if opt.eval_steps is None or opt.eval_steps > len(raw_training_data):
    opt.eval_steps = len(training_payload[0])

  valid_payload = create_batches(dim, n_layers, raw_valid_data, raw_valid_labels,
                                 lexicon, word2id, opt.batch_size, shuffle=False, sort=False,
                                 use_cuda=use_cuda)

  if opt.test_path is not None:
    test_payload = create_batches(dim, n_layers, raw_test_data, raw_test_labels,
                                  lexicon, word2id, opt.batch_size, shuffle=False, sort=False,
                                  use_cuda=use_cuda)
  else:
    test_payload = None

  model = Model(opt, word_emb_layer, dim, n_layers, n_classes, use_cuda)

  logging.info(str(model))
  if use_cuda:
    model = model.cuda()

  need_grad = lambda x: x.requires_grad
  if opt.optimizer.lower() == 'adam':
    optimizer = torch.optim.Adam(filter(need_grad, model.parameters()), lr=opt.lr)
  else:
    optimizer = torch.optim.SGD(filter(need_grad, model.parameters()), lr=opt.lr)

  try:
    os.makedirs(opt.model)
  except OSError as exception:
    if exception.errno != errno.EEXIST:
      raise

  with codecs.open(os.path.join(opt.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
    for w, i in word_emb_layer.word2id.items():
      print('{0}\t{1}'.format(w, i), file=fpo)

  with codecs.open(os.path.join(opt.model, 'label.dic'), 'w', encoding='utf-8') as fpo:
    for label, i in label_to_ix.items():
      print('{0}\t{1}'.format(label, i), file=fpo)

  json.dump(vars(opt), codecs.open(os.path.join(opt.model, 'config.json'), 'w', encoding='utf-8'))
  best_valid, test_result = -1e8, -1e8
  for epoch in range(opt.max_epoch):
    best_valid, test_result = train_model(epoch, model, optimizer,
                                          training_payload, valid_payload, test_payload,
                                          ix2label, best_valid, test_result)
    if opt.lr_decay > 0:
      optimizer.param_groups[0]['lr'] *= opt.lr_decay
    logging.info('Total encoder time: {:.2f}s'.format(model.eval_time / (epoch + 1)))
    logging.info('Total embedding time: {:.2f}s'.format(model.emb_time / (epoch + 1)))
    logging.info('Total classify time: {:.2f}s'.format(model.classify_time / (epoch + 1)))

  logging.info("best_valid_acc: {:.6f}".format(best_valid))
  logging.info("test_acc: {:.6f}".format(test_result))


def test():
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument("--input", help="the path to the test file.")
  cmd.add_argument('--output', help='the path to the output file.')
  cmd.add_argument("--models", required=True, help="path to save model")
  cmd.add_argument("--lexicon", required=True, help='path to the lexicon (hdf5) file.')

  args = cmd.parse_args(sys.argv[2:])

  if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)

  lexicon = h5py.File(args.lexicon, 'r')
  dim, n_layers = lexicon['#info'][0].item(), lexicon['#info'][1].item()
  logging.info('dim: {}'.format(dim))
  logging.info('n_layers: {}'.format(n_layers))

  model_path = args.model

  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(model_path, 'config.json'), 'r', encoding='utf-8')))

  word_lexicon = {}
  word_emb_layers = []
  with codecs.open(os.path.join(model_path, 'word.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      tokens = line.strip().split('\t')
      if len(tokens) == 1:
        tokens.insert(0, '\u3000')
      token, i = tokens
      word_lexicon[token] = int(i)

  word_emb_layer = EmbeddingLayer(args2.word_dim, word_lexicon, fix_emb=False, embs=None)

  logging.info('word embedding size: ' + str(len(word_emb_layers[0].word2id)))

  label2id, id2label = {}, {}
  with codecs.open(os.path.join(model_path, 'label.dic'), 'r', encoding='utf-8') as fpi:
    for line in fpi:
      token, i = line.strip().split('\t')
      label2id[token] = int(i)
      id2label[int(i)] = token
  logging.info('number of labels: {0}'.format(len(label2id)))

  use_cuda = args.gpu >= 0 and torch.cuda.is_available()

  model = Model(args2, word_emb_layer, dim, n_layers, len(label2id), use_cuda)
  model.load_state_dict(torch.load(os.path.join(path, 'model.pkl'), map_location=lambda storage, loc: storage))
  if use_cuda:
    model = model.cuda()

  raw_test_data, raw_test_labels = read_corpus(args.input)
  label_to_index(raw_test_labels, label2id, incremental=False)

  test_data, test_embed, test_labels, test_lens = create_batches(dim, n_layers,
                                                                 raw_test_data, raw_test_labels, lexicon, word_lexicon,
                                                                 1, shuffle=False, sort=False, use_cuda=use_cuda)

  if args.output is not None:
    fpo = codecs.open(args.output, 'w', encoding='utf-8')
  else:
    fpo = codecs.getwriter('utf-8')(sys.stdout)

  model.eval()
  for x, y, lens in zip(test_data, test_labels, test_lens):
    output, loss = model.forward(x, y)
    output_data = output.data
    for bid in range(len(x)):
      for k in range(lens[bid]):
        tag = id2label[int(output_data[bid][k])]
        print(tag, file=fpo)
      print(file=fpo)
  fpo.close()


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train()
  elif len(sys.argv) > 1 and sys.argv[1] == 'test':
    test()
  else:
    print('Usage: {0} [train|test] [options]'.format(sys.argv[0]), file=sys.stderr)
