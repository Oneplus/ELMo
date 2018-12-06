#!/usr/bin/env python
import random
import torch


def create_one_batch(x, word2id, char2id, config, oov='<oov>', pad='<pad>', sort=True):
  """

  :param x:
  :param word2id: dict
  :param char2id: dict
  :param config:
  :param oov:
  :param pad:
  :param sort:
  :return:
  """
  batch_size = len(x)
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  lens = [len(x[i]) for i in lst]
  max_len = max(lens)

  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_w[i][j] = word2id.get(x_ij, oov_id)
  else:
    batch_w = None

  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = char2id.get('<eow>', None), char2id.get('<bow>', None), char2id.get(oov, None), char2id.get(pad, None)

    assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

    if config['token_embedder']['name'].lower() == 'cnn':
      max_chars = config['token_embedder']['max_characters_per_token']
      assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
    elif config['token_embedder']['name'].lower() == 'lstm':
      max_chars = max([len(w) for i in lst for w in x[i]]) + 2  # counting the <bow> and <eow>

    batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_c[i][j][0] = bow_id
        if x_ij == '<bos>' or x_ij == '<eos>':
          batch_c[i][j][1] = char2id.get(x_ij)
          batch_c[i][j][2] = eow_id
        else:
          for k, c in enumerate(x_ij):
            batch_c[i][j][k + 1] = char2id.get(c, oov_id)
          batch_c[i][j][len(x_ij) + 1] = eow_id
  else:
    batch_c = None

  masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

  for i, x_i in enumerate(x):
    for j in range(len(x_i)):
      masks[0][i][j] = 1
      if j + 1 < len(x_i):
        masks[1].append(i * max_len + j)
      if j > 0:
        masks[2].append(i * max_len + j)

  assert len(masks[1]) <= batch_size * max_len
  assert len(masks[2]) <= batch_size * max_len

  masks[1] = torch.LongTensor(masks[1])
  masks[2] = torch.LongTensor(masks[2])

  return batch_w, batch_c, lens, masks


class Batcher(object):
  def __init__(self, data, batch_size, word2id, char2id, config, perm=None, shuffle=True, sort=True):
    self.batch_size = batch_size
    self.word2id = word2id
    self.char2id = char2id
    self.config = config
    self.perm = perm
    self.shuffle = shuffle
    self.sort = sort

    lst = perm or list(range(len(data)))
    if shuffle:
      random.shuffle(lst)

    if sort:
      lst.sort(key=lambda l: -len(data[l]))

    self.sorted_data = [data[i] for i in lst]
    self.nbatch = (len(data) - 1) // batch_size + 1

  def get(self):
    batch_ids = list(range(self.nbatch))
    if self.shuffle:
      random.shuffle(batch_ids)

    for i in batch_ids:
      start_id, end_id = i * self.batch_size, (i + 1) * self.batch_size
      bw, bc, blens, bmasks = create_one_batch(self.sorted_data[start_id: end_id], self.word2id, self.char2id,
                                               self.config, sort=self.sort)
      yield bw, bc, blens, bmasks

  def num_batches(self):
    return self.nbatch
