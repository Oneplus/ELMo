#!/usr/bin/env python
from typing import List, Dict, Union
import torch
import gzip
import logging
import collections
import random
import re
import numpy as np
from sklearn.cluster import KMeans
logger = logging.getLogger(__name__)


class VocabBatch(object):
    digit_regex = re.compile(r'\d')

    def __init__(self, lower: bool,
                 normalize_digits: bool,
                 use_cuda: bool):
        self.use_cuda = use_cuda
        self.lower = lower
        self.normalize_digits = normalize_digits
        self.mapping = {'<oov>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}

    def create_dict_from_file(self, filename: str):
        n_entries = 0
        if filename.endswith('.gz'):
            fin = gzip.open(filename, 'rb')
        else:
            fin = open(filename, 'r')
        for line in fin:
            word = line.strip().split()[0]
            self.mapping[word] = len(self.mapping)
            n_entries += 1
        logger.info('+ loaded {0} entries from input'.format(n_entries))
        logger.info('+ current number of entries in mapping is: {0}'.format(len(self.mapping)))

    def create_one_batch(self, raw_dataset: List[List[str]]):
        batch_size, seq_len = len(raw_dataset), max([len(input_) for input_ in raw_dataset])
        forward_batch = torch.LongTensor(batch_size, seq_len).fill_(1)
        backward_batch = torch.LongTensor(batch_size, seq_len).fill_(1)
        for i, raw_data in enumerate(raw_dataset):
            words = raw_data
            if self.lower:
                words = [word.lower() for word in words]
            if self.normalize_digits:
                words = [self.digit_regex.sub('0', word) for word in words]

            for j, word in enumerate(words):
                forward_batch[i, j] = self.mapping.get(words[j + 1] if j + 1 < len(words) else '<eos>', 0)
                backward_batch[i, j] = self.mapping.get((words[j - 1] if j > 0 else '<bos>'), 0)

        if self.use_cuda:
            forward_batch = forward_batch.cuda()
            backward_batch = backward_batch.cuda()
        return forward_batch, backward_batch


class InputBatchBase(object):
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def create_one_batch(self, raw_dataset_: List[List[str]]):
        raise NotImplementedError()


class TextBatch(InputBatchBase):
    def __init__(self, use_cuda: bool):
        super(TextBatch, self).__init__(use_cuda)

    def create_one_batch(self, raw_dataset: List[List[str]]):
        ret = []
        for raw_data in raw_dataset:
            ret.append(tuple([word for word in raw_data]))
        return ret


class LengthBatch(InputBatchBase):
    def __init__(self, use_cuda: bool):
        super(LengthBatch, self).__init__(use_cuda)

    def create_one_batch(self, raw_dataset: List[List[str]]):
        batch_size = len(raw_dataset)
        ret = torch.LongTensor(batch_size).fill_(0)
        for i, raw_data in enumerate(raw_dataset):
            ret[i] = len(raw_data)
        if self.use_cuda:
            ret = ret.cuda()
        return ret


class WordBatch(InputBatchBase):
    def __init__(self, min_cut: int, oov: str, pad: str, lower: bool,
                 use_cuda: bool):
        super(WordBatch, self).__init__(use_cuda)
        self.min_cut = min_cut
        self.oov = oov
        self.pad = pad
        self.mapping = {oov: 0, pad: 1}
        self.lower = lower
        self.n_tokens = 2
        logger.info('{0}'.format(self))
        logger.info('+ min_cut: {0}'.format(self.min_cut))
        logger.info('+ to lower: {0}'.format(self.lower))

    def create_one_batch(self, raw_dataset: List[List[str]]):
        batch_size, seq_len = len(raw_dataset), max([len(input_) for input_ in raw_dataset])
        batch = torch.LongTensor(batch_size, seq_len).fill_(1)
        for i, raw_data in enumerate(raw_dataset):
            for j, word in enumerate(raw_data):
                if self.lower:
                    word = word.lower()
                batch[i, j] = self.mapping.get(word, 0)
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def create_dict_from_dataset(self, raw_dataset: List[List[str]]):
        counter = collections.Counter()
        for raw_data in raw_dataset:
            for word in raw_data:
                if self.lower:
                    word = word.lower()
                counter[word] += 1

        n_entries = 0
        for key in counter:
            if counter[key] < self.min_cut:
                continue
            if key not in self.mapping:
                self.mapping[key] = len(self.mapping)
                n_entries += 1
        logger.info('+ loaded {0} entries from input'.format(n_entries))
        logger.info('+ current number of entries in mapping is: {0}'.format(len(self.mapping)))


class CharacterBatch(InputBatchBase):
    def __init__(self, oov: str, pad: str, eow: str, lower: bool, use_cuda: bool):
        super(CharacterBatch, self).__init__(use_cuda)
        self.oov = oov
        self.pad = pad
        self.eow = eow
        self.mapping = {oov: 0, pad: 1, eow: 2}
        self.lower = lower
        self.n_tokens = 3
        logger.info('{0}'.format(self))

    def create_one_batch(self, raw_dataset: List[List[str]]):
        batch_size = len(raw_dataset)
        seq_len = max([len(input_) for input_ in raw_dataset])
        max_char_len = max([len(word) for raw_data in raw_dataset for word in raw_data])

        batch = torch.LongTensor(batch_size, seq_len, max_char_len).fill_(2)
        lengths = torch.LongTensor(batch_size, seq_len).fill_(1)
        for i, raw_data in enumerate(raw_dataset):
            for j, word in enumerate(raw_data):
                if self.lower:
                    word = word.lower()
                lengths[i, j] = len(word)
                for k, key in enumerate(word):
                    batch[i, j, k] = self.mapping.get(key, 0)

        if self.use_cuda:
            batch = batch.cuda()
            lengths = lengths.cuda()
        return batch, lengths

    def create_dict_from_dataset(self, raw_dataset: List[List[str]]):
        n_entries = 0
        for raw_data in raw_dataset:
            for word in raw_data:
                if self.lower:
                    word = word.lower()
                for key in word:
                    if key not in self.mapping:
                        self.mapping[key] = len(self.mapping)
                        n_entries += 1
        logger.info('+ loaded {0} entries from input'.format(n_entries))
        logger.info('+ current number of entries in mapping is: {0}'.format(len(self.mapping)))


class BatcherBase(object):
    def __init__(self, raw_dataset: List[List[str]],
                 word_batch: Union[WordBatch, None],
                 char_batch: Union[CharacterBatch, None],
                 vocab_batch: VocabBatch,
                 batch_size: int,
                 use_cuda: bool):
        self.raw_dataset = raw_dataset
        self.word_batch = word_batch
        self.char_batch = char_batch
        self.vocab_batch = vocab_batch
        self.length_batch = LengthBatch(vocab_batch.use_cuda)
        self.text_batch = TextBatch(vocab_batch.use_cuda)
        self.batch_size = batch_size
        self.use_cuda = use_cuda

        self.batch_indices = []

    def reset_batch_indices(self):
        raise NotImplementedError

    def get(self):
        self.reset_batch_indices()

        for one_batch_indices in self.batch_indices:
            data_in_one_batch = [self.raw_dataset[i] for i in one_batch_indices]

            if self.word_batch:
                word_inputs = self.word_batch.create_one_batch(data_in_one_batch)
            else:
                word_inputs = None

            if self.char_batch:
                char_inputs = self.char_batch.create_one_batch(data_in_one_batch)
            else:
                char_inputs = None

            lengths = self.length_batch.create_one_batch(data_in_one_batch)
            text = self.text_batch.create_one_batch(data_in_one_batch)
            vocab = self.vocab_batch.create_one_batch(data_in_one_batch)

            yield word_inputs, char_inputs, lengths, text, vocab

    def num_batches(self):
        if len(self.batch_indices) == 0:
            self.reset_batch_indices()

        return len(self.batch_indices)


class Batcher(BatcherBase):
    def __init__(self, raw_dataset: List[List[str]],
                 word_batch: Union[WordBatch, None],
                 char_batch: Union[CharacterBatch, None],
                 vocab_batch: VocabBatch,
                 batch_size: int,
                 shuffle: bool = True,
                 sorting: bool = True,
                 keep_full: bool = False,
                 use_cuda: bool = False):
        super(Batcher, self).__init__(raw_dataset, word_batch, char_batch, vocab_batch,
                                      batch_size, use_cuda)
        self.shuffle = shuffle
        self.sorting = sorting
        self.keep_full = keep_full

    def reset_batch_indices(self):
        n_inputs = len(self.raw_dataset)
        new_orders = list(range(n_inputs))
        if self.shuffle:
            random.shuffle(new_orders)

        if self.sorting:
            new_orders.sort(key=lambda l: len(self.raw_dataset[l]), reverse=True)

        sorted_raw_dataset = [self.raw_dataset[i] for i in new_orders]
        orders = [0] * len(new_orders)
        for i, o in enumerate(new_orders):
            orders[o] = i

        start_id = 0
        self.batch_indices = []
        while start_id < n_inputs:
            end_id = start_id + self.batch_size
            if end_id > n_inputs:
                end_id = n_inputs

            if self.keep_full and len(sorted_raw_dataset[start_id]) != len(sorted_raw_dataset[end_id - 1]):
                end_id = start_id + 1
                while end_id < n_inputs and len(sorted_raw_dataset[end_id]) == len(sorted_raw_dataset[start_id]):
                    end_id += 1

            one_batch_indices = [orders[o] for o in range(start_id, end_id)]
            if len(one_batch_indices) > 0:
                self.batch_indices.append(one_batch_indices)
            start_id = end_id

        if self.shuffle:
            random.shuffle(self.batch_indices)


class BucketBatcher(BatcherBase):
    def __init__(self, raw_dataset: List[List[str]],
                 word_batch: Union[WordBatch, None],
                 char_batch: Union[CharacterBatch, None],
                 vocab_batch: VocabBatch,
                 batch_size: int,
                 n_buckets: int = 10,
                 use_cuda: bool = False):
        super(BucketBatcher, self).__init__(raw_dataset, word_batch, char_batch, vocab_batch,
                                            batch_size, use_cuda)
        lengths = [[len(data)] for data in raw_dataset]
        kmeans = KMeans(n_buckets, random_state=0).fit(np.array(lengths, dtype=np.int))
        self.n_buckets = n_buckets
        self._buckets = []
        for target_label in range(n_buckets):
            self._buckets.append(
                max([lengths[i][0] + 1 for i, label in enumerate(kmeans.labels_) if label == target_label]))
        self._buckets.sort()
        logger.info(self._buckets)

    def reset_batch_indices(self):
        buckets = [[] for _ in self._buckets]

        for data_id, data in enumerate(self.raw_dataset):
            length = len(data)
            for bucket_id, bucket_size in enumerate(self._buckets):
                if length < bucket_size:
                    buckets[bucket_id].append(data_id)
                    break

        random.shuffle(buckets)
        bucket_id = 0
        one_batch_indices = []
        bucket = buckets[bucket_id]
        random.shuffle(bucket)
        data_id_in_bucket = 0

        self.batch_indices = []
        while bucket_id < len(buckets):
            while len(one_batch_indices) < self.batch_size:
                one_batch_indices.append(bucket[data_id_in_bucket])

                data_id_in_bucket += 1
                if data_id_in_bucket == len(bucket):
                    # move to the next bucket
                    bucket_id += 1
                    if bucket_id < len(buckets):
                        data_id_in_bucket = 0
                        bucket = buckets[bucket_id]
                        random.shuffle(bucket)
                    # stop pushing data to the batch, even if this batch is not full.
                    break
            if len(one_batch_indices) > 0:
                self.batch_indices.append(one_batch_indices)
            one_batch_indices = []

        random.shuffle(self.batch_indices)
