#!/usr/bin/env python
from typing import List, Tuple, Dict, Union, Any
import random
import torch
import logging
import collections
import numpy as np
from sklearn.cluster import KMeans
logger = logging.getLogger(__name__)
Item = Union[List[str], Tuple[str]]


class SpecialToken(object):
    oov = '<oov>'
    pad = '<pad>'
    bos = '<S>'
    eos = '</S>'
    bow = '<W>'
    eow = '</W>'
    oov_id = 0
    pad_id = 1


class TagBatch(object):
    def __init__(self, field: int, use_cuda: int):
        self.use_cuda = use_cuda
        self.mapping = {'<pad>': 0}
        self.field = field
        self.n_tags = 1

    def create_dict_from_dataset(self, dataset: List[List[Tuple[str]]]):
        for data in dataset:
            for fields in data:
                tag = fields[self.field]
                if tag not in self.mapping:
                    self.mapping[tag] = len(self.mapping)
        self.n_tags = len(self.mapping)

    def create_one_batch(self, dataset: List[List[Tuple[str]]]):
        batch_size = len(dataset)
        seq_len = max([len(data) for data in dataset])
        batch = torch.LongTensor(batch_size, seq_len).fill_(0)
        for i, data in enumerate(dataset):
            for j, fields in enumerate(data):
                tag = fields[self.field]
                tag = self.mapping.get(tag, 0)
                batch[i, j] = tag
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def numericalize(self, input_data: List[Tuple[str]]):
        seq_len = len(input_data)
        ret = torch.LongTensor(seq_len).fill_(0)
        for i, fields in enumerate(input_data):
            tag = fields[self.field]
            ret[i] = self.mapping.get(tag, 0)
        return ret

    def fusion(self, package: List[torch.Tensor]):
        batch_size = len(package)
        seq_len = max([data.size(0) for data in package])
        batch = package[0].new_zeros(batch_size, seq_len)
        for i, data in enumerate(package):
            batch[i, : data.size(0)] = data
        if self.use_cuda:
            batch = batch.cuda()
        return batch


class InputBatchBase(object):
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def create_one_batch(self, input_dataset_: List[List[Tuple[str]]]):
        raise NotImplementedError()

    def numericalize(self, input_data: List[Tuple[str]]):
        raise NotImplementedError()

    def fusion(self, package: List[Any]):
        raise NotImplementedError()

    def get_field(self):
        raise NotImplementedError()


class WordBatch(InputBatchBase, SpecialToken):
    def __init__(self, namespace: str, field: int, min_cut: int, lower: bool, use_cuda: bool):
        super(WordBatch, self).__init__(use_cuda)
        self.namespace = namespace
        self.field = field
        self.min_cut = min_cut
        self.mapping = {self.oov: self.oov_id, self.pad: self.pad_id}
        self.lower = lower
        self.n_tokens = 2
        logger.info('{0}'.format(self))
        logger.info('+ min_cut: {0}'.format(self.min_cut))
        logger.info('+ field: {0}'.format(self.field))

    def create_one_batch(self, dataset: List[List[Tuple[str]]]):
        batch_size = len(dataset)
        seq_len = max([len(data) for data in dataset])
        batch = torch.LongTensor(batch_size, seq_len).fill_(self.pad_id)
        for i, data in enumerate(dataset):
            for j, fields in enumerate(data):
                word = fields[self.field]
                if self.lower:
                    word = word.lower()
                batch[i, j] = self.mapping.get(word, self.oov_id)
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def numericalize(self, input_data: List[Tuple[str]]):
        seq_len = len(input_data)
        ret = torch.LongTensor(seq_len).fill_(self.pad_id)
        for i, fields in enumerate(input_data):
            word = fields[self.field]
            if self.lower:
                word = word.lower()
            ret[i] = self.mapping.get(word, self.oov_id)
        return ret

    def fusion(self, package: List[torch.Tensor]):
        batch_size = len(package)
        seq_len = max([data.size(0) for data in package])
        batch = package[0].new_zeros(batch_size, seq_len).fill_(self.pad_id)
        for i, data in enumerate(package):
            batch[i, : data.size(0)] = data
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def get_field(self):
        return self.field

    def create_dict_from_dataset(self, dataset: List[List[Tuple[str]]]):
        counter = collections.Counter()
        for data in dataset:
            for fields in data:
                word = fields[self.field]
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

    def create_dict_from_file(self, filename: str, has_header: bool = True):
        n_entries = 0
        with open(filename) as fin:
            if has_header:
                fin.readline()
            for line in fin:
                word = line.strip().split()[0]
                self.mapping[word] = len(self.mapping)
                n_entries += 1
        logger.info('+ loaded {0} entries from file: {1}'.format(n_entries, filename))
        logger.info('+ current number of entries in mapping is: {0}'.format(len(self.mapping)))


class CharacterBatch(InputBatchBase, SpecialToken):
    def __init__(self, min_char: int, namespace: str, field: int, lower: bool, use_cuda: bool):
        super(CharacterBatch, self).__init__(use_cuda)
        self.namespace = namespace
        self.min_char = min_char
        self.field = field
        self.mapping = {self.oov: self.oov_id, self.pad: self.pad_id}
        self.lower = lower
        self.n_tokens = 3
        logger.info('{0}'.format(self))
        logger.info('+ field: {0}'.format(self.field))

    def create_one_batch(self, raw_dataset: List[List[Item]]):
        batch_size = len(raw_dataset)
        seq_len = max([len(input_) for input_ in raw_dataset])
        max_char_len = max([len(fields[self.field]) for raw_data_ in raw_dataset for fields in raw_data_])
        max_char_len = max(self.min_char, max_char_len)

        batch = torch.LongTensor(batch_size, seq_len, max_char_len).fill_(self.pad_id)
        lengths = torch.LongTensor(batch_size, seq_len).fill_(1)
        for i, raw_data_ in enumerate(raw_dataset):
            for j, fields in enumerate(raw_data_):
                field = fields[self.field]
                if self.lower:
                    field = field.lower()
                lengths[i, j] = len(field)
                for k, key in enumerate(field):
                    batch[i, j, k] = self.mapping.get(key, self.oov_id)
        if self.use_cuda:
            batch = batch.cuda()
            lengths = lengths.cuda()
        return batch, lengths

    def numericalize(self, input_data: List[Tuple[str]]):
        seq_len = len(input_data)
        max_char_len = max([len(fields[self.field]) for fields in input_data])
        ret = torch.LongTensor(seq_len, max_char_len).fill_(self.pad_id)
        lengths = torch.LongTensor(seq_len).fill_(1)
        for i, fields in enumerate(input_data):
            field = fields[self.field]
            if self.lower:
                field = field.lower()
            lengths[i] = len(field)
            for j, key in enumerate(field):
                ret[i, j] = self.mapping.get(key, self.oov_id)
        return ret, lengths

    def fusion(self, package: List[Tuple[torch.Tensor, torch.Tensor]]):
        batch_size = len(package)
        seq_len = max([data.size(0) for data, _ in package])
        max_char_len = max([data.size(1) for data, _ in package])

        batch = torch.LongTensor(batch_size, seq_len, max_char_len).fill_(self.pad_id)
        lengths = torch.LongTensor(batch_size, seq_len).fill_(1)

        for i, (data, one_length) in enumerate(package):
            batch[i, :data.size(0), : data.size(1)] = data
            lengths[i, :one_length.size(0)] = one_length
        if self.use_cuda:
            batch = batch.cuda()
            lengths = lengths.cuda()
        return batch, lengths

    def get_field(self):
        return self.field

    def create_dict_from_dataset(self, raw_dataset_: List[List[Item]]):
        n_entries = 0
        for raw_data_ in raw_dataset_:
            for fields in raw_data_:
                word_ = fields[self.field]
                if self.lower:
                    word_ = word_.lower()
                for key in word_:
                    if key not in self.mapping:
                        self.mapping[key] = len(self.mapping)
                        n_entries += 1
        logger.info('+ loaded {0} entries from input'.format(n_entries))
        logger.info('+ current number of entries in mapping is: {0}'.format(len(self.mapping)))


class LengthBatch(InputBatchBase):
    def __init__(self, use_cuda: bool):
        super(LengthBatch, self).__init__(use_cuda)

    def create_one_batch(self, dataset: List[List[Tuple[str]]]):
        batch_size = len(dataset)
        batch = torch.LongTensor(batch_size).fill_(0)
        for i, data in enumerate(dataset):
            batch[i] = len(data)
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def numericalize(self, input_data: List[Tuple[str]]):
        return len(input_data)

    def fusion(self, package: List[int]):
        batch = torch.LongTensor(package)
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def get_field(self):
        return None


class TextBatch(InputBatchBase):
    def __init__(self, field: int, use_cuda: bool):
        super(TextBatch, self).__init__(use_cuda)
        self.field = field

    def create_one_batch(self, dataset: List[List[List[str]]]):
        batch_size = len(dataset)
        batch = [[] for _ in range(batch_size)]
        for i, data in enumerate(dataset):
            for j, fields in enumerate(data):
                word = fields[self.field]
                batch[i].append(word)
        return batch

    def numericalize(self, input_data: List[Tuple[str]]):
        return [fields[self.field] for fields in input_data]

    def fusion(self, package: List[List[Tuple[str]]]):
        return package

    def get_field(self):
        return None


class BatcherBase(object):
    def __init__(self, raw_dataset: List[List[Tuple[str]]],
                 input_batchers: Dict[str, InputBatchBase],
                 tag_batchers: TagBatch,
                 batch_size: int,
                 use_cuda: bool,
                 caching: bool = False):
        self.raw_dataset = raw_dataset
        self.input_batches = input_batchers
        self.tag_batcher = tag_batchers
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.caching = caching

        if caching:
            self.cached_inputs, self.cached_targets = self.cache_dataset()

        self.batch_indices = []

    def reset_batch_indices(self):
        raise NotImplementedError

    def cache_dataset(self):
        cached_inputs = []
        cached_targets = []
        for data in self.raw_dataset:
            cached_targets.append(self.tag_batcher.numericalize(data))
            inputs = {}
            for name, input_batch in self.input_batches.items():
                inputs[name] = input_batch.numericalize(data)
            cached_inputs.append(inputs)
        return cached_inputs, cached_targets

    def get(self):
        self.reset_batch_indices()

        for one_batch_indices in self.batch_indices:
            if self.caching:
                targets = self.tag_batcher.fusion([self.cached_targets[i] for i in one_batch_indices])
                inputs = {}
                for name, input_batch in self.input_batches.items():
                    inputs[name] = input_batch.fusion([self.cached_inputs[i][name] for i in one_batch_indices])
            else:
                data_in_one_batch = [self.raw_dataset[i] for i in one_batch_indices]

                targets = self.tag_batcher.create_one_batch(data_in_one_batch)
                inputs = {}
                for name, input_batch in self.input_batches.items():
                    inputs[name] = input_batch.create_one_batch(data_in_one_batch)

            yield inputs, targets, one_batch_indices

    def num_batches(self):
        if len(self.batch_indices) == 0:
            self.reset_batch_indices()

        return len(self.batch_indices)


class Batcher(BatcherBase):
    def __init__(self, raw_dataset: List[List[Tuple[str]]],
                 input_batchers: Dict[str, InputBatchBase],
                 tag_batcher: TagBatch,
                 batch_size: int,
                 shuffle: bool = True,
                 sorting: bool = True,
                 keep_full: bool = False,
                 use_cuda: bool = False):
        super(Batcher, self).__init__(raw_dataset, input_batchers, tag_batcher,
                                      batch_size, use_cuda, caching=True)
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

            one_batch_indices = [new_orders[o] for o in range(start_id, end_id)]
            if len(one_batch_indices) > 0:
                self.batch_indices.append(one_batch_indices)
            start_id = end_id

        if self.shuffle:
            random.shuffle(self.batch_indices)


class BucketBatcher(BatcherBase):
    def __init__(self, raw_dataset: List[List[Tuple[str]]],
                 input_batchers: Dict[str, InputBatchBase],
                 tag_batcher: TagBatch,
                 batch_size: int,
                 n_buckets: int = 10,
                 use_cuda: bool = False):
        super(BucketBatcher, self).__init__(raw_dataset, input_batchers, tag_batcher,
                                            batch_size, use_cuda, caching=True)
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
