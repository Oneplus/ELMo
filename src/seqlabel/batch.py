#!/usr/bin/env python
from typing import List, Tuple, Dict
import random
import torch
import logging
import collections
logger = logging.getLogger(__name__)


class TagBatch(object):
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.mapping = {'<pad>': 0}
        self.n_tags = 1

    def create_dict_from_dataset(self, tag_dataset_: List[List[str]]):
        for tags_ in tag_dataset_:
            for tag in tags_:
                if tag not in self.mapping:
                    self.mapping[tag] = len(self.mapping)
        self.n_tags = len(self.mapping)

    def create_one_batch(self, batch_size: int,
                         seq_len: int,
                         tag_dataset_: List[List[str]]):
        batch_ = torch.LongTensor(batch_size, seq_len).fill_(0)
        for i, tags_ in enumerate(tag_dataset_):
            for j, label in enumerate(tags_):
                label = self.mapping.get(label, 0)
                batch_[i, j] = label
        if self.use_cuda:
            batch_ = batch_.cuda()
        return batch_


class InputBatchBase(object):
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def create_one_batch(self, input_dataset_: List[List[str]]):
        raise NotImplementedError()

    def get_field(self):
        raise NotImplementedError()


class InputBatch(InputBatchBase):
    def __init__(self, name: str, field: int, min_cut: int, oov: str, pad: str, lower: bool, use_cuda: bool):
        super(InputBatch, self).__init__(use_cuda)
        self.name = name
        self.field = field
        self.min_cut = min_cut
        self.oov = oov
        self.pad = pad
        self.mapping = {oov: 0, pad: 1}
        self.lower = lower
        self.n_tokens = 2
        logger.info('{0}'.format(self))
        logger.info('+ min_cut: {0}'.format(self.min_cut))
        logger.info('+ field: {0}'.format(self.field))

    def create_one_batch(self, input_dataset_: List[List[str]]):
        batch_size, seq_len = len(input_dataset_), max([len(input_) for input_ in input_dataset_])
        batch = torch.LongTensor(batch_size, seq_len).fill_(1)
        for i, input_ in enumerate(input_dataset_):
            for j, x_ij in enumerate(input_):
                if self.lower:
                    x_ij = x_ij.lower()
                batch[i, j] = self.mapping.get(x_ij, 0)
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def get_field(self):
        return self.field

    def create_dict_from_dataset(self, input_dataset_: List[List[str]]):
        counter = collections.Counter()
        for input_ in input_dataset_:
            for word_ in input_:
                if self.lower:
                    word_ = word_.lower()
                counter[word_] += 1

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


class LengthBatch(InputBatchBase):
    def __init__(self, use_cuda: bool):
        super(LengthBatch, self).__init__(use_cuda)

    def create_one_batch(self, input_dataset_: List[List[str]]):
        batch_size = len(input_dataset_)
        batch = torch.LongTensor(batch_size).fill_(0)
        for i, input_ in enumerate(input_dataset_):
            batch[i] = len(input_)
        if self.use_cuda:
            batch = batch.cuda()
        return batch

    def get_field(self):
        return None


class TextBatch(InputBatchBase):
    def __init__(self, use_cuda: bool):
        super(TextBatch, self).__init__(use_cuda)

    def create_one_batch(self, input_dataset_: List[List[str]]):
        return input_dataset_

    def get_field(self):
        return None


class Batcher(object):
    def __init__(self, n_tags: int,
                 input_dataset_: List[List[List[str]]],
                 segment_dataset_: List[List[Tuple[int, int, str]]],
                 input_batchers_: Dict[str, InputBatchBase],
                 tag_batcher_: TagBatch,
                 batch_size: int,
                 major_field: int = 0,
                 perm=None, shuffle=True, sorting=True, keep_full=False, use_cuda=False):
        # The first dimension of input_dataset_ is field
        self.n_tags = n_tags
        self.input_dataset_ = input_dataset_
        self.segment_dataset_ = segment_dataset_
        self.input_batchers_ = input_batchers_
        self.tag_batcher_ = tag_batcher_

        self.batch_size = batch_size
        self.major_field = major_field
        # sort data according to main field.
        self.perm = perm
        self.shuffle = shuffle
        self.sorting = sorting

        # When keep_full is configured, guareentee that there is no empty element in
        # one batch. The size of one batch various.
        self.keep_full = keep_full
        self.use_cuda = use_cuda

    def get(self):
        input_main_field_ = self.input_dataset_[self.major_field]
        n_inputs_ = len(input_main_field_)
        lst = self.perm or list(range(n_inputs_))
        if self.shuffle:
            random.shuffle(lst)

        if self.sorting:
            lst.sort(key=lambda l: -len(input_main_field_[l]))

        sorted_input_dataset_ = []
        for input_field_ in self.input_dataset_:
            sorted_input_field_ = [input_field_[i] for i in lst]
            sorted_input_dataset_.append(sorted_input_field_)

        sorted_input_main_field_ = sorted_input_dataset_[self.major_field]
        sorted_segment_dataset_ = [self.segment_dataset_[i] for i in lst]

        order = [0] * len(lst)
        for i, l in enumerate(lst):
            order[l] = i

        start_id = 0
        batch_indices = []
        while start_id < n_inputs_:
            end_id = start_id + self.batch_size
            if end_id > n_inputs_:
                end_id = n_inputs_

            if self.keep_full and \
                    len(sorted_input_main_field_[start_id]) != len(sorted_input_main_field_[end_id - 1]):
                end_id = start_id + 1
                while end_id < n_inputs_ and \
                        len(sorted_input_main_field_[end_id]) == len(sorted_input_main_field_[start_id]):
                    end_id += 1
            batch_indices.append((start_id, end_id))
            start_id = end_id

        if self.shuffle:
            random.shuffle(batch_indices)

        for start_id, end_id in batch_indices:
            seq_len = max([len(input_) for input_ in sorted_input_main_field_[start_id: end_id]])
            segment_batch_ = self.tag_batcher_.create_one_batch(end_id - start_id, seq_len,
                                                                sorted_segment_dataset_[start_id: end_id])

            input_batches_ = {}
            for name_, input_batcher_ in self.input_batchers_.items():
                field_ = input_batcher_.get_field()
                if field_ is None:
                    field_ = self.major_field
                input_batches_[name_] = input_batcher_.create_one_batch(
                    sorted_input_dataset_[field_][start_id: end_id])

            yield input_batches_, segment_batch_, order[start_id: end_id]

    def num_batches(self):
        input_main_field_ = self.input_dataset_[self.major_field]
        n_inputs_ = len(input_main_field_)
        return n_inputs_ // self.batch_size + 1
