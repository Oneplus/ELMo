from typing import List, Dict
import random
import codecs
import collections


def split_train_and_valid(data, valid_size):
    valid_size = min(valid_size, len(data) // 10)
    random.shuffle(data)
    return data[valid_size:], data[:valid_size]


def count_tokens(raw_data):
    return sum([len(s) - 1 for s in raw_data])


def break_sentences(sentences: List[List[str]], max_sent_len: int) -> List[List[str]]:
    # For example, for a sentence with 70 words, supposing the the `max_sent_len'
    # is 30, break it into 3 sentences.
    ret = []
    for sentence in sentences:
        while len(sentence) > max_sent_len:
            ret.append(sentence[:max_sent_len])
            sentence = sentence[max_sent_len:]
        if 0 < len(sentence) <= max_sent_len:
            ret.append(sentence)
    return ret


def read_corpus(path: str, max_sent_len: int = 20, max_chars: int = None):
    dataset = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            item = []
            for token in line.strip().split():
                if max_chars is not None and len(token) + 2 > max_chars:
                    token = token[:max_chars - 2]
                item.append(token)
            dataset.append(item)
    dataset = break_sentences(dataset, max_sent_len)
    return dataset


def read_corpus_with_original_text(path: str, max_sent_len: int = 20, max_chars: int = None):
    dataset, raw_dataset = [], []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            item, raw_item = [], []
            for token in line.strip().split():
                new_token = token
                if max_chars is not None and len(new_token) + 2 > max_chars:
                    new_token = new_token[:max_chars - 2]
                item.append(new_token)
                raw_item.append(token)
            dataset.append(item)
            raw_dataset.append(raw_item)
    dataset = break_sentences(dataset, max_sent_len)
    return dataset, raw_dataset


def dict2namedtuple(dic: Dict):
    return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_conll_corpus(path: str, max_chars: int = None):
    # read text in CoNLL-U format.
    dataset = []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for payload in fin.read().strip().split('\n\n'):
            item = []
            lines = payload.splitlines()
            body = [line for line in lines if not line.startswith('#')]
            for line in body:
                fields = line.split('\t')
                num, token = fields[0], fields[1]
                if '-' in num or '.' in num:
                    continue
                if max_chars is not None and len(token) + 2 > max_chars:
                    token = token[:max_chars - 2]
                item.append(token)

            dataset.append(item)
    return dataset


def read_conll_corpus_with_original_text(path: str, max_chars: int = None):
    # read text in CoNLL-U format.
    dataset, raw_dataset = [], []
    with codecs.open(path, 'r', encoding='utf-8') as fin:
        for payload in fin.read().strip().split('\n\n'):
            item, raw_item = [], []
            lines = payload.splitlines()
            body = [line for line in lines if not line.startswith('#')]
            for line in body:
                fields = line.split('\t')
                num, token = fields[0], fields[1]
                if '-' in num or '.' in num:
                    continue
                if max_chars is not None and len(token) + 2 > max_chars:
                    token = token[:max_chars - 2]
                item.append(token)
                raw_item.append(fields[1])

            dataset.append(item)
            raw_dataset.append(raw_item)
    return dataset, raw_dataset
