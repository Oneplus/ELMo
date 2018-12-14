#!/usr/bin/env python
# The tag conversion scheme is inspired by
# https://github.com/kyzhouhzau/BERT-NER/blob/master/BERT_NER.py
import argparse
from pytorch_pretrained_bert import BertTokenizer


def main():
  cmd = argparse.ArgumentParser()
  cmd.add_argument('-vocab', required=True, help='the path to the vocab file.')
  cmd.add_argument('filename', help='The path to the input tag file.')
  opt = cmd.parse_args()

  tokenizer = BertTokenizer.from_pretrained(opt.vocab)

  dataset = open(opt.filename, 'r').read().strip().split('\n\n')
  for data in dataset:
    lines = data.splitlines()
    for line in lines:
      tag, token = line.strip().split()
      pieces = tokenizer.tokenize(token)
      for i, piece in enumerate(range(len(pieces))):
        if i == 0:
          print('{0}\t{1}'.format(tag, piece))
        else:
          print('{0}\t{1}'.format('-word-piece-', piece))
    print()


if __name__ == "__main__":
  main()
