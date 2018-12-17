#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import sys


def get_words(lines, mask_):
  return [lines[i].strip().split()[0].lower() for i in mask_]


def get_word_mask(lines):
  mask_ = []
  for i, line in enumerate(lines):
    line = line.strip().split()[0].lower()
    if line.lower() == '-word-piece-':
      continue
    mask_.append(i)
  return mask_


gold_dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')
pred_dataset = open(sys.argv[2], 'r').read().strip().split('\n\n')
assert len(gold_dataset) == len(pred_dataset)

n_corr, n_total = 0, 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
  gold_lines = gold_data.splitlines()
  pred_lines = pred_data.splitlines()
  assert len(gold_lines) == len(pred_lines)
  mask = get_word_mask(gold_lines)
  gold_words = get_words(gold_lines, mask)
  pred_words = get_words(pred_lines, mask)
  for gold_word, pred_word in zip(gold_words, pred_words):
    if gold_word == pred_word:
      n_corr += 1
  n_total += len(gold_words)
print(n_corr / n_total)
