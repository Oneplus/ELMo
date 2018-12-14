#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import sys


def get_words(lines):
  start, end, label = None, None, None
  ret = []
  for i, line in enumerate(lines):
    line = line.strip().split()[0].lower()
    if line.lower() == '-word-piece-':
      if i == 0:
        # error: starts with -word-piece- tag
        return ret
    if line.lower() == '-word-piece-':
      end += 1
    else:
      if start is not None:
        ret.append((start, end, label))
      start, end, label = i, i, line
  if start is not None:
    ret.append((start, end, label))
  return ret


gold_dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')
pred_dataset = open(sys.argv[2], 'r').read().strip().split('\n\n')
assert len(gold_dataset) == len(pred_dataset)

n_correct, n_gold, n_pred = 0, 0, 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
  gold_lines = gold_data.splitlines()
  pred_lines = pred_data.splitlines()
  assert len(gold_lines) == len(pred_lines)
  gold_words = get_words(gold_lines)
  pred_words = get_words(pred_lines)
  for gold_word in gold_words:
    if gold_word in pred_words:
      n_correct += 1
  n_gold += len(gold_words)
  n_pred += len(pred_words)
p = n_correct / n_pred
r = n_correct / n_gold
print(2 * p * r / (p + r))
