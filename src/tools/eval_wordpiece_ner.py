#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import sys
gold_dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')
pred_dataset = open(sys.argv[2], 'r').read().strip().split('\n\n')
assert len(gold_dataset) == len(pred_dataset)


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
      start, end, tag = i, i, line
  if start is not None:
    ret.append((start, end, label))
  return ret


def get_segments(words):
  segs = set()
  start, tag = None, None
  for new_start, new_end, label in words:
    if label.startswith('b-') or label == 'o':
      if start is not None:
        segs.add((start, new_start - 1, tag))
      if label.startswith('b-'):
        start, tag = new_start, label.split('-', 1)[1]
      else:
        start, tag = None, None
  if start is not None:
    segs.add((start, new_end, tag))
  return segs


n_pred, n_gold, n_correct = 0, 0, 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
  gold_lines = gold_data.splitlines()
  pred_lines = pred_data.splitlines()
  assert len(gold_lines) == len(pred_lines)
  gold_words = get_words(gold_lines)
  pred_words = get_words(pred_lines)
  gold_segs = get_segments(gold_words)
  pred_segs = get_segments(pred_words)
  for gold_seg in gold_segs:
    if gold_seg in pred_segs:
      n_correct += 1
  n_pred += len(pred_segs)
  n_gold += len(gold_segs)
p = n_correct / n_pred
r = n_correct / n_gold
print(2 * p * r / (p + r))

