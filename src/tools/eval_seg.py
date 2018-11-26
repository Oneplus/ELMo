#!/usr/bin/env python
from __future__ import print_function
from __future__ import division 
import sys
gold_dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')
pred_dataset = open(sys.argv[2], 'r').read().strip().split('\n\n')
assert len(gold_dataset) == len(pred_dataset)


def get_segments(lines):
    segs = set()
    start = None
    for i, line in enumerate(lines):
        label = line.split()[0].lower()
        if label == 'b' or label == 'o' or label == 's':
            if start is not None:
                segs.add((start, i - 1))
            if label == 'b' or label == 's':
                start = i
            else:
                start = None
    if start is not None:
        segs.add((start, len(lines) - 1))
    return segs


n_pred, n_gold, n_correct = 0, 0, 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
    gold_lines = gold_data.splitlines()
    pred_lines = pred_data.splitlines()
    assert len(gold_lines) == len(pred_lines)
    gold_segs = get_segments(gold_lines)
    pred_segs = get_segments(pred_lines)
    for gold_seg in gold_segs:
        if gold_seg in pred_segs:
            n_correct += 1
    n_pred += len(pred_segs)
    n_gold += len(gold_segs)
p = n_correct / n_pred
r = n_correct / n_gold
print(2 * p * r / (p + r))

