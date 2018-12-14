#!/usr/bin/env python
from __future__ import print_function
from __future__ import division 
import sys
gold_dataset = open(sys.argv[1], 'r').read().strip().split('\n\n')
pred_dataset = open(sys.argv[2], 'r').read().strip().split('\n\n')
assert len(gold_dataset) == len(pred_dataset)

n_count = 0
n_correct = 0
for gold_data, pred_data in zip(gold_dataset, pred_dataset):
    gold_lines = gold_data.splitlines()
    pred_lines = pred_data.splitlines()
    assert len(gold_lines) == len(pred_lines)
    for gold_line, pred_line in zip(gold_lines, pred_lines):
        gold_tag = gold_line.split()[0]
        pred_tag = pred_line.split()[0]
        n_count += 1
        if gold_tag == pred_tag:
            n_correct += 1
print(n_correct / n_count)
