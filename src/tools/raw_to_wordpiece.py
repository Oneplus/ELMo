#!/usr/bin/env python
import argparse
from pytorch_pretrained_bert import BertTokenizer


def main():
    cmd = argparse.ArgumentParser()
    cmd.add_argument('-cased', default=False, action='store_true', help='keep case')
    cmd.add_argument('-vocab', required=True, help='the path to the vocab file.')
    cmd.add_argument('filename', help='the path to the file name.')
    opt = cmd.parse_args()

    tokenizer = BertTokenizer.from_pretrained(opt.vocab, do_lower_case=not opt.cased)
    for line in open(opt.filename, 'r'):
        tokens = tokenizer.tokenize(line.strip())
        print(' '.join(tokens))


if __name__ == "__main__":
    main()
