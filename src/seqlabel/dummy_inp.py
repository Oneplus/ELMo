#!/usr/bin/env python
from .encoder_base import EncoderBase


class DummyEncoder(EncoderBase):
    def __init__(self):
        super(DummyEncoder, self).__init__()

    def forward(self, x, *args):
        return x
