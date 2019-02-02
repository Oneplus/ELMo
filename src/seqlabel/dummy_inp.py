#!/usr/bin/env python
from .input_encoder_base import InputEncoderBase


class DummyInputEncoder(InputEncoderBase):
    def __init__(self):
        super(DummyInputEncoder, self).__init__()

    def forward(self, x):
        return x
