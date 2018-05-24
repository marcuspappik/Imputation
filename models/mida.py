import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MIDATorch(nn.Module):

    def __init__(self, input_dimension, hidden_layer=1, layer_difference=7,
                 corruption=0.5):
        super(MIDATorch, self).__init__()
        self.train_mode = False
        self.corruption = corruption
        self.encoder = []
        self.decoder = []
        # build layer
        prev_layer = input_dimension
        for i in range(hidden_layer):
            next_layer = prev_layer + layer_difference
            self.encoder.append(nn.Linear(prev_layer, next_layer))
            self.decoder.append(nn.Linear(next_layer, prev_layer))
        self.decoder.reverse()

    def train(self, mode=True):
        super().train(mode)
        self.train_mode = mode

    def forward(self, data):
        x = data
        if self.train_mode:
            x = F.dropout(data, self.corruption)
        for layer in self.encoder:
            x = F.tanh(layer(x))
        for layer in self.decoder:
            x = f.tanh(layer(x))
        return x
