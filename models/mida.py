# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-12 17:55:02


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import pandas as pd


class MIDATorch(nn.Module):

    def __init__(self, input_dimension, hidden_layer=3, layer_difference=7,
                 corruption=0.5):
        super(MIDATorch, self).__init__()
        self.train_mode = False
        self.corruption = corruption
        self.encoder = []
        self.decoder = []
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
        x = F.dropout(x, self.corruption, training=self.train_mode)
        for layer in self.encoder:
            x = F.tanh(layer(x))
        for layer in self.decoder:
            x = f.tanh(layer(x))
        return x


class Mida():

    def __init__(self, imputations=5, hidden=3, layer_diff=7, corruption=0.5, epoches=500):
        self.epoches = epoches
        self.imputations = imputations
        self.corruption = corruption
        self.hidden = hidden
        self.layer_diff = layer_diff

    def _default_imputation(self, data):
        defaults = {c: data[c].mean() for c in data.columns}
        return data.fillna(defaults)

    def _tensored(self, data):
        m = data.as_matrix()
        return autograd.Variable(torch.from_numpy(m),
                                 requires_grad=False).float()

    def _untensor(self, tensor):
        return tensor.data.numpy()

    def _train(self, model, data):
        data = self._tensored(data)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adadelta(model.parameters(), rho=0.99)
        for t in range(self.epoches):
            optimizer.zero_grad()
            data_pred = model(data)
            loss = criterion(data_pred, data)
            loss.backward()
            optimizer.step()
        return model

    def _apply(self, model, data):
        data = self_tensored(data)
        data_pred = model(data)
        return self._untensor(data_pred)

    def complete(self, data):
        input_dim = data.shape[1]
        default_imputation = self._default_imputation(data)
        results = []
        for i in range(self.imputations):
            model = MIDATorch(input_dim, self.hidden,
                              self.layer_diff, self.corruption)
            model = self._train(model, default_imputation)
            results.append(self._apply(model, default_imputation))
        return results
