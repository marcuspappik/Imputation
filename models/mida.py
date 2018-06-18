# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-18 18:59:00


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import deque


class MIDATorch(nn.Module):

    def __init__(self, input_dimension, hidden_layer=3, layer_difference=7,
                 corruption=0.5):
        super().__init__()
        self.train_mode = False
        self.corruption = corruption
        encoder = []
        decoder = []
        prev_layer = input_dimension
        for i in range(hidden_layer):
            next_layer = prev_layer + layer_difference
            encoder.append(nn.Linear(prev_layer, next_layer))
            decoder.append(nn.Linear(next_layer, prev_layer))
            prev_layer = next_layer
        decoder.reverse()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

    def train(self, mode=True):
        super().train(mode)
        self.train_mode = mode

    def forward(self, x):
        x = F.dropout(x, self.corruption, training=self.train_mode)
        for layer in self.encoder:
            x = F.tanh(layer(x))
        for layer in self.decoder:
            x = F.tanh(layer(x))
        return x


class Mida():

    def __init__(self, imputations=5, hidden=3, layer_diff=7, corruption=0.5,
                 epoches=500, verbose=False):
        self.epoches = epoches
        self.imputations = imputations
        self.corruption = corruption
        self.hidden = hidden
        self.layer_diff = layer_diff
        self.verbose = verbose

    def _default_imputation(self, data):
        defaults = {c: data[c].mean() for c in data.columns}
        return data.fillna(defaults)

    def _tensored(self, data):
        m = np.matrix(data)
        return autograd.Variable(torch.from_numpy(m),
                                 requires_grad=False).float()

    def _untensor(self, tensor):
        return tensor.data.numpy()

    def _train(self, model, data):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adadelta(list(model.parameters()), rho=0.99)
        train, valid = train_test_split(data, test_size=0.4)
        loss_queue = deque()
        loss_avg = np.inf
        x_train = self._tensored(train)
        x_train = x_train.view(train.shape[0], -1, train.shape[1])
        x_valid = self._tensored(valid)
        x_valid = x_valid.view(valid.shape[0], -1, valid.shape[1])
        for t in range(self.epoches):
            model.train(True)
            optimizer.zero_grad()
            y_train = model(x_train)
            loss_train = criterion(y_train, x_train)
            loss_train.backward()
            optimizer.step()

            model.train(False)
            y_valid = model(x_valid)
            loss_valid = criterion(y_valid, x_valid)

            if self.verbose:
                print(str(t) + ' train: ' + str(loss_train.data[0]) +
                      ' valid: ' + str(loss_valid.data[0]))

            # early stopping according to paper
            loss_queue.append(loss_valid.data[0])
            if loss_valid.data[0] <= 1e-06:
                break
            elif len(loss_queue) >= 5:
                loss_queue.popleft()
                new_avg = np.mean(loss_queue)
                if new_avg >= loss_avg:
                    break
                loss_avg = new_avg

        model.train(False)
        return model

    def _apply(self, model, data):
        x = self._tensored(data)
        x = x.view(data.shape[0], -1, data.shape[1])
        x_pred = model(x)
        data_pred = x_pred.view(data.shape[0], data.shape[1])
        return self._untensor(data_pred)

    def complete(self, data):
        input_dim = data.shape[1]
        default_imputation = self._default_imputation(data)
        scaler = MinMaxScaler()
        default_imputation = scaler.fit_transform(default_imputation)
        results = []
        for i in range(self.imputations):
            model = MIDATorch(input_dim, self.hidden,
                              self.layer_diff, self.corruption)
            model = self._train(model, default_imputation)
            result = self._apply(model, default_imputation)
            result = scaler.inverse_transform(result)
            results.append(result)
        return results
