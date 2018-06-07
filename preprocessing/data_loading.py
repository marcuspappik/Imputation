# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-07 16:55:34


import numpy as np
import pandas as pd
from math import floor
from sklearn.preprocessing import LabelEncoder


class DataSet():

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target_ = target
        self.missing_data_ = None
        self.mechanism = None
        self.read_data()
        self.encoder = LabelEncoder().fit(self.data[self.target_])

    def construct_path(self):
        return 'data/'+self.dataset+'/'

    def read_data(self):
        path = self.construct_path()+self.dataset+'.csv'
        self.data = pd.read_csv(path)

    def init_missing_data(self, mechanism=None):
        if mechanism is not None:
            self.mechanism = mechanism
        if self.mechanism is None:
            raise(ValueError('no mechanism specified'))
        missing_dim = 4 # floor(self.data.shape[0]*0.3)
        self.mechanism.init_dataset(self.data, missing_dim, [self.target_])

    def ampute_values(self, probability):
        if self.mechanism is None:
            raise(ValueError('no mechanism specified'))
        self.missing_data_ = self.mechanism.ampute(self.data, probability)

    def columns(self, with_target=False):
        if with_target:
            return self.data.columns
        else:
            return [c for c in self.data.columns
                    if c != self.target_]

    def complete_data(self, with_target=False):
        columns = self.columns(with_target)
        return self.data.copy(deep=True)[columns]

    def missing_data(self, with_target=False):
        columns = self.columns(with_target)
        return self.missing_data_.copy(deep=True)[columns]

    def target(self):
        return self.encoder.transform(self.data[self.target_])
