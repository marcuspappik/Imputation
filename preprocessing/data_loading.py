import numpy as np
import pandas as pd


class DataSet():

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.targte = target
        self.missing_data = None
        self.mechanism = None
        self.read_data()

    def construct_path(self):
        return 'data/'+self.dataset+'/'+self.dataset+'.csv'

    def read_data(self):
        path = self.construct_path()
        self.data = 

    def init_missing_data(self, mechanism=None):
        if mechanism not is None:
            self.mechanism = meachnism
        if self.mechanism is None:
            raise(ValueError('no mechanism specified'))
        self.mechanism.init_dataset(self.data, 4, [self.target])

    def ampute_values(self, probability):
        if self.mechanism is None:
            raise(ValueError('no mechanism specified'))
        self.missing_data = self.mechanism.ampute(self.data, probability)

    def columns(self, with_target=False):
        if with_target:
            return self.data.columns
        else:
            return [c for c in self.data.columns
                    if c != target]

    def complete_data(self, with_target=False):
        columns = self.columns(with_target)
        return data.copy(deep=True)[columns]

    def missing_data(self, with_target=False):
        columns = self.columns(with_target)
        return self.missing_data.copy(deep=True)[columns]
