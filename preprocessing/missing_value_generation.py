# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-07 16:55:37


import numpy as np
import pandas as pd
from random import shuffle
from copy import deepcopy
from scipy.stats.mstats import zscore


def logistic(x, coefficients):
    return np.sum(coefficients*np.exp(x)/(1+np.exp(x)))


def threshold_selection(scores, p):
    n = len(scores)
    values, counts = np.unique(scores, return_counts=True)
    cum_counts = np.cumsum(counts)
    threshold_position = np.argmax(cum_counts >= (1-p)*n)
    threshold = values[threshold_position]
    return (scores >= threshold).astype(bool)


def bernoulli_selection(scores, p):
    n = len(scores)
    adjusted = scores - scores.mean() + p
    adjusted = np.maximum(adjusted, np.zeros(n))
    return np.random.binomial(1, adjusted, n).astype(bool)


class missing_value_generator():

    def init_dataset(self):
        raise NotImplementedError()

    def model(self):
        raise NotImplementedError()

    def ampute(self):
        raise NotImplementedError()


class mcar_generator(missing_value_generator):

    def __init__(self):
        self.missing_columns = []
        self.missing_scores = {}

    def init_dataset(self, data, num_columns, blacklist):
        n = len(data)
        usable_columns = [c for c in data.columns
                          if c not in blacklist]
        shuffle(usable_columns)
        self.missing_columns = usable_columns[:num_columns]
        for c in self.missing_columns:
            self.missing_scores[c] = np.random.uniform(size=n)

    def ampute(self, data, probability):
        result = data.copy(deep=True)
        for c in self.missing_columns:
            rows = (self.missing_scores[c] > 1-probability).astype(bool)
            result[c][rows] = np.nan
        return result

    def name(self):
        return 'mcar'


class mar_generator(missing_value_generator):

    def __init__(self, num_dependent_columns, selection=threshold_selection):
        self.selection = selection
        self.missing_columns = []
        self.missing_scores = {}
        self.num_dependent_columns = num_dependent_columns

    def scoring(self, dependent):
        n = len(dependent)
        coefficients = np.random.rand(dependent.shape[1])
        coefficients = coefficients/coefficients.sum()
        normal_data = pd.DataFrame(columns=dependent.columns, data=zscore(dependent))
        scores = normal_data.apply(lambda x: logistic(x, coefficients), axis=1)
        return scores

    def init_dataset(self, data, num_columns, blacklist):
        usable_columns = [c for c in data.columns
                          if c not in blacklist]
        shuffle(usable_columns)
        self.missing_columns = usable_columns[:num_columns]
        usable_columns = usable_columns[num_columns:]
        for c in self.missing_columns:
            shuffle(usable_columns)
            dependent_columns = usable_columns[:self.num_dependent_columns]
            self.missing_scores[c] = self.scoring(data[dependent_columns])

    def ampute(self, data, probability):
        result = data.copy(deep=True)
        for c in self.missing_columns:
            rows = self.selection(self.missing_scores[c], probability)
            result[c][rows] = np.nan
        return result

    def name(self):
        return 'mar'


class mnar_generator(missing_value_generator):

    def __init__(self, num_dependent_columns,
                 observable_influence=0.5, selection=threshold_selection):
        self.selection = selection
        self.missing_columns = []
        self.missing_scores = {}
        self.observable_influence = observable_influence
        self.num_dependent_columns = num_dependent_columns

    def scoring(self, dependent, feature):
        n = len(dependent)
        coefficients = np.random.rand(dependent.shape[1])
        coefficients = self.observable_influence * coefficients/coefficients.sum()
        coefficients = np.array([1-self.observable_influence] + coefficients.tolist())
        data = pd.concat([feature, dependent], axis=1)
        normal_data = pd.DataFrame(columns=data.columns, data=zscore(data))
        scores = normal_data.apply(lambda x: logistic(x, coefficients), axis=1)
        return scores

    def init_dataset(self, data, num_columns, blacklist):
        usable_columns = [c for c in data.columns
                          if c not in blacklist]
        shuffle(usable_columns)
        self.missing_columns = usable_columns[:num_columns]
        usable_columns = usable_columns[num_columns:]
        for c in self.missing_columns:
            shuffle(usable_columns)
            dependent_columns = usable_columns[:self.num_dependent_columns]
            self.missing_scores[c] = self.scoring(data[dependent_columns], data[c])

    def ampute(self, data, probability):
        result = data.copy(deep=True)
        for c in self.missing_columns:
            rows = self.selection(self.missing_scores[c], probability)
            result[c][rows] = np.nan
        return result

    def name(self):
        return 'mnar'
