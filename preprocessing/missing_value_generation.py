import numpy as np
import pandas as pd
from random import shuffle

def logistic(x, coefficients):
    return np.sum(coefficients*np.exp(x)/(1+np.exp(x)))


class missing_value_generator():

    def ampute(self):
        raise NotImplementedError()


class mcar_generator(missing_value_generator):

    def __init__(self, num_columns, probability, blacklist):
        self.blacklist = blacklist
        self.prob = probability
        self.num_columns = num_columns
        self.missing_columns = []
        self.missing_rows = {}

    def ampute(self, data):
        result = data.copy(deep=True)
        n = len(data)
        usable_columns = [c for c in data.columns
                          if c not in self.blacklist]
        shuffle(usable_columns)
        self.missing_columns = usable_columns[:self.num_columns]
        for c in self.missing_columns:
            rows = np.random.binomial(1, self.prob, n).astype('bool')
            self.missing_rows[c] = rows
            result[c][rows] = np.nan
        return result


class mar_generator(missing_value_generator):

    def __init__(self, num_columns, probability, blacklist, dependent_columns):
        self.blacklist = blacklist
        self.prob = probability
        self.num_columns = num_columns
        self.num_dependent_columns = dependent_columns
        self.missing_columns = []
        self.missing_rows = {}

    def model(self, data):
        n = len(data)
        coefficients = np.random.rand(data.shape[1])
        coefficients = coefficients/coefficients.sum()
        scores = data.apply(lambda x: logistic(x, coefficients), axis=1)
        scores = scores + self.prob - scores.mean()
        scores = np.maximum(scores, np.zeros(len(scores)))
        return np.random.binomial(1, scores, n).astype('bool')

    def ampute(self, data):
        result = data.copy(deep=True)
        usable_columns = [c for c in data.columns
                          if c not in self.blacklist]
        shuffle(usable_columns)
        self.missing_columns = usable_columns[:self.num_columns]
        usable_columns = usable_columns[self.num_columns:]
        for c in self.missing_columns:
            shuffle(usable_columns)
            dependent_columns = usable_columns[:self.num_dependent_columns]
            rows = self.model(data[dependent_columns])
            self.missing_rows[c] = rows
            result[c][rows] = np.nan
        return result


class mnar_generator(missing_value_generator):

    def __init__(self, num_columns, probability, blacklist,
                 dependent_columns, observable_influence):
        self.blacklist = blacklist
        self.prob = probability
        self.num_columns = num_columns
        self.num_dependent_columns = dependent_columns
        self.observable_influence = observable_influence
        self.missing_columns = []
        self.missing_rows = {}

    def model(self, dependent, feature):
        n = len(dependent)
        coefficients = np.random.rand(dependent.shape[1])
        coefficients = coefficients/2*coefficients.sum()
        coefficients = np.array([1-self.observable_influence] + coefficients.tolist())
        data = pd.concat([feature, dependent], axis=1)
        scores = data.apply(lambda x: logistic(x, coefficients), axis=1)
        scores = scores + self.prob - scores.mean()
        scores = np.maximum(scores, np.zeros(len(scores)))
        return np.random.binomial(1, scores, n).astype('bool')

    def ampute(self, data):
        result = data.copy(deep=True)
        usable_columns = [c for c in data.columns
                          if c not in self.blacklist]
        shuffle(usable_columns)
        self.missing_columns = usable_columns[:self.num_columns]
        usable_columns = usable_columns[self.num_columns:]
        for c in self.missing_columns:
            shuffle(usable_columns)
            dependent_columns = usable_columns[:self.num_dependent_columns]
            rows = self.model(data[dependent_columns], data[c])
            self.missing_rows[c] = rows
            result[c][rows] = np.nan
        return result
