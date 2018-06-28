# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-28 17:13:48


class OutlierMethodWrapper():

    def __init__(self, method):
        self.method = method
        self._name = self.method.__class__.__name__

    def name(self):
        return self._name

    def fit(self, X):
        self.method.fit(X)
        return self

    def decision_function(self, X):
        if hasattr(self.method, 'decision_function'):
            return -1*self.method.decision_function(X)
        elif hasattr(self.method, '_decision_function'):
            return -1*self.method._decision_function(X)
        else:
            raise Exception('No outlier scoring available')


class ClassificationMethodWrapper():

    def __init__(self, method):
        self.method = method
        self._name = self.method.__class__.__name__

    def name(self):
        return self._name

    def fit(self, X, y):
        self.method.fit(X, y)
        return self

    def predict(self, X):
        return self.method.predict(X)
