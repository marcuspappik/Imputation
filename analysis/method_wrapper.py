# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-07 17:00:23


class OutlierMethodWrapper():

    def __init__(self, method):
        self.method = method

    def fit(self, X):
        self.method.fit(X)
        return self

    def decision_function(self, X):
        if hasattr(self.method, 'decision_function'):
            return self.method.decision_function(X)
        elif hasattr(self.method, '_decision_function'):
            return self.method._decision_function(X)
        else:
            raise Exception('No outlier scoring available')