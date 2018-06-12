# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-12 17:38:14


from fancyimpute import MICE


class Mice():

    def __init__(self, imputations=5, verbose=False):
        self.imputations = imputations
        self.verbose = verbose

    def complete(self, data):
        results = []
        for i in range(self.imputations):
            results.append(MICE(n_imputations=1, verbose=self.verbose).complete(data))
        return results
