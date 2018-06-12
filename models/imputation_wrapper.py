# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-12 18:11:35

import numpy as np
import pandas as pd
import time


class SingleImputationWrapper():

    def __init__(self, imputation_method):
        self.imputation = imputation_method

    def complete(self, data):
        columns = data.columns
        start_time = time.time()
        complete = self.imputation.complete(data)
        exec_time = time.time() - start_time
        return [pd.DataFrame(columns=columns, data=complete)], exec_time

    def name(self):
        return self.imputation.__class__.__name__

    def number(self):
        return 1


class MultiImputationWrapper():

    def __init__(self, imputation_method):
        self.imputation = imputation_method

    def complete(self, data):
        columns = data.columns
        start_time = time.time()
        complete = self.imputation.complete(data)
        exec_time = time.time() - start_time
        return [pd.DataFrame(columns=columns, data=c) for c in complete], exec_time

    def name(self):
        return self.imputation.__class__.__name__

    def number(self):
        return self.imputation.imputations
