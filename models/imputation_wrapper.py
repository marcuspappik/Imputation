import numpy as np
import pandas as pd


class SingleImputationWrapper():

    def __init__(self, imputation_method):
        self.imputation = imputation_method

    def complete(self, data):
        columns = data.columns
        complete = self.imputation.complete(data)
        return [pd.DataFrame(columns=columns, data=complete)]

    def name(self):
        return self.imputation.__class__.__name__


class MultiImputationWrapper():

    def __init__(self, imputation_method):
        self.imputation = imputation_method

    def complete(self, data):
        columns = data.columns
        complete = self.imputation.complete(data)
        return [pd.DataFrame(columns=columns, data=c) for c in complete]

    def name(self):
        return self.imputation.__class__.__name__
