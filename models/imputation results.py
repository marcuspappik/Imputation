import numpy as np
import pandas as pd


class ImputationResult():

    def __init__(self, imputation_method, dataset):
        self.imputation = imputation_method
        self.dataset = dataset
        self.results = []

    def generate_results(self):
        columns = self.dataset.columns()
        missing_data = self.dataset.missing_data()
        results = self.imputation.complete(missing_data)
        if isistance(results, list):
            self.results = results
        else:
            self.results = [results]
