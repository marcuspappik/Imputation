import numpy as np
import pandas as pd


class ImputationResult():

    def __init__(self, imputation_method, method_name, dataset):
        self.imputation = imputation_method
        self.name = name
