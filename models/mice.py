from fancyimpute import MICE


class Mice():

    def __init__(self, imputations=5):
        self.imputations = imputations

    def complete(self, data):
        results = []
        for i in range(self.imputations):
            results.append(MICE(n_imputations=1).complete(data))
        return results
