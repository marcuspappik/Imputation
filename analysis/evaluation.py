import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import copy


class Evaluation():

    def __init_(self, method, scores, dataset):
        self.dataset = dataset
        self.method = method
        self.setting = ['p', 'mechanism', 'imputation', 'number']
        self.scores = scores
        self.columns = []
        for s in self.setting:
            self.columns.append(('settings', s))
        for s in self.scores:
            self.columns.append(s, 'post')
            self.columns.append(s, 'prev')
        self.evaluation_results = pd.DataFrame(columns=self.columns)

    def evaluate_result(self, results, p, mechanism, imputation, run):
        if not isinstance(results, list):
            results = [results]
        setting_update = {('settings', 'p'): p,
                          ('settings', 'mechanism'): mechanism,
                          ('settings', 'imputation'): imputation}
        evaluation_scores = self._calculate_scores(results)
        for i in range(len(evaluation_scores)):
            evaluation_scores[i].update(setting_update)
        self.evaluation_results = self.evaluation_results.append(to_append)

    def dump_results(self):
        path = self.dataset.construct_path()+'results.csv'
        self.evaluation_results.to_csv(path, index=False)

    def _calculate_scores(self, results):
        raise(NotImplementedError())


class EvaluateClassification():

    def __init__(self, classifier, dataset):
        scores = {'accuracy': accuracy_score}
        super().__init__(classifier, scores, dataset)
        self.split = self._generate_split()

    def _train(self, result):
        clf = copy.deepcopy(self.method)
        X = result[self.split]
        y = self.dataset.target()[self.split]
        clf.fit(X, y)
        return clf

    def _test(self, classifiers result):
        X = result[self.split == 0]
        prediction = classifier.predict(X)
        return prediction

    def _mode_prediction(self, predictions):
        return np.array(pd.DataFrame(predictions).mode(axis=0))[0]

    def _mean_result(self, results):
        new_result = 0
        for r in results:
            new_result = new_result + r
        new_result = new_result/len(results)
        return new_results

    def _apply_score(self, f, prediction):
        y = self.dataset.target()[self.split==0]
        return f(y, prediction)

    def _calculate_scores(self, results):
        classifiers = [self._train(r) for r in results]
        predictions = [self._test(r, c) for r, c in zip(results, classifiers)]

        calculations = [{(s, 'prev'): self._apply_score(f, predictions[0])
                         for (s, f) in self.scores}]
        calculations[0].update({(s, 'post'): self._apply_score(f, predictions[0])
                                for (s, f) in self.scores})
        calculations[0].update({('settings', 'number'): 1})

        if len(results < 2):
            return calculations

        for i in range(5, len(results)+1, 5):
            post_prediction = self._mode_prediction(predictions[:i])

            mean_result = self._mean_result(results[:i])
            prev_classifier = self._train(mean_result)
            prev_prediction = self._test(mean_result, prev_classifier)

            new_calculation = {('setting', 'number'): i}
            new_calculation.update({(s, 'prev'): self._apply_score(f, prev_prediction)
                                    for (s, f) in self.scores})
            new_calculation.update({(s, 'post'): self._apply_score(f, post_prediction)
                                    for (s, f) in self.scores})
            calculations.append(new_calculation)
        return calculations


