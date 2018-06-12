# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-12 18:37:47


import numpy as np
import pandas as pd
from random import shuffle
from math import ceil
from sklearn.metrics import accuracy_score, f1_score, \
                            matthews_corrcoef, roc_auc_score
from analysis.scores import L1_score, L2_score
import copy


class Evaluation():

    def __init__(self, method, scores, dataset):
        self.dataset = dataset
        self.method = method
        self.setting = ['p', 'mechanism', 'imputation', 'number']
        self.scores = scores
        self.columns = []
        for s in self.setting:
            self.columns.append(('settings', s))
        for s in self.scores:
            self.columns.append((s, 'post'))
            self.columns.append((s, 'prev'))
        self.evaluation_results = pd.DataFrame(columns=self.columns)

    def evaluate_result(self, results, p, mechanism, imputation):
        if not isinstance(results, list):
            results = [results]
        setting_update = {('settings', 'p'): p,
                          ('settings', 'mechanism'): mechanism,
                          ('settings', 'imputation'): imputation}
        evaluation_scores = self._calculate_scores(results)
        for i in range(len(evaluation_scores)):
            evaluation_scores[i].update(setting_update)
        self.evaluation_results = self.evaluation_results.append(evaluation_scores,
                                                                 ignore_index=True)

    def dump_results(self, suffix='evaluation'):
        path = self.dataset.construct_path()+'results_'+suffix+'.csv'
        self.evaluation_results.to_csv(path, index=False)

    def _calculate_scores(self, results):
        raise(NotImplementedError())


class ClassificationEvaluation(Evaluation):

    def __init__(self, classifier, dataset):
        scores = {'accuracy': accuracy_score,
                  'f1-score': f1_score,
                  'matthews': matthews_corrcoef}
        super().__init__(classifier, scores, dataset)
        self.train_indicator = self._generate_train_indicator()

    def _generate_train_indicator(self):
        n = len(self.dataset.target())
        indicator = np.arange(0, 1, 1/n) > 0.7
        np.random.shuffle(indicator)
        return indicator

    def _train(self, result):
        clf = copy.deepcopy(self.method)
        X = result[self.train_indicator]
        y = self.dataset.target()[self.train_indicator]
        clf.fit(X, y)
        return clf

    def _test(self, result, classifier):
        X = result[~self.train_indicator]
        prediction = classifier.predict(X)
        return prediction

    def _mode_prediction(self, predictions):
        return np.array(pd.DataFrame(predictions).mode(axis=0))[0]

    def _mean_result(self, results):
        new_result = 0
        for r in results:
            new_result = new_result + r
        new_result = new_result/len(results)
        return new_result

    def _apply_score(self, f, prediction):
        y = self.dataset.target()[~self.train_indicator]
        return f(y, prediction)

    def _calculate_scores(self, results):
        classifiers = [self._train(r) for r in results]
        predictions = [self._test(r, c) for r, c in zip(results, classifiers)]

        calculations = [{(s, 'prev'): self._apply_score(f, predictions[0])
                         for s, f in self.scores.items()}]
        calculations[0].update({(s, 'post'): self._apply_score(f, predictions[0])
                                for s, f in self.scores.items()})
        calculations[0].update({('settings', 'number'): 1})

        if len(results) < 2:
            return calculations

        for i in range(5, len(results)+1, 5):
            post_prediction = self._mode_prediction(predictions[:i])

            mean_result = self._mean_result(results[:i])
            prev_classifier = self._train(mean_result)
            prev_prediction = self._test(mean_result, prev_classifier)

            new_calculation = {('settings', 'number'): i}
            new_calculation.update({(s, 'prev'): self._apply_score(f, prev_prediction)
                                    for s, f in self.scores.items()})
            new_calculation.update({(s, 'post'): self._apply_score(f, post_prediction)
                                    for s, f in self.scores.items()})
            calculations.append(new_calculation)
        return calculations


class OutlierEvaluation(Evaluation):

    def __init__(self, detection, dataset):
        scores = {'auc': roc_auc_score}
        super().__init__(detection, scores, dataset)

    def _mean_scoring(self, predictions):
        return np.array(pd.DataFrame(predictions).mode(axis=0))[0]

    def _mean_result(self, results):
        new_result = 0
        for r in results:
            new_result = new_result + r
        new_result = new_result/len(results)
        return new_result

    def _apply_score(self, f, prediction):
        y = self.dataset.target()
        return f(y, prediction)

    def _fit(self, result):
        return copy.deepcopy(self.method).fit(result)

    def _scoring(self, result, detection):
        return detection.decision_function(result)

    def _calculate_scores(self, results):
        detections = [self._fit(r) for r in results]
        out_scores = [self._scoring(r, d) for r, d in zip(results, detections)]

        calculations = [{(s, 'prev'): self._apply_score(f, out_scores[0])
                         for s, f in self.scores.items()}]
        calculations[0].update({(s, 'post'): self._apply_score(f, out_scores[0])
                                for s, f in self.scores.items()})
        calculations[0].update({('settings', 'number'): 1})

        if len(results) < 2:
            return calculations

        for i in range(5, len(results)+1, 5):
            post_out_scores = self._mean_scoring(out_scores[:i])

            mean_result = self._mean_result(results[:i])
            prev_detection = self._fit(mean_result)
            prev_out_scores = self._scoring(mean_result, prev_detection)

            new_calculation = {('settings', 'number'): i}
            new_calculation.update({(s, 'prev'): self._apply_score(f, prev_out_scores)
                                    for s, f in self.scores.items()})
            new_calculation.update({(s, 'post'): self._apply_score(f, post_out_scores)
                                    for s, f in self.scores.items()})
            calculations.append(new_calculation)
        return calculations


class LnormEvaluation(Evaluation):

    def __init__(self, dataset):
        scores = {'L1': L1_score, 'L2': L2_score}
        super().__init__(None, scores, dataset)

    def _mean_result(self, results):
        new_result = 0
        for r in results:
            new_result = new_result + r
        new_result = new_result/len(results)
        return new_result

    def _mean_score(self, f, original, results):
        score_sum = 0
        for r in results:
            score_sum = score_sum + f(original, r)
        return score_sum/len(results)

    def _calculate_scores(self, results):
        original = self.dataset.complete_data()
        calculations = [{(s, 'prev'): f(original, results[0])
                         for s, f in self.scores.items()}]
        calculations[0].update({(s, 'post'): f(original, results[0])
                                for s, f in self.scores.items()})
        calculations[0].update({('settings', 'number'): 1})
        if len(results) < 2:
            return calculations

        for i in range(5, len(results)+1, 5):
            current_results = results[:i]
            mean_result = self._mean_result(current_results)
            new_calculation = {('settings', 'number'): i}
            new_calculation.update({(s, 'prev'): f(original, mean_result)
                                    for s, f in self.scores.items()})
            new_calculation.update({(s, 'post'): self._mean_score(f, original, current_results)
                                    for s, f in self.scores.items()})
            calculations.append(new_calculation)
        return calculations

    def dump_results(self, suffix='Lnorm'):
        super().dump_results(suffix=suffix)


class TimeEvaluation():

    def __init__(self):
        self.columns = ['p', 'mechanism', 'imputation', 'number', 'runtime']
        self.evaluation_results = pd.DataFrame(columns=self.columns)

    def evaluate_result(self, runtime, p, mechanism, imputation, number):
        to_append = {'p': p, 'mechanism': mechanism, 'imputation': imputation,
                     'number': number, 'runtime': runtime}
        self.evaluation_results = self.evaluation_results.append(to_append,
                                                                 ignore_index=True)

    def dump_results(self, suffix='Time'):
        super().dump_results(suffix=suffix)