# -*- coding: utf-8 -*-
# @Author: Marcus Pappik
# @Date:   2018-06-07 16:49:03
# @Last Modified by:   marcus
# @Last Modified time: 2018-06-28 13:02:27


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

    def __init__(self, classifier, dataset, cv=10):
        scores = {'accuracy': accuracy_score,
                  'f1-score': lambda x, y: f1_score(x, y, average='micro')}
        super().__init__(classifier, scores, dataset)
        self.cv = cv
        self.train_indicators = [self._generate_train_indicator()
                                 for i in range(self.cv)]

    def _generate_train_indicator(self):
        n = len(self.dataset.target())
        indicator = np.arange(0, 1, 1/n) > 0.7
        np.random.shuffle(indicator)
        return indicator

    def _train(self, X, y):
        clf = copy.deepcopy(self.method)
        clf.fit(X, y)
        return clf

    def _test(self, X, classifier):
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

    def _calculate_scores(self, results):
        all_calculations = []
        for k in range(self.cv):
            train_indicator = self.train_indicators[k]
            train_Xs = [r[train_indicator] for r in results]
            test_Xs = [r[~train_indicator] for r in results]
            y_train = self.dataset.target()[train_indicator]
            y_test = self.dataset.target()[~train_indicator]

            classifiers = [self._train(X, y_train) for X in train_Xs]
            predictions = [self._test(X, c) for X, c in zip(test_Xs, classifiers)]

            calculations = [{(s, 'prev'): f(y_test, predictions[0])
                             for s, f in self.scores.items()}]
            calculations[0].update({(s, 'post'): f(y_test, predictions[0])
                                    for s, f in self.scores.items()})
            calculations[0].update({('settings', 'number'): 1})

            if len(results) < 2:
                all_calculations += calculations
                continue

            for i in range(5, len(results)+1, 5):
                post_prediction = self._mode_prediction(predictions[:i])

                mean_train = self._mean_result(train_Xs[:i])
                mean_test = self._mean_result(test_Xs[:i])
                prev_classifier = self._train(mean_train, y_train)
                prev_prediction = self._test(mean_test, prev_classifier)

                new_calculation = {('settings', 'number'): i}
                new_calculation.update({(s, 'prev'): f(y_test, prev_prediction)
                                        for s, f in self.scores.items()})
                new_calculation.update({(s, 'post'): f(y_test, post_prediction)
                                        for s, f in self.scores.items()})
                calculations.append(new_calculation)
            all_calculations += calculations

        score_columns = [(s, 'prev') for s in self.scores.keys()]
        score_columns += [(s, 'post') for s in self.scores.keys()]
        calculations_df = pd.DataFrame.from_dict(all_calculations)
        calculation_groups = calculations_df.groupby([('settings', 'number')])
        mean_df = calculation_groups[score_columns].mean().reset_index()
        return list(mean_df.T.to_dict().values())

    def dump_results(self):
        super().dump_results(suffix=self.method.name())


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

    def dump_results(self):
        super().dump_results(suffix=self.method.name())


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

    def dump_results(self):
        super().dump_results(suffix='Lnorm')


class TimeEvaluation():

    def __init__(self, dataset):
        self.columns = ['p', 'mechanism', 'imputation', 'number', 'runtime']
        self.evaluation_results = pd.DataFrame(columns=self.columns)
        self.dataset = dataset

    def evaluate_result(self, runtime, p, mechanism, imputation, number):
        to_append = {'p': p, 'mechanism': mechanism, 'imputation': imputation,
                     'number': number, 'runtime': runtime}
        self.evaluation_results = self.evaluation_results.append(to_append,
                                                                 ignore_index=True)

    def dump_results(self):
        path = self.dataset.construct_path()+'results_Time'+'.csv'
        self.evaluation_results.to_csv(path, index=False)