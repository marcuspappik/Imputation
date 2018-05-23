import unittest
import numpy as np
import pandas as pd

from preprocessing.missing_value_generation import mcar_generator, \
                                                   mar_generator, \
                                                   mnar_generator


def mock_data(dimensionality=5, samples=10000):
    columns = ['f'+str(i+1) for i in range(dimensionality)]
    A = np.random.rand(dimensionality, dimensionality)
    cov = np.dot(A, A.T)
    means = np.random.rand(dimensionality) * 5
    data = pd.DataFrame(columns=columns,
                        data=np.random.multivariate_normal(means, cov, samples))
    return data


class test_mcar_generator(unittest.TestCase):

    def test_ampute_data(self):
        data = mock_data()
        probability = 0.5
        num_columns = 3
        blacklist = ['f1']
        generator = mcar_generator()
        generator.init_dataset(data, num_columns, blacklist)
        mv_data = generator.ampute(data, probability)
        missing_columns = mv_data.columns[mv_data.isnull().any()].tolist()
        self.assertEqual(len(missing_columns), num_columns)
        self.assertTrue('f1' not in missing_columns)
        self.assertEqual(set(missing_columns), set(generator.missing_columns))
        for c in generator.missing_columns:
            mv_relative = mv_data[c].isnull().sum()/len(mv_data)
            self.assertAlmostEqual(mv_relative, probability, delta=0.05)


class test_mar_generator(unittest.TestCase):

    def test_ampute_data(self):
        data = mock_data()
        probability = 0.5
        num_columns = 3
        num_dependant_columns = 1
        blacklist = ['f1']
        generator = mar_generator(num_dependant_columns)
        generator.init_dataset(data, num_columns, blacklist)
        mv_data = generator.ampute(data, probability)
        missing_columns = mv_data.columns[mv_data.isnull().any()].tolist()
        self.assertEqual(len(missing_columns), num_columns)
        self.assertTrue('f1' not in missing_columns)
        self.assertEqual(set(missing_columns), set(generator.missing_columns))
        for c in generator.missing_columns:
            mv_relative = mv_data[c].isnull().sum()/len(mv_data)
            self.assertAlmostEqual(mv_relative, probability, delta=0.05)


class test_mnar_generator(unittest.TestCase):

    def test_ampute_data(self):
        data = mock_data()
        probability = 0.5
        num_columns = 3
        num_dependant_columns = 1
        blacklist = ['f1']
        generator = mnar_generator(num_dependant_columns)
        generator.init_dataset(data, num_columns, blacklist)
        mv_data = generator.ampute(data, probability)
        missing_columns = mv_data.columns[mv_data.isnull().any()].tolist()
        self.assertEqual(len(missing_columns), num_columns)
        self.assertTrue('f1' not in missing_columns)
        self.assertEqual(set(missing_columns), set(generator.missing_columns))
        for c in generator.missing_columns:
            mv_relative = mv_data[c].isnull().sum()/len(mv_data)
            self.assertAlmostEqual(mv_relative, probability, delta=0.05)