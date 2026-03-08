import unittest

import numpy
import pandas as pd
from task_1_naive_bayes import *
from task_2_evaluation import *
from task_3_cross_validation import *


class Task_3_Testing(unittest.TestCase):
    # This function contains one unit test for evaluating results. It is in no way a comprehensive one.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #
    #
    # Other tests are not given as they show the problem solutions.
    def test_evaluate_results(self):
        class_values = ['Female', 'Male', 'Primate', 'Rodent', 'Food']
        actual_class_list = [pd.Series(['Female', 'Male', 'Female', 'Female']),
                             pd.Series(['Food', 'Female', 'Male', 'Female']),
                             pd.Series(['Female', 'Female', 'Rodent', 'Female', 'Female'])]
        predicted_class_list = [pd.Series(['Female', 'Male', 'Rodent', 'Primate']),
                                pd.Series(['Food', 'Food', 'Female', 'Female']),
                                pd.Series(['Rodent', 'Female', 'Male', 'Primate', 'Female'])]

        try:
            output = evaluate_results(actual_class_list, predicted_class_list, class_values)
            expected_output = {'avg_macro_precision': 0.266666667, 'avg_macro_recall': 0.222222222,
                               'avg_macro_f_measure': 0.222222222, 'avg_weighted_precision': 0.725,
                               'avg_weighted_recall': 0.4666666666666666, 'avg_weighted_f_measure': 0.5249999999999999,
                               'avg_standard_accuracy': 0.466666667, 'avg_balanced_accuracy': 0.222222222}
            res = True
            for key in expected_output.keys():
                res = res and round_equal(expected_output[key], output[key])
            res = res and expected_output.keys() == output.keys()

            with self.subTest(msg="Checking if evaluating results with student function worked"):
                self.assertEqual(res, True, "evaluate_results failed")
        except Exception as e:
            print(e)
            with self.subTest(msg="Code crashed with student functions"):
                self.assertEqual(False, True, "evaluate_results crashed")


def round_equal(nums1: float, nums2: float):
    v1 = round(float(nums1), 5)
    v2 = round(float(nums2), 5)
    if v1 != v2:
        return False
    return True


if __name__ == '__main__':
    unittest.main()
