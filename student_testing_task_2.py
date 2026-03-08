import copy
import unittest
import pandas as pd

from task_1_naive_bayes import *
from task_2_evaluation import *

actual_class = pd.Series(
    ['Female', 'Male', 'Rodent', 'Female', 'Female', 'Rodent', 'Male', 'Male', 'Male', 'Female', 'Male', 'Rodent',
     'Rodent', 'Male', 'Primate', 'Male', 'Food', 'Food', 'Male', 'Food', 'Primate'],
    name='Class', index=list(range(0, 21)))
predicted_class = pd.Series(
    ['Female', 'Rodent', 'Rodent', 'Male', 'Rodent', 'Rodent', 'Female', 'Female', 'Rodent', 'Female', 'Male', 'Rodent',
     'Female', 'Male', 'Primate', 'Primate', 'Food', 'Primate', 'Food', 'Female', 'Rodent'],
    name='PredictedClass', index=list(range(0, 21)))
class_values = ['Female', 'Male', 'Primate', 'Rodent', 'Food']
expected_matrix = pd.DataFrame([[2, 2, 0, 1, 1], [1, 2, 0, 0, 0], [0, 1, 1, 0, 1], [1, 2, 1, 3, 0], [0, 1, 0, 0, 1]],
                               columns=class_values, index=class_values)

expected_matrix2 = pd.DataFrame([[5, 10, 166, 1], [23, 540, 96, 2], [17, 21, 436, 5],
                                 [17, 14, 110, 87]],
                                columns=['a', 'b', 'c', 'd'], index=['a', 'b', 'c', 'd'])


class Task_2_Testing(unittest.TestCase):

    # This function contains one unit test for confusion_matrix.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #
    def test1_confusion_matrix_basic(self):
        expected = copy.deepcopy(expected_matrix)
        student_matrix = confusion_matrix(copy.deepcopy(actual_class), copy.deepcopy(predicted_class), class_values)
        # Using a helper function for equality checking, given some funky data formats sometimes
        result = (student_matrix.equals(expected_matrix)
                  and expected.columns.equals(student_matrix.columns) and expected.index.equals(student_matrix.index))
        result_message = "confusionMatrix construction failed. Got \n" + str(student_matrix) + " and expected \n" + str(
            expected)
        self.assertEqual(result, True, result_message)

    # Here we check if TPs are calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    #
    def test2_compute_TPs(self):
        student_result = compute_TPs(copy.deepcopy(expected_matrix))
        expected = {'Female': 2, 'Male': 2, 'Primate': 1, 'Rodent': 3, 'Food': 1}
        result = student_result == expected
        result_message = "Computing TPs failed. Expected \n" + str(expected) + " and got \n" + str(student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if FPs are calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    def test3_compute_FPs(self):
        student_result = compute_FPs(copy.deepcopy(expected_matrix))
        expected = {'Female': 4, 'Male': 1, 'Primate': 2, 'Rodent': 4, 'Food': 1}
        result = student_result == expected
        result_message = "Computing FPs failed. Expected \n" + str(expected) + " and got \n" + str(student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if FNs are calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    def test4_compute_FNs(self):
        student_result = compute_FNs(copy.deepcopy(expected_matrix))
        expected = {'Female': 2, 'Male': 6, 'Primate': 1, 'Rodent': 1, 'Food': 2}
        result = student_result == expected
        result_message = "Computing FNs failed. Expected \n" + str(expected) + " and got \n" + str(student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if binary precision is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def test5_compute_binary_precision(self):
        student_result = compute_binary_precision(2, 6, 1)
        expected = 0.25

        result = round_equal(student_result, expected)
        result_message = "Computing binary precision failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if binary recall is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def test6_compute_binary_recall(self):
        student_result = compute_binary_recall(2, 6, 1)
        expected = 0.66666666666667

        result = round_equal(student_result, expected)
        result_message = "Computing binary recall failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if binary f-measure is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def test7_compute_binary_f_measure(self):
        student_result = compute_binary_f_measure(2, 6, 1)
        expected = 0.363636364

        result = round_equal(student_result, expected)
        result_message = "Computing binary f-measure failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if macro precision is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.
    def test8_compute_macro_precision(self):
        student_result = compute_macro_precision(copy.deepcopy(expected_matrix))
        expected = 0.4523809523809524

        result = round_equal(student_result, expected)
        result_message = "Computing macro precision failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if macro recall is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def test9_compute_macro_recall(self):
        student_result = compute_macro_recall(copy.deepcopy(expected_matrix))
        expected = 0.4666666666666667

        result = round_equal(student_result, expected)
        result_message = "Computing macro recall failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if macro f-measure is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def testx10_compute_macro_f_measure(self):
        student_result = compute_macro_f_measure(copy.deepcopy(expected_matrix))
        expected = 0.42181818181818176

        result = round_equal(student_result, expected)
        result_message = "Computing macro f-measure failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

        # Here we check if weighted precision is calculated well
        # This function contains just one test.
        # The function simply checks one possible behaviour, and there are many more possible. More than that
        # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
        # or checked.

    def testx11_compute_weighted_precision(self):
        student_result = compute_weighted_precision(copy.deepcopy(expected_matrix))
        expected = 0.502267573696145

        result = round_equal(student_result, expected)
        result_message = "Computing weighted precision failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

        # Here we check if weighted recall is calculated well
        # This function contains just one test.
        # The function simply checks one possible behaviour, and there are many more possible. More than that
        # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
        # or checked.

    def testx12_compute_weighted_recall(self):
        student_result = compute_weighted_recall(copy.deepcopy(expected_matrix))
        expected = 0.42857142857142855

        result = round_equal(student_result, expected)
        result_message = "Computing weighted recall failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

        # Here we check if weighted f-measure is calculated well
        # This function contains just one test.
        # The function simply checks one possible behaviour, and there are many more possible. More than that
        # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
        # or checked.

    def testx13_compute_weighted_f_measure(self):
        student_result = compute_weighted_f_measure(copy.deepcopy(expected_matrix))
        expected = 0.41385281385281386

        result = round_equal(student_result, expected)
        result_message = "Computing weighted f-measure failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if standard accuracy is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def testx14_compute_standard_accuracy(self):
        student_result = compute_standard_accuracy(copy.deepcopy(expected_matrix))
        expected = 0.42857142857142855

        result = round_equal(student_result, expected)
        result_message = "Computing standard accuracy failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)

    # Here we check if balanced accuracy is calculated well
    # This function contains just one test.
    # The function simply checks one possible behaviour, and there are many more possible. More than that
    # is also supporting markers. Feel free to expand on these tests for your own purposes. This area is not marked
    # or checked.

    def testx15_compute_balanced_accuracy(self):
        student_result = compute_balanced_accuracy(expected_matrix)
        # Using a helper function for equality checking, given some funky data formats sometimes
        expected = 0.4666666666666666

        result = round_equal(student_result, expected)
        result_message = "Computing balanced accuracy failed. Expected \n" + str(expected) + " and got \n" + str(
            student_result)
        self.assertEqual(result, True, result_message)


def frame_round_equal(data1: pd.DataFrame, data2: pd.DataFrame):
    nums1 = data1.to_numpy().flatten()
    nums2 = data2.to_numpy().flatten()
    return list_round_equal(nums1, nums2)


def list_round_equal(nums1, nums2):
    if len(nums1) != len(nums2):
        return False
    for i in range(0, len(nums1)):
        v1 = round(float(nums1[i]), 5)
        v2 = round(float(nums2[i]), 5)
        if v1 != v2:
            return False
    return True


def round_equal(nums1: float, nums2: float):
    v1 = round(float(nums1), 5)
    v2 = round(float(nums2), 5)
    if v1 != v2:
        return False
    return True


if __name__ == '__main__':
    unittest.main()
