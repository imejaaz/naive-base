# Task 2 [10 points out of 40] Classifier evaluation
# This task focuses on evaluating the naïve Bayes classifier from Task 1. On your own, implement binary precision,
# recall and f-measure, as well as their macro and weighted counterparts.
# You are also asked to implement the multiclass version of accuracy, and its weighted counterpart. You
# need to follow the formulas covered in the module. Remember to be mindful of edge cases (the approach for handling
# them is explained in lecture slides).
# Please note that this template also contains empty functions pertaining to
# creating a confusion matrix and calculating TPs, FPs and FNs based on it. These will be implemented during the
# practicals, with the code to be released later. They are not a part of the marking criteria.

import pandas as pd


# This function computes the confusion matrix based on the provided series of actual and predicted classes.
# The returned data frame must contain appropriate column and row names, and be filled with integers.
# The columns correspond to actual classes and rows to predicted classes, in the sense that the i-th row
# is the row representing how often entries actually belonging to some class, were predicted as the i-th class value;
# the i-th column represents how often entries predicted as some other class, actually belonged to the i-th class.
#
# At input, function takes:
# - actual_class, predicted_class - series of class values representing actual and predicted classes of some dataset.
#                                   NOT guaranteed to contain all possible class values from the classification schema.
# - class_values - all possible values of the class from which actual_class and predicted_class were drawn.
#
# As output, it produces:
# - matrix : a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. The data frame must contain appropriate column and row names, and be
#            filled with integers.

def confusion_matrix(actual_class: pd.Series, predicted_class: pd.Series, class_values: list[str]) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=class_values, columns=class_values)

    # WE WILL IMPLEMENT THIS IN CLASS, DON'T WORRY!

    return matrix


# These functions compute per-class true positives and false positives/negatives based on the provided confusion matrix.
# WE WILL IMPLEMENT THEM IN CLASS, DON'T WORRY!
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
#
# As output, these functions produce:
# - tps/fps/fns - dictionaries that for every class value in the classification scheme (corresponding to names of
#                 all rows and/or all columns in the matrix) return the true positive, false positive or
#                 false negative values for that class.

def compute_TPs(matrix: pd.DataFrame) -> dict[str, int]:
    # WE WILL IMPLEMENT THIS IN CLASS, DON'T WORRY!
    return {}


def compute_FPs(matrix: pd.DataFrame) -> dict[str, int]:
    # WE WILL IMPLEMENT THIS IN CLASS, DON'T WORRY!
    return {}


def compute_FNs(matrix: pd.DataFrame) -> dict[str, int]:
    # WE WILL IMPLEMENT THIS IN CLASS, DON'T WORRY!
    return {}


# These functions compute the binary measures based on the provided values. Not all measures use all the values.
# Do not remove the unused variables from the function pattern.
# At input, the functions take:
# - tp, fp, fn : the single values of true positives, false positive and negatives
#
# As output, they produce:
# - binary precision/recall/f-measure - appropriate evaluation measure created using the binary approach.

def compute_binary_precision(tp: int, fp: int, fn: int) -> float:
    return -1


def compute_binary_recall(tp: int, fp: int, fn: int) -> float:
    return -1


def compute_binary_f_measure(tp: int, fp: int, fn: int) -> float:
    return -1


# These functions compute the macro precision, macro recall, macro f-measure, based on the offered confusion matrix.
# You are expected to use appropriate binary counterparts when needed (binary recall for macro recall, binary precision
# for macro precision, binary f-measure for macro f-measure) and the functions for computing tps/fps/fns as needed.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - macro precision/recall/f-measure - appropriate evaluation measures created using the macro average approach.

def compute_macro_precision(matrix: pd.DataFrame) -> float:
    return -1


def compute_macro_recall(matrix: pd.DataFrame) -> float:
    return -1


def compute_macro_f_measure(matrix: pd.DataFrame) -> float:
    return -1


# These functions compute the weighted precision, macro recall, macro f-measure, based on the offered confusion matrix.
# You are expected to use appropriate binary counterparts when needed (binary recall for weighted recall,
# binary precision for weighted precision, binary f-measure for weighted f-measure) and the functions
# for computing tps/fps/fns as needed.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - weighted precision/recall/f-measure - appropriate evaluation measures created using the weighted average approach.

def compute_weighted_precision(matrix: pd.DataFrame) -> float:
    return -1


def compute_weighted_recall(matrix: pd.DataFrame) -> float:
    return -1


def compute_weighted_f_measure(matrix: pd.DataFrame) -> float:
    return -1


# These functions compute the standard and balanced multiclass accuracies based on the offered confusion matrix.
# You are expected to use appropriately select and use the functions defined previously.
#
# As input, these functions take:
# - matrix - a data frame representing the confusion matrix computed based on the offered series of actual
#            and predicted classes. See confusion_matrix function for description.
# As output, they produce:
# - standard/balanced multiclass accuracy - appropriate evaluation measures created using the
#                                           standard/balanced approach.


def compute_standard_accuracy(matrix: pd.DataFrame) -> float:
    return -1


def compute_balanced_accuracy(matrix: pd.DataFrame) -> float:
    return -1


# In this function you are expected to compute precision, recall, f-measure and accuracy of your classifier using
# the macro average approach.
# At input, the function takes:
# - actual_class - a pandas Series of actual class values
# - predicted_class - a pandas Series of predicted class values
# - class_values - a list of all possible class values (actual and predicted classes are not guaranteed to be complete)
# - confusion_func - function to be invoked to compute the confusion matrix
# Function outputs:
# - computed measures - a dictionary of measures, explicitly listing 'macro_precision', 'macro_recall',
#                       'macro_f_measure', 'weighted_precision', 'weighted_recall', 'weighted_f_measure',
#                       'standard_accuracy' and 'balanced_accuracy'

def evaluate_classification(actual_class: pd.Series, predicted_class: pd.Series, class_values: list[str],
                            confusion_func=confusion_matrix) \
        -> dict[str, float]:
    # Have fun with the computations!
    macro_precision = -1.0
    macro_recall = -1.0
    macro_f_measure = -1.0

    weighted_precision = -1.0
    weighted_recall = -1.0
    weighted_f_measure = -1.0

    standard_accuracy = -1.0
    balanced_accuracy = -1.0
    # once ready, we return the values
    return {'macro_precision': macro_precision, 'macro_recall': macro_recall, 'macro_f_measure': macro_f_measure,
            'weighted_precision': weighted_precision, 'weighted_recall': weighted_recall,
            'weighted_f_measure': weighted_f_measure, 'standard_accuracy': standard_accuracy,
            'balanced_accuracy': balanced_accuracy}
