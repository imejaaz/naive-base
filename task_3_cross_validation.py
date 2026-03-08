import pandas as pd

import task_2_evaluation
from task_1_naive_bayes import *


# This function takes in the data for cross evaluation and the number of partitions to split the data into.
# As input, the function takes:
# - training_data     - a pandas DataFrame containing the data set to be split
# - f                 - the number of partitions to split the data into, value is greater than 0,
#                       not guaranteed to be smaller than data size. If f exceeds the size of the data,
#                       cap it at data size.
# As output, it produces:
# - partition_list   - a list of pandas DataFrames, where each data frame represents a partition, so a subset of entries
#                     of the original dataset s.t. all partitions are disjoint, roughly same size (can differ by
#                     at most 1), and the union of all partitions equals the original dataset. The indexing must be
#                     preserved - i.e. row with row name/index 12 in the original dataset will have
#                     same name whatever partition it is in. The column and row names in the partition must be the same
#                     as in training_data.
def partition_data(training_data: pd.DataFrame, f: int) -> list[pd.DataFrame]:

    partition_list = []
    # Compute, compute, compute!
    return partition_list


# This function transforms partitions into training and testing data for each cross-validation round (there are
# as many rounds as there are partitions); in other words, we prepare the folds. The column and row names of the
# new testing and training datasets must be preserved.
# At input, the function takes:
# - partition_list - a list of data frames, where each data frame represents a partition (see partition_data function)
# - f - the number of folds to use in cross-validation, which is the same as the number of partitions
#       the data was supposed to be split to, and the number of rounds in cross-validation. Value is greater than 0.
#
# The function produces:
# - folds - a list of 3-tuple s.t. the first element is the round number, second is the training data for that round,
#           and third is the testing data for that round. The round numbers START WITH 0.
#           The indexing must be preserved - i.e. row with row name/index 12 in the original dataset will have
#           same name whatever fold it is in

def arrange_data_for_cv(partition_list: list[pd.DataFrame], f: int) \
        -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    # This is just for error handling, if for some magical reason f and number of partitions are not the same,
    # then something must have gone wrong in the other functions and you should investigate it
    if len(partition_list) != f:
        print("Something went really wrong! Why is the number of partitions different from f??")
        return []
    folds = []
    # Do your thing!
    return folds


# This function takes the lists of actual and predicted classes for each round, and produces averaged metrics.
# Invoke either the Task 2 evaluation here; do not do everything from scratch!
#
# At input, it takes:
# - actual_class_list, predicted_class_list
#                           - lists of pandas Series representing the actual and predicted classes
#                           for each cross validation round
#        class_values - the list of all possible class values
# Function outputs:
# - computed measures - a dictionary of measures, explicitly listing 'average_macro_precision', 'average_macro_recall',
#                       'average_macro_f_measure', 'average_weighted_precision', 'average_weighted_recall',
#                       'average_weighted_f_measure', 'average_standard_accuracy' and 'average_balanced_accuracy'

def evaluate_results(actual_class_list: list[pd.Series], predicted_class_list: list[pd.Series],
                     class_values: list[str]) -> dict[str, float]:
    results = {'avg_macro_precision': 0.0, 'avg_macro_recall': 0.0, 'avg_macro_f_measure': 0.0,
               'avg_weighted_precision': 0.0, 'avg_weighted_recall': 0.0,
               'avg_weighted_f_measure': 0.0, 'avg_standard_accuracy': 0.0,
               'avg_balanced_accuracy': 0.0}
    # Black magic time!
    return results


# In this task you are expected to perform and evaluate cross-validation on a given dataset.
# You are expected to partition the input dataset into f partitions, then arrange them into training and testing
# data for each cross validation round, and then train and execute naive Bayes for each round using this data.
#
# You are then asked to produce an output dataset which extends the original input training_data by adding
# "PredictedClass" and "Fold" columns, which for each entry state what class it got predicted when it
# landed in a testing fold and what the number of that fold was (remember, numbering starts from 0). This
# is paired with a dictionary listing average evaluation measures.
# You must use the other relevant function defined in this file.
#
# At input, the function takes:
# - nb - naive Bayes classifier from Task 1
# - training_data - a pandas DataFrame representing the data
# - partition_func - the function used to partition the input dataset (by default, it is the one above)
# - prep_func - the function used to transform the partitions into appropriate folds
#                            (by default, it is the one above)
#  -eval_func - the function used to evaluate cross validation (by default, it is the one above)
#
# As output, it produces a tuple consisting of
# - output_dataset - a pandas DataFrame which extends the original input training_data by adding "PredictedClass"
#                    and "Fold" columns, which for each entry state what class it got predicted when it
#                    landed in a testing fold and what the number of that fold was (numbering starts from 0).
# - evaluation metrics - average evaluation metrics as produced by eval_func
def cross_validate(nb: NaiveBayes, training_data: pd.DataFrame, f: int,
                   partition_func=partition_data, prep_func=arrange_data_for_cv, eval_func=evaluate_results) \
        -> tuple[pd.DataFrame, dict[str, float]]:
    output_dataset = None

    return output_dataset, {}
