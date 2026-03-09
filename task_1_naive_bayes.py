# Task 1 [15 points out of 40] Naïve Bayes classifier

# Your first task is to implement the naïve Bayes classifier on your own. This involves calculating all the necessary
# probabilities from the provided data, and using them to make predictions for new unseen records.
# The required version is the one covered in the module. Implementing other naïve Bayes approaches (e.g. Gaussian) or
# using further modifications that do not correspond to the technique practiced in the module will lead to
# significant loss of points.
# The template contains a range of functions you must implement and use appropriately for this task. The template also
# uses a range of functions implemented by the module leader to support you in this task.

import pandas as pd


class NaiveBayes:

    # This function simply initializes an instance of NaiveBayes class. The constructor takes at input:
    # - class_info - pair that contains the name of the class column and its permitted values
    # - feature_info - dictionary that states attribute names and their permitted values

    def __init__(self, class_info: tuple[str, list[str]], feature_info: dict[str, list[str]]):
        self.class_info = class_info
        self.feature_info = feature_info
        # You can add further variables/attributes/etc. here
        # local variables for counts and probabilities
        self._trained = False
        self._class_counts: dict[str, int] = {}
        self._conditional_counts: dict[str, dict[str, dict[str, int]]] = {}
        self._total_instances = 0

    # This function trains the model, aka calculates all the necessary probabilities that a naive Bayes model needs.
    # How you store the computed probabilities internally is up to you - you may want to extend the init function.
    # For the purpose of this task, numerical values are treated just like categorical ones. Any new training
    # should purge old data.
    # At input, train_model takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    def train_model(self, training_data: pd.DataFrame):
        # purge old trining
        self._trained = False
        self._class_counts = {}
        self._conditional_counts = {}
        self._total_instances = 0

        # get class column name
        class_col = self.class_info[0]
        # count clases
        self._total_instances = len(training_data)
        for class_val in self.class_info[1]:
            self._class_counts[class_val] = 0
            # initialize conditional count for every feature and each permitted value
            self._conditional_counts[class_val] = {}
            for feat, vals in self.feature_info.items():
                self._conditional_counts[class_val][feat] = {v: 0 for v in vals}

        # iterate rows and populate counts
        for _, row in training_data.iterrows():
            c = row[class_col]
            # if class value not in expected list, skip
            if c not in self._class_counts:
                # include it to be robust
                self._class_counts[c] = 0
                self._conditional_counts[c] = {feat: {v: 0 for v in vals} for feat, vals in self.feature_info.items()}
            self._class_counts[c] += 1
            for feat in self.feature_info.keys():
                val = row[feat]
                # convert to str for dictionary keys
                if val not in self._conditional_counts[c][feat]:
                    # unseen value - add key with count 1
                    self._conditional_counts[c][feat][val] = 1
                else:
                    self._conditional_counts[c][feat][val] += 1

        self._trained = True

    # This function predicts the classes for entries in the training_data and produces an extended data frame.
    # At input, it takes:
    # - training_data - a pandas DataFrame that contains all the attribute values and class value for a given entry
    # The function outputs:
    # classified_data - a pandas DataFrame which expands the training_data by adding the "PredictedClass" column
    #                   that for every entry states the class value predicted for that entry. In case of ties,
    #                   the chosen class is the one that appears earlier alphabetically.
    def predict(self, testing_data: pd.DataFrame) -> pd.DataFrame:
        if not self._trained:
            # nothing trained, return input with PredictedClass as None
            df = testing_data.copy()
            df['PredictedClass'] = None
            return df

        class_col = self.class_info[0]
        class_values = self.class_info[1]

        # for each row compute posterior proportional score = P(class) * product P(feat=val | class)
        results = []
        for _, row in testing_data.iterrows():
            best_classes = []
            best_score = -1
            # compute for all classes
            for c in sorted(class_values):
                # if class was never seen in training, probability 0
                class_count = self._class_counts.get(c, 0)
                if self._total_instances == 0 or class_count == 0:
                    score = 0.0
                else:
                    score = class_count / self._total_instances
                    for feat in self.feature_info.keys():
                        val = row[feat]
                        # conditional probability = count(feature=value and class)/count(class)
                        count = self._conditional_counts.get(c, {}).get(feat, {}).get(val, 0)
                        cond = count / class_count if class_count > 0 else 0.0
                        score = score * cond
                if score > best_score:
                    best_score = score
                    best_classes = [c]
                elif score == best_score:
                    best_classes.append(c)

            # break ties alphabetically: sorted class_values ensures alphabetical order was used when iterating,
            # but tie list may contain classes in that order; pick earliest alphabetically
            predicted = sorted(best_classes)[0] if best_classes else None
            results.append(predicted)

        df = testing_data.copy()
        df['PredictedClass'] = results
        return df

    # The function returns the probability of a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value for which we want to calculate the probability
    # The function outputs:
    # - probability - float representing the probability of the given class value
    def retrieve_class_probability(self, class_value: str) -> float:
        if not self._trained or self._total_instances == 0:
            return 0.0
        return self._class_counts.get(class_value, 0) / self._total_instances

    # The function returns the conditional probably of a feature value assuming a given class value. You can assume
    # that this function simply retrieves the desired probability after training rather than
    # recomputes them from scratch. A value of 0 should be returned if no training took place.
    # At input, it takes:
    # - class_value - the class value on which the feature_value is conditional
    # - feature_name - the name of the feature we want to calculate for
    # - feature_value - the feature value we want to calculate the conditional probability for
    # The function outputs:
    # - probability - float representing the calculated conditional probability
    #
    def retrieve_conditional_probability(self, class_value: str, feature_name: str, feature_value: str) -> float:
        if not self._trained:
            return 0.0
        class_count = self._class_counts.get(class_value, 0)
        if class_count == 0:
            return 0.0
        count = self._conditional_counts.get(class_value, {}).get(feature_name, {}).get(feature_value, 0)
        return count / class_count
