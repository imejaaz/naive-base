# This is just a simple function to execute the code in general
# This file does not need to be submitted. Fiddle with it any way you see fit.

from cleveland_data_preprocessor import *
from task_1_naive_bayes import NaiveBayes
from task_2_evaluation import evaluate_classification
from task_3_cross_validation import cross_validate

# We load the dataset. Change the path as you see fit.
dataset_path = "../processed_cleveland_dataset.csv"
dataset = read_data(dataset_path)

# That's the name of the class variable in this dataset
class_name = 'target'

# We preprocess the dataset. This includes splitting it into training and testing data, cleaning, discretising, etc.
# The process used here is ugly. Don't worry about it. You will be improving on it in a different assessment :)
training_data, testing_data = preprocess(dataset, class_name)

# We extract some data for booting up our classifier
full_data = pd.concat([training_data, testing_data])
feature_info = {col: sorted(full_data[col].unique().tolist()) for col in full_data.columns}
class_values = feature_info.pop(class_name, None)

# Classifier is created, trained, and predictions are made
nb_classifier = NaiveBayes((class_name, class_values), feature_info)
nb_classifier.train_model(training_data)
classified_data = nb_classifier.predict(testing_data)

# Classifier is evaluated
metrics = evaluate_classification(classified_data[class_name], classified_data['PredictedClass'], class_values)
print(metrics)

# Classifier is evaluated using cross-validation
cv_results = cross_validate(nb_classifier, training_data, 5)
print(cv_results)