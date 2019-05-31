# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os

import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import data_processor
import learner

print(os.listdir("input"))

CSV_COLUMN_NAMES = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                    'Fare', 'Cabin', 'Embarked']
LABEL_COLUMN_NAME = 'Survived'
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "output/submission.csv"
TEST_SET_FRACTION = 0.2


def load_data_for_tf(y_name='Survived'):
    """Returns the dataset as (train_x, train_y), (test_x, test_y)."""

    # The given "test.csv" is for predictions
    train_data = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    prediction_data = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES.remove(LABEL_COLUMN_NAME), header=0)

    # Clean the train_data data
    # Remove the id and ticket number columns as they are not relevant for predicting survival
    train_data.pop('PassengerId')
    train_data.pop('Ticket')
    # Transform cabin into 1 for "has a cabin" and 0 otherwise (thus replacing na)
    transform_cabin(train_data)
    transform_name(train_data)
    # TODO check this does not improve accuracy
    # There are many age = na , replace them with average to keep information from other columns
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    train_data.dropna(inplace=True)

    # Clean prediction data
    # Remove ticket number columns as they are not relevant for predicting survival
    prediction_data.pop('Ticket')
    transform_cabin(prediction_data)
    transform_name(prediction_data)
    prediction_data['Age'].fillna(prediction_data['Age'].mean(), inplace=True)

    # Before splitting we need to shuffle it to get a random order
    train_data.sample(frac=1)

    # Split the data set into training set and test set (validation) to improve extrapolation
    train_set_size = int(round(len(train_data) * (1 - TEST_SET_FRACTION)))
    train = train_data[0:train_set_size - 1]
    test = train_data[train_set_size:train_data.size - 1]

    # Split features and labels
    train_x, train_y = train, train.pop(y_name)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y), prediction_data


def load_data_for_knn():
    train_column_names = list(CSV_COLUMN_NAMES)
    prediction_column_names = list(CSV_COLUMN_NAMES)
    prediction_column_names.remove(LABEL_COLUMN_NAME)

    # Manually remove some irrelevant columns
    train_column_names.remove('PassengerId')
    train_column_names.remove('Ticket')
    prediction_column_names.remove('Ticket')

    train_data = pd.read_csv(TRAIN_PATH, names=train_column_names, header=0)
    prediction_data = pd.read_csv(TEST_PATH, names=prediction_column_names, header=0)

    transform_cabin(train_data)
    transform_name(train_data)
    transform_cabin(prediction_data)
    transform_name(prediction_data)

    # There are many age = na , replace them with average to keep information from other columns
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    train_data.dropna(inplace=True)

    prediction_data['Age'].fillna(train_data['Age'].mean(), inplace=True)

    # Before splitting we need to shuffle it to get a random order
    train_data.sample(frac=1)

    # Split the data set into training set and test set (validation) to improve extrapolation
    train_set_size = int(round(len(train_data) * (1 - TEST_SET_FRACTION)))
    train = train_data[0:train_set_size - 1]
    test = train_data[train_set_size:train_data.size - 1]

    return train, test, prediction_data


def transform_cabin(df):
    df['Cabin'] = df['Cabin'].fillna('')
    df['Cabin'] = df['Cabin'].transform(lambda c: cabin_lambda(c))


def cabin_lambda(cabin):
    if cabin == '':
        return 0
    else:
        # return cabin.count(' ') + 1
        return 1
        # return ord(cabin[0])


def transform_name(df):
    df['Name'] = df['Name'].fillna('None')
    df['Name'] = df['Name'].transform(lambda c: name_lambda(c))


def name_lambda(name):
    for title in ['Mr', 'Miss', 'Mrs', 'Dr', 'Master', 'Pr']:
        if title in name:
            return title

    return 'None'


def submit(x, y):
    submission = pd.DataFrame({"PassengerId": x["PassengerId"], "Survived": y})
    submission.to_csv(SUBMISSION_PATH, index=False)


# **************************** Algos *************************** #
def use_tensorflow():
    # Load the data
    (train_x, train_y), (test_x, test_y), predict_x = load_data_for_tf()

    # Extract useful features
    my_feature_columns = data_processor.extract_tensorflow_features(train_x)

    my_classifier_1 = learner.tensorflow_dnn(my_feature_columns, train_x, train_y, test_x, test_y)
    my_classifier_2 = learner.tensorflow_linear(my_feature_columns, train_x, train_y, test_x, test_y)

    # Predict y using our classifier
    my_predictions_y_1, probabilities_1 = learner.tensorflow_predict_y(my_classifier_1, predict_x)
    my_predictions_y_2, probabilities_2 = learner.tensorflow_predict_y(my_classifier_2, predict_x)

    (my_predictions_y, probabilities) = learner.aggregate_predictions(my_predictions_y_1, my_predictions_y_2,
                                                                      probabilities_1, probabilities_2)

    print('predictions: \t', my_predictions_y)

    return my_predictions_y


def use_sklearn_knn(train_x, train_y, test_x, test_y, predict_x):
    x_train_num = data_processor.transform_to_numeric_columns(train_x)
    x_test_num = data_processor.transform_to_numeric_columns(test_x)
    x_predict_num = data_processor.transform_to_numeric_columns(predict_x)

    print('x_train_num.shape() : \t', x_train_num.shape)
    print('x_predict_num.shape() : \t', x_predict_num.shape)

    # my_classifier = learner.sklearn_knn(x_train_num, train_y, x_test_num, test_y)
    #
    # my_predictions_y = learner.sklearn_knn_predict(my_classifier, x_predict_num)
    #
    # print('predictions: \t', my_predictions_y)

    return my_predictions_y


# **************************** Main *************************** #
def main():
    # Train & predict using tensorflow
    # my_predictions_y = use_tensorflow()

    # Train & predict using sklearn knn
    my_predictions_y = use_sklearn_knn(my_train_x, my_train_y, my_test_x, my_test_y, my_predict_x)

    # Submit
    if my_predictions_y:
        submit(my_predict_x, my_predictions_y)


if __name__ == "__main__":
    main()
