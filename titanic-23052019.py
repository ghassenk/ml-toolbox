# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
TRAIN_PATH = "../input/train.csv"
TEST_PATH = "../input/test.csv"
SUBMISSION_PATH = "./submission.csv"

# TODO extract column names from the csv
CSV_COLUMN_NAMES = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                    'Fare', 'Cabin', 'Embarked']
CSV_COLUMN_NAMES_SUBMISSON = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                              'Fare', 'Cabin', 'Embarked']

TEST_SET_FRACTION = 0.2

BATCH_SIZE = 10
TRAIN_STEPS = 100


# **************************** Data processor functions *************************** #
def extract_tensorflow_features(df):
    features = []
    i = 0
    for dtype in df.dtypes:
        col_name = df.columns[i]
        # TODO check if it is better to bucketize or categorize int type
        if dtype is np.dtype(int) or dtype is np.dtype(float):
            # Bucketize numerical features
            print('bucketizing ', col_name, ' ...')
            num_column = tf.feature_column.numeric_column(key=col_name)
            value_boundaries = [0, df[col_name].max() * 0.25, df[col_name].max() * 0.5,
                                df[col_name].max() * 0.75, df[col_name].max()]
            bucketized_column = tf.feature_column.bucketized_column(source_column=num_column,
                                                                    boundaries=value_boundaries)
            features.append(bucketized_column)

        else:
            # Categorize other features and add them as wrapped columns
            print('categorizing ', col_name, ' ...')
            column = tf.feature_column.categorical_column_with_vocabulary_list(
                key=col_name, vocabulary_list=list(df[col_name].unique()))
            features.append(tf.feature_column.indicator_column(column))

        i += 1

    return features


# **************************** Learner functions *************************** #
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def predict_input_fn(features, batch_size):
    """An input function for training"""

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features)))
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def tensorflow_nn(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)

    return model


def tensorflow_dnn(tf_feature_columns, train_x, train_y, x_test, y_test, batch_size=10, train_steps=100):
    # Check if dnn can be improved by changing architecture
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=tf_feature_columns,
        # ? hidden layers of ? nodes each.
        hidden_units=[10, 10],
        # The model must choose between 2 classes.
        n_classes=2)
    # Train
    dnn_classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=train_steps)
    # Eval
    eval_dnn = dnn_classifier.evaluate(input_fn=lambda: eval_input_fn(x_test, y_test, batch_size))
    print('\nDNNClassifier Test set accuracy: {accuracy:0.3f}\n'.format(**eval_dnn))

    return dnn_classifier


def tensorflow_linear(tf_feature_columns, train_x, train_y, x_test, y_test, batch_size=10, train_steps=100):
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=tf_feature_columns)

    # Train
    linear_classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps=train_steps)

    # Eval
    eval_linear = linear_classifier.evaluate(input_fn=lambda: eval_input_fn(x_test, y_test, batch_size))
    print('LinearClassifier Test set accuracy: {accuracy:0.3f}\n'.format(**eval_linear))

    return linear_classifier


def predict_y(classifier, predict_x, batch_size=10):
    predictions = classifier.predict(input_fn=lambda: predict_input_fn(predict_x, batch_size))

    predictions_y = []
    predictions_y_probabilities = []
    for p in predictions:
        predictions_y_probabilities.append(p['probabilities'].max())
        predictions_y.append(p['class_ids'][0])

    return predictions_y, predictions_y_probabilities


def aggregate_predictions(predictions_y_1, predictions_y_2, probabilities_1, probabilities_2):
    i = 0
    agg_predictions = []
    agg_probabilities = []
    for _ in predictions_y_1:
        if probabilities_1[i] >= probabilities_2[i]:
            agg_predictions.append(predictions_y_1[i])
            agg_probabilities.append(probabilities_1[i])
        else:
            agg_predictions.append(predictions_y_2[i])
            agg_probabilities.append(probabilities_2[i])
        i += 1

    # print('dnn: \t', probabilities_1)
    # print('lin: \t', probabilities_2)
    # print('agg: \t', agg_probabilities)

    return aggregate_predictions, agg_probabilities


# **************************** Local functions  *************************** #
def load_data(y_name='Survived'):
    """Returns the dataset as (train_x, train_y), (test_x, test_y)."""

    # The given "test.csv" is for predictions
    train_data = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    prediction_data = pd.read_csv(TEST_PATH, names=CSV_COLUMN_NAMES_SUBMISSON, header=0)

    # Clean the data
    clean_training_data(train_data)
    clean_prediction_data(prediction_data)

    # Plot some features
    #  plot_features(train_data)

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


def clean_training_data(input_data):
    # Remove the id and ticket number columns as they are not relevant for predicting survival
    input_data.pop('PassengerId')
    input_data.pop('Ticket')

    # Transform cabin into 1 for "has a cabin" and 0 otherwise (thus replacing na)
    transform_cabin(input_data)
    transform_name(input_data)

    # TODO check this does not improve accuracy
    # There are many age = na , replace them with average to keep information from other columns
    input_data['Age'].fillna(input_data['Age'].mean(), inplace=True)
    input_data.dropna(inplace=True)

    return


def clean_prediction_data(input_data):
    # Remove ticket number columns as they are not relevant for predicting survival
    input_data.pop('Ticket')

    transform_cabin(input_data)
    transform_name(input_data)

    input_data['Age'].fillna(input_data['Age'].mean(), inplace=True)

    return


def plot_features(input_data):
    sns.set()
    sns.relplot(y='Survived', x='Fare', kind="line", data=input_data)

    # sns.countplot(x='Cabin', hue='Survived', data=input_data)
    # sns.countplot(x='Sex', hue='Survived', data=input_data)
    # sns.countplot(x='Embarked', hue='Survived', data=input_data)
    # sns.countplot(x='SibSp', hue='Survived', data=input_data)
    #  sns.countplot(x='Parch', hue='Survived', data=input_data)
    # sns.countplot(x='Pclass', hue='Survived', data=input_data)

    plt.show()

    return


def print_feature_importance(input_data):
    print(input_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived',
                                                                                                    ascending=False))
    print(input_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',
                                                                                              ascending=False))
    print(input_data[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived',
                                                                                                ascending=False))
    print(input_data[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived',
                                                                                                  ascending=False))
    print(input_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived',
                                                                                                        ascending=False))
    print(input_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',
                                                                                                  ascending=False))
    print(input_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',
                                                                                                  ascending=False))
    return


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


# **************************** Main *************************** #
def main():
    # Load the data
    (my_train_x, my_train_y), (my_test_x, my_test_y), my_predict_x = load_data()

    # Extract useful features
    my_feature_columns = extract_tensorflow_features(my_train_x)

    my_classifier_1 = tensorflow_dnn(my_feature_columns, my_train_x, my_train_y, my_test_x, my_test_y)
    my_classifier_2 = tensorflow_linear(my_feature_columns, my_train_x, my_train_y, my_test_x, my_test_y)

    # Predict y using our classifier
    my_predictions_y_1, probabilities_1 = predict_y(my_classifier_1, my_predict_x)
    my_predictions_y_2, probabilities_2 = predict_y(my_classifier_2, my_predict_x)

    my_predictions_y, probabilities = aggregate_predictions(my_predictions_y_1, my_predictions_y_2, probabilities_1, probabilities_2)

    # Submit
    submit(my_predict_x, my_predictions_y)


if __name__ == "__main__":
    main()
