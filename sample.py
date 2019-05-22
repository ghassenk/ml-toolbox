import pandas as pd

import learner

CSV_COLUMN_NAMES = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                    'Fare', 'Cabin', 'Embarked']
LABEL_COLUMN_NAME = 'Survived'
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "output/submission.csv"
TEST_SET_FRACTION = 0.2


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


def clean_data(train_data, prediction_data):
    transform_cabin(train_data)
    transform_name(train_data)
    transform_cabin(prediction_data)
    transform_name(prediction_data)

    # There are many age = na , replace them with average to keep information from other columns
    train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    train_data.dropna(inplace=True)

    prediction_data['Age'].fillna(train_data['Age'].mean(), inplace=True)


def split_data(train_data, test_set_fraction):
    # Before splitting we need to shuffle it to get a random order
    train_data.sample(frac=1)

    # Split the data set into training set and test set (validation) to improve extrapolation
    train_set_size = int(round(len(train_data) * (1 - test_set_fraction)))
    train = train_data[0:train_set_size - 1]
    test = train_data[train_set_size:train_data.size - 1]

    # Split features and labels
    train_x, train_y = train, train.pop(LABEL_COLUMN_NAME)
    test_x, test_y = test, test.pop(LABEL_COLUMN_NAME)

    return (train_x, train_y), (test_x, test_y)


def main():
    train_column_names = list(CSV_COLUMN_NAMES)
    prediction_column_names = list(CSV_COLUMN_NAMES)
    prediction_column_names.remove(LABEL_COLUMN_NAME)

    # Manually remove some irrelevant columns
    train_column_names.remove('PassengerId')
    train_column_names.remove('Ticket')
    prediction_column_names.remove('Ticket')

    train_data = pd.read_csv(TRAIN_PATH, names=train_column_names, header=0)
    prediction_data = pd.read_csv(TEST_PATH, names=prediction_column_names, header=0)

    clean_data(train_data, prediction_data)

    (my_train_x, my_train_y), (my_test_x, my_test_y) = split_data(train_data, TEST_SET_FRACTION)
    predict_x = prediction_data

    model = learner.tensorflow_nn(my_train_x, my_train_y, my_test_x, my_test_y)

    return


if __name__ == "__main__":
    main()
