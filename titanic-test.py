import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import feature_utils

CSV_COLUMN_NAMES = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                    'Fare', 'Cabin', 'Embarked']
LABEL_COLUMN_NAME = 'Survived'
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "output/submission.csv"
CROSS_VALIDATION_SET_FRACTION = 0.2


def load_data():
    test_column_names = list(CSV_COLUMN_NAMES)
    test_column_names.remove(LABEL_COLUMN_NAME)

    train_data = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    test_data = pd.read_csv(TEST_PATH, names=test_column_names, header=0)

    return train_data, test_data


def main():
    train_data, test_data = load_data()

    train_data['Cabin'].fillna('None', inplace=True)
    train_data['Embarked'].fillna('None', inplace=True)

    df2 = feature_utils.transform_into_numerical_columns(train_data, False)
    print(df2)

    pass


if __name__ == "__main__":
    main()
