import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

CSV_COLUMN_NAMES = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                    'Fare', 'Cabin', 'Embarked']
LABEL_COLUMN_NAME = 'Survived'
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "output/submission.csv"


def transform_sex(df):
    df['Sex'] = df['Sex'].dropna()
    df['Sex'] = df['Sex'].transform(lambda s: sex_lambda(s))


def sex_lambda(sex):
    if sex == 'male':
        return 1
    elif sex == 'female':
        return 2
    else:
        print('Unexpected value for sex : ', sex)
        return 0


def transform_embarked(df):
    df['Embarked'] = df['Embarked'].dropna()
    df['Embarked'] = df['Embarked'].transform(lambda v: embarked_lambda(v))


def embarked_lambda(embarked):
    if embarked == 'S':
        return 1
    elif embarked == 'C':
        return 2
    elif embarked == 'Q':
        return 3
    else:
        print('Unexpected value for embarked : ', embarked)
        return 0


def transform_age(df):
    # There are many age = na , replace them with average to keep information from other columns
    mean_age = int(df['Age'].dropna(inplace=False).mean())
    df['Age'] = df['Age'].fillna(mean_age)
    df['Age'] = df['Age'].transform(lambda v: age_lambda(v))


def age_lambda(age):
    if age <= 20:
        return 1
    elif age <= 40:
        return 2
    elif age <= 60:
        return 3
    else:
        return 4


def transform_fare(df):
    # There are many age = na , replace them with average to keep information from other columns
    max_fare = df['Fare'].max()
    mean_fare = df['Fare'].dropna(inplace=False).mean()
    df['Fare'] = df['Fare'].fillna(mean_fare)
    df['Fare'] = df['Fare'].transform(lambda v: fare_lambda(v, max_fare))


def fare_lambda(fare, max_fare):
    if fare <= max_fare * 0.25:
        return 1
    elif fare <= max_fare * 0.5:
        return 2
    elif fare <= max_fare * 0.75:
        return 3
    else:
        return 4


def transform_cabin(df):
    df['Cabin'] = df['Cabin'].fillna('')
    df['Cabin'] = df['Cabin'].transform(lambda v: cabin_lambda(v))


def cabin_lambda(cabin):
    if cabin == '':
        return 0
    else:
        return 1


def transform_name(df):
    df['Name'] = df['Name'].fillna('')
    df['Name'] = df['Name'].transform(lambda v: name_lambda(v))


def name_lambda(name):
    if 'Mr' in name:
        return 1
    elif 'Miss' in name:
        return 2
    elif 'Mrs' in name:
        return 3
    elif 'Dr' in name:
        return 4
    elif 'Master' in name:
        return 5
    elif 'Pr' in name:
        return 6
    else:
        # Some names have no title
        return 7


def clean_df(df, drop_na):
    transform_sex(df)
    transform_embarked(df)
    transform_cabin(df)
    transform_name(df)
    transform_age(df)
    transform_fare(df)

    if drop_na:
        df.dropna(inplace=True)


def load_data():
    test_column_names = list(CSV_COLUMN_NAMES)
    test_column_names.remove(LABEL_COLUMN_NAME)

    train_data = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    test_data = pd.read_csv(TEST_PATH, names=test_column_names, header=0)

    return train_data, test_data


def clean_data(train_data, test_data):
    train_df = train_data.copy()
    test_df = test_data.copy()

    # Manually remove some irrelevant columns
    train_df.pop('PassengerId')
    train_df.pop('Ticket')
    test_df.pop('PassengerId')
    test_df.pop('Ticket')

    clean_df(train_df, True)
    clean_df(test_df, False)

    return train_df, test_df


def main():
    train_data, test_data = load_data()
    train_df, test_df = clean_data(train_data, test_data)

    print(train_df)

    # Before using knn we separate the survived column (Y) from the X
    x_train_df = train_df
    y_train_df = x_train_df.pop('Survived')

    x_train_array = np.array(x_train_df)
    y_train_array = np.array(y_train_df)

    # X that must be predicted
    x_test_array = np.array(test_df)

    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(x_train_array, y_train_array)

    y_test = knn_classifier.predict(x_test_array)

    submission_df = pd.DataFrame({"PassengerId": test_data["PassengerId"], "Survived": y_test})
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(submission_df)

    pass


if __name__ == "__main__":
    main()
