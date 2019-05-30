# P(Y\X) = P(Y) * P(X\Y) / P(X)
# P(X) = P(x1) * P(x2) * ... * P(xn)            (true only if features are independent)
# P(X\Y) = P(x1\Y) * P(x2\Y) * ... * P(xn\Y)    (true only if features are independent)

import pandas as pd

CSV_COLUMN_NAMES = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                    'Fare', 'Cabin', 'Embarked']
LABEL_COLUMN_NAME = 'Survived'
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "output/submission.csv"


def transform_age(df):
    # There are many age = na , replace them with average to keep information from other columns
    mean_age = int(df['Age'].dropna(inplace=False).mean())
    df['Age'] = df['Age'].fillna(mean_age)
    df['Age'] = df['Age'].transform(lambda c: age_lambda(c))


def age_lambda(age):
    if age <= 20:
        return 'A'
    elif age <= 40:
        return 'B'
    elif age <= 60:
        return 'C'
    else:
        return 'D'


def transform_fare(df):
    # There are many age = na , replace them with average to keep information from other columns
    max_fare = df['Fare'].max()
    mean_fare = df['Fare'].dropna(inplace=False).mean()
    df['Fare'] = df['Fare'].fillna(mean_fare)
    df['Fare'] = df['Fare'].transform(lambda c: fare_lambda(c, max_fare))


def fare_lambda(fare, max_fare):
    if fare <= max_fare * 0.25:
        return 'A'
    elif fare <= max_fare * 0.5:
        return 'B'
    elif fare <= max_fare * 0.75:
        return 'C'
    else:
        return 'D'


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
    transform_cabin(prediction_data)
    transform_name(train_data)
    transform_name(prediction_data)
    transform_age(train_data)
    transform_age(prediction_data)
    transform_fare(train_data)
    transform_fare(prediction_data)

    train_data.dropna(inplace=True)


def main():
    prediction_column_names = list(CSV_COLUMN_NAMES)
    prediction_column_names.remove(LABEL_COLUMN_NAME)

    train_data = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    prediction_data = pd.read_csv(TEST_PATH, names=prediction_column_names, header=0)

    # Manually remove some irrelevant columns
    train_data.pop('PassengerId')
    train_data.pop('Ticket')
    prediction_data.pop('PassengerId')
    prediction_data.pop('Ticket')

    #print(train_data)

    clean_data(train_data, prediction_data)

    #print(train_data)

    for col in train_data.columns:
        # print(train_data[col])
        print(train_data[col].value_counts() / train_data[col].size)
        pass

        # x_columns = list(CSV_COLUMN_NAMES)
        # x_columns.remove(LABEL_COLUMN_NAME)
        #
        # df = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
        #
        # # TODO temp
        # df.dropna(inplace=True)
        # #print(df)
        #
        # X = df[x_columns]
        # Y = df[LABEL_COLUMN_NAME]
        #
        # P_Y = pd.DataFrame(Y.value_counts() / Y.size)
        # P_X = pd.DataFrame()
        # for col in x_columns:
        #     print(X[col].value_counts())
        #     #P_X[col] = X[col].value_counts() / X[col].size
        #
        # print(P_X)


if __name__ == "__main__":
    main()
