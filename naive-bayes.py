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


def clean_data(df, drop_na):
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

    # Manually remove some irrelevant columns
    train_data.pop('PassengerId')
    train_data.pop('Ticket')
    test_data.pop('PassengerId')
    test_data.pop('Ticket')

    clean_data(train_data, True)
    clean_data(test_data, False)

    return train_data, test_data


def compute_vector_probability(x, column_names, probabilities_matrix):
    product = 1
    for col in column_names:
        product = product * probabilities_matrix[col][x[col]]
        pass

    return product


def main():
    train_data, test_data = load_data()

    # To calculate P(X) we use global X data (occurrence of X values in both train and test set)
    merged_x = pd.concat([train_data, test_data], ignore_index=True, sort=False)

    survived = train_data[train_data["Survived"] == 1]

    y = train_data.pop('Survived')

    probabilities_x = dict()
    probabilities_survived = dict()

    # TODO replace with matrix operations instead of loop
    for col in train_data.columns:
        probabilities_x[col] = (merged_x[col].value_counts() / merged_x[col].size)
        probabilities_survived[col] = (survived[col].value_counts() / survived[col].size)
        pass

    # The matrix containing probabilities for each Y (survived) value
    p_y = (y.value_counts() / y.size)

    # The matrix containing probabilities of each X vector from the train and test set
    p_x = probabilities_x

    # The matrix containing probabilities "knowing survived == true" of each X vector from the train set
    p_x_survived = probabilities_x

    # print('\n****** P(Y) ****** \n\n', p_y)
    print('\n****** P(X) ****** \n\n', p_x)
    # print('\n****** P(X/Y=1) ******\n\n', p_x_survived)

    # Now given the X vectors from the test set, we use our matrix to calculate P(X) the probability of the vector X
    # by multiplying the probability of each of its coordinates
    # TODO replace with matrix operations instead of loop
    for index, row in test_data.iterrows():
        compute_vector_probability(row, test_data.columns, p_x)
        # print(index, ' = ', row)


if __name__ == "__main__":
    main()
