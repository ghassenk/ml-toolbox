import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CSV_COLUMN_NAMES = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
                    'Fare', 'Cabin', 'Embarked']
LABEL_COLUMN_NAME = 'Survived'
TRAIN_PATH = "input/train.csv"
TEST_PATH = "input/test.csv"
SUBMISSION_PATH = "output/submission.csv"


def main():
    train_column_names = list(CSV_COLUMN_NAMES)
    prediction_column_names = list(CSV_COLUMN_NAMES)
    prediction_column_names.remove(LABEL_COLUMN_NAME)

    # Remove some irrelevant columns
    # train_column_names.remove('PassengerId')
    # train_column_names.remove('Ticket')

    train_data = pd.read_csv(TRAIN_PATH, names=CSV_COLUMN_NAMES, header=0)
    prediction_data = pd.read_csv(TEST_PATH, names=prediction_column_names, header=0)

    correlation = train_data.corr()
    cor_target = abs(correlation[LABEL_COLUMN_NAME])
    relevant_features = cor_target[cor_target > 0.5]
    print(correlation[LABEL_COLUMN_NAME])

    pass


if __name__ == "__main__":
    main()
