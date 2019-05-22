import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


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


# TODO in progress
def transform_to_numeric_columns(df, drop_na=True):
    df_copy = df.copy()
    if drop_na:
        df_copy.dropna(inplace=True)
    i = 0
    for dtype in df_copy.dtypes:
        if dtype is np.dtype(int):
            pass
        elif dtype is np.dtype(float):
            pass
        else:
            col_name = df_copy.columns[i]
            enc = LabelEncoder()
            try:
                # print('transforming ', df_copy.columns[i], dtype)
                enc.fit(df_copy[col_name])
                df_copy[col_name] = enc.transform(df_copy[col_name])
            except Exception as e:
                print('failed to transform ', col_name, ' ', e)
                pass

        i += 1

    # print(df_copy)
    return df_copy


# TODO in progress
def remove_uncorrelated(df, target_column_name):
    correlation = df.corr().abs()
    # cor_target = correlation[target_column_name]
    # relevant_features = cor_target[cor_target > 0.5]
    print('\n\n', correlation)
    # print('\n\n', cor_target)
