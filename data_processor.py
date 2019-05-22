import numpy as np
from sklearn.preprocessing import LabelEncoder


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


def remove_uncorrelated(df, target_column_name):
    correlation = df.corr().abs()
    # cor_target = correlation[target_column_name]
    # relevant_features = cor_target[cor_target > 0.5]
    print('\n\n', correlation)
    # print('\n\n', cor_target)
