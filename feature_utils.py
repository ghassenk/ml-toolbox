import numpy as np
from sklearn.preprocessing import LabelEncoder


def transform_into_numerical_columns(df, drop_na=True):
    """ Transform into numerical columns """
    df_copy = df.copy()
    if drop_na:
        df_copy.dropna(inplace=True)

    i = 0
    for d_type in df_copy.dtypes:
        if d_type is np.dtype(int):
            pass
        elif d_type is np.dtype(float):
            pass
        else:
            col_name = df_copy.columns[i]
            enc = LabelEncoder()
            try:
                print('transforming ', df_copy.columns[i], 'of type : ', d_type)
                enc.fit(df_copy[col_name])
                df_copy[col_name] = enc.transform(df_copy[col_name])
            except Exception as e:
                print('failed to transform ', col_name, ' ', e)
                pass

        i += 1

    return df_copy


# TODO not really working
def bucketize(df, nb_buckets):
    df_copy = df.copy()

    for col_name in df_copy.columns:
        if df_copy[col_name].unique().size > nb_buckets:
            max_col, min_col = df_copy[col_name].max(), df_copy[col_name].min()
            df_copy[col_name] = df_copy[col_name].transform(lambda v: __bucket_lambda(v, min_col, max_col, nb_buckets))

    return df_copy


def __bucket_lambda(value, min_col, max_col, nb_buckets):
    step = (max_col - min_col) / nb_buckets
    for i in range(1, nb_buckets):
        if value <= (min_col + step * i):
            return i
