import pandas as pd
import plotly.express as px
from numpy import percentile
from sklearn import svm


def drop_columns(data_frame):
    return data_frame[['salary_from', 'salary_to', 'count_days']].dropna(axis=0)
    # return data_frame.drop(
    #     ['responsibility', 'conditions', 'requirement', 'description', 'name', 'city',
    #      'employer', 'published_at', 'experience', 'employment', 'schedule', 'key_skills', 'group_name',
    #      ], axis=1).dropna(axis=0)


def count_outliers(df):
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(df)
    y_pred_train = clf.predict(df)
    n_error_train = y_pred_train[y_pred_train == -1].size
    print(n_error_train)
    return [i for i, elem in enumerate(y_pred_train.tolist()) if elem == -1]


def build_box(df):
    fig = px.box(df, y="salary_to", notched=True)
    fig.show()
    fig = px.box(df, y="salary_from", notched=True)
    fig.show()


def field_outliers(df, field):
    data = df[field]
    q25, q75 = percentile(data, 25), percentile(data, 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers = [x for x in data if x < lower or x > upper]

    def app(x):
        if x in outliers:
            return lower if lower > x else upper
        return x

    df[field] = df[field].apply(app)
    return df


if __name__ == '__main__':
    data_frame = drop_columns(pd.read_csv('./data/train.csv').drop(['Unnamed: 0'], axis=1))
    bad_obj = count_outliers(data_frame)
    df_field = field_outliers(data_frame, 'salary_to')
    df_field = field_outliers(df_field, 'salary_from')
    bad_obj2 = count_outliers(df_field)
    rem_obj = [b1 for b1, b2 in zip(bad_obj, bad_obj2) if b1 == b2]
    data_frame.drop(data_frame.index[[rem_obj]]).to_csv('./data/train_proc.csv')