import os
import re
from collections import Counter

import pandas as pd


def get_norm_data(column):
    _min = column.min()
    _max = column.max()
    return (column - _min) / (_max - _min)


def conversion_salary_to_one_hot(data_frame, field_name, cnt_group):
    series = data_frame[field_name]
    _max = int(series.max() + 1)
    _min = int(series.min())
    bins = pd.cut(
        series, list(range(_min, _max, (_max - _min) // (cnt_group - 1))),
        right=False
    )
    cols = pd.crosstab(data_frame.index, bins)
    for name in cols:
        data_frame[f"{field_name}: {name}"] = cols[name]


def conversion_categories_to_one_hot(data_frame, field_name):
    cols = pd.crosstab(data_frame.index, data_frame[field_name])
    for name in cols:
        data_frame[f"{field_name}: {name}"] = cols[name]


def get_top_skills(data_frame, n):
    skills = list(data_frame.key_skills.apply(lambda x: x.split("|")))
    flatten = [item for line in skills for item in line]
    cnt = Counter(flatten)
    return [item[0] for item in cnt.most_common(n)]


def dummy_skills(data_frame):
    top_skills = get_top_skills(data_frame, 10)
    data_frame['list_skills'] = data_frame.key_skills.apply(lambda x: x.split("|"))
    for skill in top_skills:
        data_frame[skill] = data_frame.list_skills.apply(lambda x: int(skill in x))
    data_frame.drop(['list_skills'], axis=1)
    data_frame['etc'] = data_frame.apply(lambda row: int(sum(list(row.loc[top_skills])) == 0), axis=1)


def delete_bad_text(data_frame):
    for column_name in ("responsibility", "conditions", "requirement", "description"):
        data_frame[column_name] = data_frame[column_name].apply(lambda x: re.sub(r'&quot', '', x))


def split_data_frame(data_frame):
    name_groups = set(data_frame['group_name'])
    return dict([(gr_name, data_frame.loc[df['group_name'] == gr_name]) for gr_name in name_groups])


if __name__ == '__main__':
    df = pd.read_csv('./data/fill_data.csv')
    dummy_skills(df)
    df.count_days = get_norm_data(df.count_days)
    conversion_salary_to_one_hot(df, "salary_to", 7)
    conversion_salary_to_one_hot(df, "salary_from", 6)
    for field in ('city', 'experience', 'employment', 'schedule'):
        conversion_categories_to_one_hot(df, field)
    result_groups = split_data_frame(df)
    # for key, group in result_groups.items():
    #     dummy_skills(group)
    for name_gr, val in result_groups.items():
        res = val.drop(['Unnamed: 0'], axis=1).drop(['list_skills'], axis=1)
        res.to_csv(f'./result/pre_process/{name_gr}.csv')
