import re
from collections import Counter

import pandas as pd


def get_norm_data(column):
    _min = column.min()
    _max = column.max()
    return (column - _min) / (_max - _min)


def conversion_salary_to_one_hot(data_frame, field_name, cnt_group, range_salary=None):
    series = data_frame[field_name]
    if range_salary is None:
        _max = int(series.max()) + 1
        _min = int(series.min())
        range_salary = list(range(_min, _max, (_max - _min) // (cnt_group - 1)))
    bins = pd.cut(
        series, range_salary,
        right=True
    )
    cols = pd.crosstab(data_frame.index, bins, dropna=False)
    for name in cols:
        data_frame[f"{field_name}: {name}"] = cols[name]
    return range_salary


def conversion_categories_to_one_hot(data_frame, field_name):
    cols = pd.crosstab(data_frame.index, data_frame[field_name])
    for name in cols:
        data_frame[f"{field_name}: {name}"] = cols[name]


def get_top_skills(data_frame, n, return_dict=False):
    skills = list(data_frame.list_skills)
    flatten = [item for line in skills for item in line]
    cnt = Counter(flatten)
    if return_dict:
        return cnt.most_common(n)
    return [item[0] for item in cnt.most_common(n)]


def dummy_skills(data_frame, top_skills=None):
    data_frame['list_skills'] = data_frame.key_skills.apply(lambda x: [item.lower() for item in x.split("|")])
    if top_skills is None:
        top_skills = get_top_skills(data_frame, 150)
    for skill in top_skills:
        data_frame[skill] = data_frame.list_skills.apply(lambda x: int(skill in x))
    data_frame.drop(['list_skills'], axis=1)
    data_frame['etc'] = data_frame.apply(lambda row: int(sum(list(row.loc[top_skills])) == 0), axis=1)
    return top_skills


def delete_bad_text(data_frame):
    for column_name in ("responsibility", "conditions", "requirement", "description"):
        data_frame[column_name] = data_frame[column_name].apply(lambda x: re.sub(r'&quot', '', x))


def split_data_frame(data_frame):
    name_groups = set(data_frame['group_name'])
    return dict([(gr_name, data_frame.loc[df['group_name'] == gr_name]) for gr_name in name_groups])


def write_to_file(data_frame, save_path):
    data_frame = data_frame.drop(['Unnamed: 0'], axis=1).drop(['list_skills'], axis=1)
    data_frame.to_csv(save_path)


def create_test(top_skills, salary_to_list=None, salary_from_list=None):
    test = pd.read_csv('./train_data/test.csv')
    dummy_skills(test, top_skills)
    # conversion_salary_to_one_hot(test, "salary_to", 7, salary_to_list)
    # conversion_salary_to_one_hot(test, "salary_from", 6, salary_from_list)
    print(test.shape)
    write_to_file(test, './train_data/test_full.csv')


if __name__ == '__main__':
    df = pd.read_csv('./data/fill_data_without_empty_row.csv')
    n_top_skills = dummy_skills(df)
    df.count_days = get_norm_data(df.count_days)
    salary_to_range = conversion_salary_to_one_hot(df, "salary_to", 7)
    salary_from_range = conversion_salary_to_one_hot(df, "salary_from", 6)
    for field in ('city', 'experience', 'employment', 'schedule'):
        conversion_categories_to_one_hot(df, field)
    write_to_file(df, './data/train_clst.csv')
    # create_test(n_top_skills)
    # create_test(n_top_skills, salary_to_range, salary_from_range)
    # result_groups = split_data_frame(df)
    # for name_gr, val in result_groups.items():
    #     val.to_csv(f'./groups/extract_feature/{name_gr}.csv')
