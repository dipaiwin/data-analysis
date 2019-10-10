import re
from datetime import datetime

import pandas as pd
import os
import math


def to_dict(date_frame, left_value='', right_value=''):
    min_sal = math.ceil(left_value) if isinstance(left_value, float) else left_value
    max_sal = math.ceil(right_value) if isinstance(right_value, float) else right_value
    return {'data': date_frame, 'left_value': min_sal, 'right_value': max_sal}


def separate_data_frame(data_frame, cnt_groups):
    right_value = data_frame['salary_to'].max()
    left_value = data_frame['salary_to'].min()
    step = (right_value - left_value) / (cnt_groups - 1)
    df_list = []
    lower_value = 0
    nan_group = to_dict(data_frame[data_frame['salary_to'].isnull()], 'NAN', 'NAN')
    for i in range(0, cnt_groups - 1):
        upper_value = left_value + step * i
        select = data_frame.loc[(data_frame['salary_to'] < upper_value) & (data_frame['salary_to'] >= lower_value)]
        df_list.append(to_dict(select, lower_value, upper_value))
        lower_value = upper_value
    df_list.append(to_dict(data_frame.loc[(data_frame['salary_to'] >= lower_value)], left_value=lower_value))
    df_list.append(nan_group)
    return df_list


def count_skills(group, ):
    skills = dict()
    for skill in group.key_skills.values:
        if skill == skill:
            skills_list = skill.split('|')
            for val in skills_list:
                val = val.strip()
                if val not in skills:
                    skills[val] = 0
                skills[val] += 1
    keys, values = [], []
    for key, value in skills.items():
        keys.append(key)
        values.append(value)
    return pd.DataFrame(data=values, index=keys)


def count_days(group):
    a = pd.to_datetime(group.published_at).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    delta_day = [date.days for date in datetime.now() - pd.to_datetime(a)]
    return pd.DataFrame(data=[sum(delta_day) / len(delta_day), max(delta_day), min(delta_day)],
                        index=['avg', 'max', 'min'])


def write_data(record, item_folder):
    data, keys = [], []
    for i, (key, value) in enumerate(record.items()):
        data.append(value)
        keys.append(key)
        data.append(pd.Series({'': ''}))
        keys.append(' ' * i)
    pd.concat(data, keys=keys).to_csv(item_folder, sep='\t', header=None)


def calculate_main_statistic(record, data, item, path_to_result):
    record['days'] = count_days(data)
    record['experience'] = data.experience.value_counts()
    record['employment'] = data.employment.value_counts()
    record['schedule'] = data.schedule.value_counts()
    record['skills'] = count_skills(data)
    left_value = item['left_value']
    right_value = item['right_value']
    item_path = os.path.join(path_to_result, f"{left_value} {right_value}")
    os.mkdir(item_path)
    write_data(record, os.path.join(item_path, "analysis.csv"))
    data.to_csv(os.path.join(item_path, "data.csv"))


def calculate_statistic(dfs, path_to_result):
    for item in dfs:
        data = item['data']
        if len(data) != 0:
            record = dict()
            record['vacancies_name'] = data.name.value_counts()
            calculate_main_statistic(record, data, item, path_to_result)


def calculate_statistic_vacancies(origin_data, salary_separate_list, path_to_result):
    vacancies = set(origin_data.name)
    separate_vacancies = []
    for name in vacancies:
        separate_vacancies.append(to_dict(origin_data.loc[(origin_data.name == name)], name))
    for item in separate_vacancies:
        data = item['data']
        if len(data) != 0:
            record = dict()
            border_salary = dict()
            for salary in salary_separate_list:
                key = f'{salary["left_value"]} - {salary["right_value"]}'
                cnt = len(salary['data'][salary['data'].name == item['left_value']])
                if cnt != 0:
                    border_salary[key] = cnt
            record['salarys'] = pd.Series(border_salary)
            calculate_main_statistic(record, data, item, path_to_result)


if __name__ == "__main__":
    df = pd.read_csv('./data/vacancies.csv')
    df.name = df.name.apply(lambda x: re.sub(r'[\s/|\\-]+', ' ', x.lower().strip()))
    df = df.sort_values(by=['salary_to', 'salary_from'])
    dfs_list = separate_data_frame(df, 10)
    # calculate_statistic(dfs_list, './result/task1')
    calculate_statistic_vacancies(df, dfs_list, './result/task2')
