from collections import defaultdict

import math
import pandas as pd

from text_preprocess import TextProcessManager
from group_combiner import GroupCombiner


def simple_process(origin_names):
    text_handler = TextProcessManager()
    gc = GroupCombiner()
    res = defaultdict(list)
    for item in origin_names:
        n_key = text_handler.proceed_text(item)
        special_key = gc.search_key_groups(n_key)
        res[special_key].append(item)
    res = gc.create_etc(res)
    return res


def create_list_dfs(data_frame):
    data_frame.name = data_frame.name.apply(lambda x: x.lower().strip())
    names = set(data_frame.name)
    groups_name = simple_process(names)
    df_list = []
    for key, value in groups_name.items():
        df_list.append(data_frame[data_frame.name.isin(value)].drop(['Unnamed: 0'], axis=1))
    return df_list


def get_average_salary_interval(group):
    cities = set(group.city)
    salary_city = dict()
    for city in cities:
        city_group = group[group.city == city]
        salary_city[city] = {
            'from': city_group.salary_from.mean(),
            'to': city_group.salary_to.mean()
        }
    return salary_city


def get_valid_salary(group, global_salary):
    salary_city = get_average_salary_interval(group)
    for city, salary in salary_city.items():
        avg_salary_city = global_salary[city]
        is_nan_from = math.isnan(salary['from'])
        is_nan_to = math.isnan(salary['to'])
        if is_nan_to or is_nan_from:
            if is_nan_from and is_nan_to:
                salary = {'from': avg_salary_city['from'], 'to': avg_salary_city['to']}
            else:
                from_salary = salary['from']
                to_salary = salary['to']
                salary['from'] = to_salary if is_nan_from else from_salary
                salary['to'] = from_salary if is_nan_to else to_salary
        if salary['from'] > salary['to']:
            salary['to'] = salary['from']
        salary_city[city] = salary
    return salary_city


def set_proceed_salary_value(group, city, salary_field, value):
    group.loc[
        (group.city == city) & (group[salary_field] != group[salary_field]),
        salary_field
    ] = value


def fill_salary_gaps(groups, global_salary):
    for group in groups:
        salary_city = get_valid_salary(group, global_salary)
        for city, salary in salary_city.items():
            for sf in salary:
                set_proceed_salary_value(group, city, f'salary_{sf}', salary[sf])


if __name__ == '__main__':
    df = pd.read_csv("./data/vacancies.csv")
    global_avg_salary = get_average_salary_interval(df)
    job_groups = create_list_dfs(df)
    fill_salary_gaps(job_groups, global_avg_salary)
