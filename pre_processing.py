import math
import os
import shutil
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd

from processing_for_clustering.group_combiner import GroupCombiner
from processing_for_clustering.salary_handler import SalaryHandler
from processing_for_clustering.text_preprocess import TextProcessManager


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
    df_list = dict()
    for key, value in groups_name.items():
        df_list[key] = (data_frame[data_frame.name.isin(value)].drop(['Unnamed: 0'], axis=1))
    return df_list


def count_days(groups):
    for _, group in groups.items():
        date_group = pd.to_datetime(group.published_at).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        group['count_days'] = [date.days for date in datetime.now() - pd.to_datetime(date_group)]


def fill_gap_description(data_frame):
    values = dict([(key, 'empty') for key in ('responsibility', 'conditions', 'requirement', 'description')])
    return data_frame.fillna(values)


def calculate_skills(key_skills):
    cnt_skills = Counter()
    for line_skills in key_skills:
        if not isinstance(line_skills, str) and math.isnan(line_skills):
            continue
        tokenize_skills = line_skills.split('|')
        tokenize_skills = [item.strip().lower() for item in tokenize_skills]
        cnt_skills.update(tokenize_skills)
    return cnt_skills


def fill_key_skills(groups):
    result = dict()
    for key_name in groups:
        group = groups[key_name]
        count_skills = calculate_skills(group.key_skills.values)
        values_skills = count_skills.values()
        top_skills = []
        if len(values_skills) == 0:
            top_skills.append('empty')
        else:
            max_cs = max(values_skills)
            min_cs = min(values_skills)
            if min_cs == max_cs:
                top_skills = list(count_skills.keys())[:5]
            else:
                delta = max_cs - min_cs
                for key, value in count_skills.items():
                    norm_value = (value - min_cs) / delta
                    if norm_value >= 0.4:
                        top_skills.append(key)
        result[key_name] = group.fillna({'key_skills': '|'.join(top_skills)})
    return result


def add_group_name(groups):
    for name, group in groups.items():
        group['group_name'] = name


def convert_groups_to_main_df(groups):
    add_group_name(groups)

    pd.concat([group for _, group in groups.items()]).to_csv('./data/fill_data_without_empty_row.csv')


def drop_emtpy_row(data_frame):
    number_columns = len(data_frame.columns)
    drop_list = []
    for i in data_frame.index:
        coef_fill_column = 1 - data_frame.iloc[i].isnull().sum() / number_columns
        if coef_fill_column < 0.7:
            drop_list.append(i)
    return data_frame.drop(drop_list)


if __name__ == '__main__':
    save_path = './result/groups_6'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    df = pd.read_csv("./data/big_vacancies.csv")
    df = drop_emtpy_row(df)
    df = fill_gap_description(df)
    global_avg_salary = SalaryHandler.get_average_salary_interval(df)
    job_groups = create_list_dfs(df)
    SalaryHandler.fill_salary_gaps(job_groups, global_avg_salary)
    count_days(job_groups)
    result_groups = fill_key_skills(job_groups)
    for name, val in result_groups.items():
        name_file = f"{' '.join(name) if isinstance(name, frozenset) else name}.csv"
        val.to_csv(os.path.join(save_path, name_file))
    convert_groups_to_main_df(result_groups)
