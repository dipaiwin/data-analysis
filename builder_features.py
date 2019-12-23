from collections import Counter

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./data/train.csv')
    skills = df.key_skills.apply(lambda x: [item.lower() for item in x.split("|")]).tolist()
    flatten = [item for line in skills for item in line]
    cnt = Counter(flatten)
    cnt_data = len(df)
    common_skills = set([key for key, value in cnt.items() if value / cnt_data >= 0.1])
    avg_salary_groups = df.salary_to.mean()
    for file, save in zip(('./data/train.csv', './train_data/test_full.csv'), ('train2.csv', 'test2.csv')):
        main_df = pd.read_csv(file)
        main_df = main_df.loc[main_df.key_skills != '']
        main_df['list_skills'] = main_df.key_skills.apply(lambda x: [item.lower() for item in x.split("|")])
        main_df['coef'] = main_df.list_skills.apply(lambda x: len(common_skills & set(x)) ** 2 / (len(x)))
        main_df = main_df.drop(['list_skills'], axis=1)
        main_df.to_csv(f'./data/{save}', index=False)
