from collections import defaultdict

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


if __name__ == '__main__':
    df = pd.read_csv("./data/vacancies.csv")
    names = set(df.name.apply(lambda x: x.lower().strip()))
    print(f'Origin len:{len(names)}')
    a = simple_process(names)
    with open('groups.txt', "w") as f:
        for key, value in a.items():
            line_value = '\n\t'.join(value)
            f.write(f'{key} : {len(value)}\n\t{line_value}\n')
