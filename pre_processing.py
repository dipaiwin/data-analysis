import re
from collections import defaultdict

import nltk
import pandas as pd
from nltk.corpus import stopwords

nltk.download('stopwords')


def simple_process(origin_names):
    stop_words = set(stopwords.words('russian'))
    res = defaultdict(list)
    for item in origin_names:
        txt = re.sub(r'\([^)]+\)', '', item)
        txt = re.sub(r'developer|программист|engineer|разработчик', 'программист', txt)
        txt = re.sub(r'design(er)?', 'дизайнер', txt)
        txt = re.sub(r'front[\s\-]+end|фронт(енд)?', 'frontend', txt)
        txt = re.sub(r'back[\s\-]+end', 'backend', txt)
        txt = re.sub(r'java[\s\-]+script', 'javascript', txt)
        txt = re.sub(r'full[\s\-]+stack', 'fullstack', txt)
        txt = re.sub(r'react\.?(js)?', 'react', txt)
        txt = re.sub(r'.*\.?(js)|angular', 'программиcт javascript', txt)
        txt = re.sub(r'1:*[сc]', ' 1c ', txt)
        txt = re.sub(r'[сc][\s/]+[сc]\+\+|[сc]\+\+[\s/]+[сc]| [сc] |[сc]\+\+', ' c++ ', txt)
        txt = re.sub(r'[сc]#', ' c# ', txt)
        txt = re.sub(r'ведущий|старший|стажер|младший|senior|middle|junior|team(\s+)lead|lead', '', txt)
        txt = re.sub(r'project', 'проекта', txt)
        txt = re.sub(r'битрикс[^ ]+', '1c', txt)
        txt = re.sub(r'продаж[^ ]+|sales', 'продаж', txt)
        txt = re.sub(r'техподдержки', 'поддержки', txt)
        txt = re.sub(r'support', 'поддержки', txt)
        txt = re.sub(r'manager|руководитель', 'менеджер', txt)
        txt = re.sub(r'проектов', 'проекта', txt)
        txt = re.sub(r'product|продукт|продукту|продукта|продакт', 'продукт', txt)
        txt = re.sub(r'администратор[а-я]*|administrator', 'админиcтратор', txt)
        txt = re.sub(r'тест[а-я]*|test[a-z]*', ' тестировщик ', txt)
        txt = re.sub(r'асутп', 'асу', txt)
        txt = re.sub(r'преподаватель|учитель', 'педагог', txt)
        txt = re.sub(r'analyst', 'аналитик', txt)
        txt = re.sub(r'писатель|копирайтер|writer|copywriter', 'автор', txt)
        txt = re.sub(r'3d|2d', 'дизайнер', txt)
        txt = re.sub(r'it', 'ит', txt)
        txt = re.sub(r'hr|рекрутер|recruiter', 'персонала', txt)
        txt = re.sub(r'web', 'веб', txt)
        txt = re.sub(r'сопровожд[а-я]*', 'сопровождение', txt)
        txt = re.sub(r'аналитик[а-я]*', 'аналитик', txt)
        txt = re.sub(r'ассистент', 'помощник', txt)
        txt = re.sub(r'\s+|-', ' ', txt).strip()
        n_key = [word for word in re.split(r' |\\|/', txt) if word != '']
        key_spec = [
            'c++', 'c#', 'php', 'seo', '1c', 'java', 'react', 'qa', ['sql', 'бд'], ['sql', 'database'],
            'ios', ['ux', 'ui'], 'ruby', 'learning', 'backend', 'fullstack', 'frontend',
            'linux', 'дизайнер', ['abap', 'sap'], 'монтажник', 'smm', 'маркетолог', 'админиcтратор', 'веб',
            'сопровождение', 'аналитик', 'javascript', 'scientist', 'оператор', 'devops', ['веб', 'html'], 'b2b',
            'поддержки', 'продаж', 'консультант', 'безопасности', 'педагог', 'автоматизации', ['директор','director']
            , 'начальник', 'data',
            'тестировщик', 'ремонту', 'автор', 'асу', 'связи', 'персонала', 'помощник', 'главный', 'программист',
            'менеджер', 'специалист', 'инженер', 'owner', 'редактор', 'таргетолог'
        ]
        n_key = frozenset([word for word in n_key if word not in stop_words])
        for spec in key_spec:
            is_inst = isinstance(spec, list)
            key_save = spec[0] if is_inst else spec
            key_search = spec[1] if is_inst else key_save
            if key_save in n_key or key_search in n_key:
                n_key = frozenset([key_save])
                break
        res[n_key].append(item)
    groups = defaultdict(list)
    for key, value in res.items():
        if len(value) == 1:
            groups[frozenset(['etc'])] += value
        else:
            groups[key] = value
    s = [(k, groups[k]) for k in sorted(groups, key=lambda x: len(groups[x]), reverse=True)]
    print(f'New len:{len(s)}')
    for key, value in s:
        print(f"{' '.join(key)}: {len(value)}")


if __name__ == '__main__':
    df = pd.read_csv("./data/vacancies.csv")
    names = set(df.name.apply(lambda x: x.lower().strip()))
    print(f'Origin len:{len(names)}')
    simple_process(names)
