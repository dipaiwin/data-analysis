import json
import re
import urllib.parse

import requests
import pandas as pd
from tqdm import tqdm


def remove_tags_and_whitespaces(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def parse_text(text, title):
    pattern = r'<strong>' + title + r'.*?</strong>:?(.*?)'
    texts = re.findall(pattern + '(?:<strong>|$)', text)
    if len(texts) == 0:
        return '', text
    cln_text = texts[0]
    cln_text = remove_tags_and_whitespaces(cln_text)
    text = re.sub(pattern, '', text)
    return cln_text, text


def handle_salary(record):
    exchange_rates = {'USD': 60, 'EUR': 70}
    salary = record['salary']
    if salary is None:
        salary = dict()
    currency = salary.get('currency')
    if currency in exchange_rates:
        for key in ('to', 'from'):
            if salary[key] is None:
                continue
            salary[key] *= exchange_rates[salary['currency']]
    return salary


def extract_experience(record):
    experiences = re.findall('[\d+]', record['experience']['id'])
    min_experience, max_experience = 0, 0
    if len(experiences) > 0:
        min_experience = max_experience = experiences[0]
        if len(experiences) == 2:
            max_experience = experiences[1]
    return min_experience, max_experience


def convert_to_my_format(record):
    description = record['description'].lower()
    mapping = {
        "обязанности": 'responsibility',
        "условия": 'conditions',
        "требования": 'requirement'
    }
    result = {}
    for key, value in mapping.items():
        result[value], description = parse_text(description, key)
    result['description'] = remove_tags_and_whitespaces(description)
    salary = handle_salary(record)
    extract_experience(record)
    min_experience, max_experience = extract_experience(record)
    result.update(
        {
            'name': record['name'],  # название
            'city': record['area']['name'],  # город
            'salary_from': salary.get('from'),  # минимальная зарплата
            'salary_to': salary.get('to'),  # максимальная зарплата
            'employer': record['employer']['name'],  # название компании
            'published_at': record['published_at'],  # дата размещения вакансии
            'experience_from': min_experience,  # минимальный требуемый опыт работы
            'experience_to': max_experience,  # максимальный требуемый опыт работы
            'employment': record['employment']['id'],  # тип занятости
            'schedule': record['schedule']['id'],  # рабочий график
            'key_skills': '|'.join([item['name'] for item in record['key_skills']]),  # ключевые навыки
        }
    )
    return result


def parse_vacancies():
    result = []
    base_url = 'https://api.hh.ru/vacancies/'
    prev_len_result = 0
    for area_id in (2, 95):
        parameters = {"area": area_id, 'per_page': 100, 'page': 1, 'specialization': 1}
        for i in tqdm(range(1, 12)):
            parameters['page'] = i
            r = requests.get(base_url, params=parameters)
            candidates = json.loads(r.text)['items']
            for candidate in candidates:
                a = json.loads(requests.get(urllib.parse.urljoin(base_url, candidate['id'])).text)
                result.append(convert_to_my_format(a))
            l_res = len(result)
            if prev_len_result == l_res:
                break
            prev_len_result = l_res
    return result


if __name__ == '__main__':
    res = parse_vacancies()
    pd.DataFrame(res).to_csv('./test.csv')
