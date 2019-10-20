import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


class TextProcessManager:
    def __init__(self):
        self.stop_words = set(stopwords.words('russian'))
        self.patterns = [
            (r'\(|\)|\\|,', ' '),
            ('developer|программист|engineer|разработчик', 'программист'),
            ('design(er)?|3d|2d', 'дизайнер'),
            ('front[\s\-]+end|фронт(енд)?|react\.?(js)?|java[\s\-]*script|angular', 'frontend'),
            ('back[\s\-]+end|серверный', 'backend'),
            ('full[\s\-]+stack', 'fullstack'),
            ('1:*[сc]', ' 1c '),
            ('битрикс[^ ]*|bitrix24', '1c'),
            ('c[\s/]+c\+\+|c\+\+[\s/]+c| c |[сc]\+\+', ' c++ '),
            ('[сc]#', ' c# '),
            ('ведущий|старший|senior|team(\s+)lead|lead(er)?', 'lead'),
            ('project|проекта', 'проект'),
            ('продаж[^ ]+|sales', 'продаж'),
            ('техподдержки|support', 'поддержки'),
            ('manager|руководитель', 'менеджер'),
            ('product|продукт[а-я]*|продакт', 'продукт'),
            ('администратор[а-я]*|administrator', 'администратор'),
            ('тест[а-я]*|test[a-z]*', ' тестировщик '),
            ('асутп', 'асу'),
            ('преподаватель|учитель', 'педагог'),
            ('analyst|analytics|аналитик[а-я]*', 'аналитик'),
            ('писатель|копирайтер|writer|copywriter', 'автор'),
            ('it', 'ит'),
            ('hr|рекрутер|recruiter', 'персонала'),
            ('web|html', 'веб'),
            ('сопровожд[а-я]*', 'сопровождение'),
            ('ассистент', 'помощник'),
            ('\s+|-', ' '),
            ('database|бд|баз данных|postgresql', 'sql'),
            ('sap', 'abap'),
            ('ui', 'ux'),
            ('director', 'директор'),
            ('scrum master', 'scrum')
        ]

    def extract_text(self, text):
        cln_text = text
        for pattern in self.patterns:
            desired = pattern[0]
            new_value = pattern[1]
            cln_text = re.sub(desired, new_value, cln_text)
        return cln_text

    def check_word(self, word):
        return word != '' and word not in self.stop_words

    def proceed_text(self, text):
        purified_text = self.extract_text(text)
        separate_list = re.split(r' |\\|/', purified_text.strip())
        return frozenset([word for word in separate_list if self.check_word(word)])
