from collections import defaultdict


class GroupCombiner:
    def __init__(self):
        self.main_groups = [
            'c++', 'c#', 'php', 'seo', '1c', 'java', 'react', 'qa', 'sql', 'python', 'android', 'data',
            'ios', 'ux', 'ruby', 'learning', 'backend', 'fullstack', 'frontend', 'веб', 'delphi', 'owner',
            'devops', 'дизайнер', 'lead', 'middle', 'abap', 'монтажник', 'smm', 'маркетолог', 'администратор', 'scrum',
            'сопровождение', 'аналитик', 'javascript', 'оператор', 'b2b', 'поддержки', 'продаж', 'консультант',
            'безопасности', 'педагог', 'автоматизации', 'директор', 'начальник', 'тестировщик', 'ремонту', 'автор',
            'асу', 'связи', 'персонала', 'помощник', 'главный', 'программист', 'менеджер', 'специалист', 'инженер',
            'owner', 'редактор', 'таргетолог',
        ]

    def search_key_groups(self, record):
        for spec in self.main_groups:
            if spec in record:
                return spec
        return record

    @staticmethod
    def create_etc(groups):
        result = defaultdict(list)
        for key, value in groups.items():
            if len(value) == 1:
                result['etc'] += value
            else:
                result[key] = value
        return result
