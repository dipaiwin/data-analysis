import math


class SalaryHandler:
    @staticmethod
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

    @classmethod
    def get_valid_salary(cls, group, global_salary):
        salary_city = cls.get_average_salary_interval(group)
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

    @staticmethod
    def set_proceed_salary_value(group, city, salary_field, value):
        group.loc[
            (group.city == city) & (group[salary_field] != group[salary_field]),
            salary_field
        ] = value

    @classmethod
    def fill_salary_gaps(cls, groups, global_salary):
        for _, group in groups.items():
            salary_city = cls.get_valid_salary(group, global_salary)
            for city, salary in salary_city.items():
                for sf in salary:
                    cls.set_proceed_salary_value(group, city, f'salary_{sf}', salary[sf])
