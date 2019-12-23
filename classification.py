import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle


def add_label(data_frame, need_write=False, le=None):
    group_name = data_frame.group_name
    if le is None:
        le = preprocessing.LabelEncoder()
        le.fit(group_name)
    labels = le.transform(group_name)
    if need_write:
        pd.DataFrame({'group_name': group_name, 'class_label': labels}) \
            .drop_duplicates() \
            .sort_values(['class_label']) \
            .to_csv('train_data/label_class.csv', index=False)
    return labels, le


def drop_columns(data_frame):
    try:
        return data_frame.drop(
            ['responsibility', 'conditions', 'requirement', 'description', 'name', 'city', 'salary_from', 'salary_to',
             'employer', 'published_at', 'experience', 'employment', 'schedule', 'key_skills', 'group_name',
             'count_days',
             'Unnamed: 0'
             ], axis=1)
    except:
        return data_frame.drop(['group_name'], axis=1)


def train_models(model, data, le):
    model.fit(data['xTrain'], data['yTrain'])
    print(model)
    expected = data['yTest']
    predicted = model.predict(data['xTest'])
    res_d = metrics.classification_report(expected, predicted, output_dict=True)
    pd.DataFrame({'Origin label': le.inverse_transform(expected), 'Predict label': le.inverse_transform(predicted)}) \
        .to_csv('classification_result.csv')
    return int(res_d['accuracy'] * 100)


if __name__ == '__main__':
    result = dict()
    for train_file, test_file in (
            ('train.csv', 'test_full.csv'), ('train2.csv', 'test2.csv'), ('train3.csv', 'test3.csv'),('train4.csv', 'test4.csv')):
        train = shuffle(pd.read_csv(f'./data/{train_file}'))
        class_labels, labenc = add_label(train, True)
        train = drop_columns(train)
        # print('Len train: ', train.shape[0])
        test = shuffle(pd.read_csv(f'./data/{test_file}'))
        bb = set(test.columns)
        test_labels, _ = add_label(test, False, labenc)
        test = drop_columns(test)
        name_data = ('xTrain', 'xTest', 'yTrain', 'yTest')
        dataset = dict()
        for key, value in zip(name_data, (train, test, class_labels, test_labels)):
            dataset[key] = value
        models = {
            KNeighborsClassifier,
            DecisionTreeClassifier,
            LogisticRegression
        }
        best_model = None
        best_avg_percent = 0
        for model in models:
            cross = cross_val_score(model(), train, class_labels, cv=3)
            avg_percent = sum(cross) / len(cross)
            if avg_percent > best_avg_percent:
                best_avg_percent = avg_percent
                best_model = model
        result[test_file] = train_models(best_model(), dataset, labenc)
    print(result)
