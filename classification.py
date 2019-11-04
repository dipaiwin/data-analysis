import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


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
    return data_frame.drop(
        ['responsibility', 'conditions', 'requirement', 'description', 'name', 'city', 'salary_from', 'salary_to',
         'employer', 'published_at', 'experience', 'employment', 'schedule', 'key_skills', 'group_name',
         'city: Екатеринбург', 'city: Краснодар', 'city: Новосибирск', 'city: Санкт-Петербург', 'city: Тюмень',
         'city: Челябинск'
         ], axis=1)


def train_models(model, data, le):
    model.fit(data['xTrain'], data['yTrain'])
    print(model)
    expected = data['yTest']
    predicted = model.predict(data['xTest'])
    res_d = metrics.classification_report(expected, predicted, output_dict=True)
    pd.DataFrame({'Origin label': le.inverse_transform(expected),'Predict label':le.inverse_transform(predicted)})\
        .to_csv('classification_result.csv')
    print(res_d['accuracy'])


if __name__ == '__main__':
    train = shuffle(pd.read_csv('./data/train.csv').drop(['Unnamed: 0'], axis=1))
    class_labels, labenc = add_label(train, True)
    train = drop_columns(train)

    test = shuffle(pd.read_csv('./data/test.csv').drop(['Unnamed: 0'], axis=1))
    test_labels, _ = add_label(test, False, labenc)
    test = drop_columns(test)
    name_data = ('xTrain', 'xTest', 'yTrain', 'yTest')
    dataset = dict()
    # train_test_split(df, class_labels, test_size=0.2, random_state=0)
    for key, value in zip(name_data, (train, test, class_labels, test_labels)):
        dataset[key] = value
    # train_models(KNeighborsClassifier(n_neighbors=7), dataset)
    # train_models(DecisionTreeClassifier(max_depth=40), dataset)
    train_models(LogisticRegression(), dataset,labenc)
    # print('Len train data: ', len(yTrain), 'Len test data: ', len(yTest))
    # model = KNeighborsClassifier(n_neighbors=7)
    # model.fit(xTrain, yTrain)
    # print(model)
    # expected = yTest
    # predicted = model.predict(xTest)

    # res_d = metrics.classification_report(expected, predicted, output_dict=True)
    # cnt_good_classes = 0
    # for key, value in res_d.items():
    #     try:
    #         label = int(key)
    #         if value['f1-score'] < 0.6:
    #             print("Label: ", label, labenc.inverse_transform([label])[0], value)
    #         else:
    #             cnt_good_classes += 1
    #     except:
    #         print(key, value)
    # print('Good classes: ', cnt_good_classes)
    # print('All classes: ', len(set(class_labels)))
    # print(pd.crosstab(labenc.inverse_transform(expected), labenc.inverse_transform(predicted), rownames=['True'],
    #                   colnames=['Predicted'], margins=True).to_string())
