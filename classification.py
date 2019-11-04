import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def add_label(data_frame, need_write=False):
    le = preprocessing.LabelEncoder()
    group_name = data_frame.group_name
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
         'employer', 'published_at', 'experience', 'employment', 'schedule', 'key_skills', 'group_name'], axis=1)


if __name__ == '__main__':
    df = pd.read_csv('./data/input_data_skills35.csv').drop(['Unnamed: 0'], axis=1)
    class_labels, labenc = add_label(df, True)
    df = drop_columns(df)
    xTrain, xTest, yTrain, yTest = train_test_split(df, class_labels, test_size=0.3, random_state=0)
    print('Len train data: ', len(yTrain), 'Len test data: ', len(yTest))
    model = KNeighborsClassifier(n_neighbors=7)

    # model.fit(df, class_labels)
    # expected = class_labels
    # predicted = model.predict(df)

    model.fit(xTrain, yTrain)
    print(model)
    expected = yTest
    predicted = model.predict(xTest)

    res_d = metrics.classification_report(expected, predicted, output_dict=True)
    cnt_good_classes = 0
    for key, value in res_d.items():
        try:
            label = int(key)
            if value['f1-score'] < 0.6:
                print("Label: ", label, labenc.inverse_transform([label])[0], value)
            else:
                cnt_good_classes += 1
        except:
            print(key, value)
    print('Good classes: ', cnt_good_classes)
    print('All classes: ', len(set(class_labels)))
    print(pd.crosstab(labenc.inverse_transform(expected), labenc.inverse_transform(predicted), rownames=['True'],
                      colnames=['Predicted'], margins=True).to_string())
    # print(metrics.confusion_matrix(expected, predicted))
    # print(labenc.inverse_transform(key),value)
    # print(metrics.classification_report(expected, predicted))
