import operator

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression as lr
from sklearn.feature_selection import VarianceThreshold
from classification import drop_columns, add_label
import pandas as pd


def get_data(path):
    res = pd.read_csv(path)
    id_labs, _ = add_label(res)
    labs = res.group_name
    return drop_columns(res), labs, id_labs


def select_sfs(train, test, labels_id, labels):
    sfs1 = SFS(lr(), k_features=50, forward=True, floating=False, verbose=2, scoring='accuracy', cv=0)
    sfs1 = sfs1.fit(train, labels_id, custom_feature_names=train.columns.tolist())
    max_features = []
    max_val = 0
    for _, value in sfs1.subsets_.items():
        if value['avg_score'] > max_val:
            max_features = value['feature_names']
    max_features = list(max_features) + ['group_name']
    train['group_name'] = labels
    train[max_features].to_csv('data/train4.csv', index=False)
    test[max_features].to_csv('data/test4.csv', index=False)


def select_vt(train, test, labels):
    selector = VarianceThreshold(.01)
    selector.fit(train)
    res_df = train[train.columns[selector.get_support(indices=True)]]
    res_df['group_name'] = labels
    res_df.to_csv('./data/train3.csv', index=False)
    columns = res_df.columns.tolist()
    print(len(columns))
    test[columns].to_csv('data/test3.csv', index=False)


if __name__ == '__main__':
    main_df, origin_marks, id_labels = get_data('data/train2.csv')
    test_df = pd.read_csv('data/test2.csv')
    # select_vt(main_df, test_df, origin_marks)
    select_sfs(main_df, test_df, id_labels, origin_marks)
