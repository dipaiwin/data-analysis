from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min

if __name__ == '__main__':
    df_origin = pd.read_csv('data/train_clst.csv')
    df = df_origin.drop(
        ['responsibility', 'conditions', 'requirement', 'description', 'name', 'city', 'salary_from', 'salary_to',
         'employer', 'published_at', 'experience', 'employment', 'schedule', 'key_skills', 'group_name', 'Unnamed: 0'
         ], axis=1)

    kmeans = KMeans(n_clusters=9, random_state=0).fit(df)
    labels = list(kmeans.labels_)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, df)
    df_centers = df.loc[closest, :]
    for i, (_, row) in enumerate(df_centers.iterrows()):
        index_cls = [ind for ind, item in enumerate(labels) if item == i]
        print(len(index_cls))
        print(df_origin.loc[index_cls, :])
        input()
