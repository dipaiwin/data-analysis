from sklearn.cluster import KMeans, AffinityPropagation
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin_min
from extract_features import get_top_skills

model_parameters = [{'n_clusters': 9, 'random_state': 0}, {"damping": 0.97}]
origin_columns = ['name', 'city', 'salary_from', 'salary_to', 'employer', 'experience', 'employment', "schedule",
                  'key_skills', 'count_days', 'group_name', ]
drop_columns = ['responsibility', 'conditions', 'requirement', 'description', 'name', 'city', 'salary_from', 'salary_to',
                'employer', 'published_at', 'experience', 'employment', 'schedule', 'key_skills', 'group_name',
                'Unnamed: 0']

if __name__ == '__main__':
    df_origin = pd.read_csv('data/clean_clst.csv')
    df = df_origin.drop(drop_columns, axis=1)
    df_origin = df_origin[origin_columns]
    for model_creater, params, name in zip([KMeans, AffinityPropagation], model_parameters, ['kmeans', 'ap']):
        top_n_classes = []
        model = model_creater(**params).fit(df)
        labels = list(model.labels_)
        closest, _ = pairwise_distances_argmin_min(model.cluster_centers_, df)
        df_centers = df_origin.loc[closest, :]
        output_text = []
        for i, (_, row) in enumerate(df_centers.iterrows()):
            output_text.append(f'Cluster:{i}')
            values = row.to_csv(index=False).strip('\n').split('\n')
            description_center = ','.join([f'{head}={value}' for head, value in zip(origin_columns, values)])
            output_text.append(description_center)
            index_cls = [ind for ind, item in enumerate(labels) if item == i]
            size_cluster = len(index_cls)
            output_text.append(f'Count object={size_cluster}')
            df_obj_in_class = df_origin.loc[index_cls, :]
            df_obj_in_class['list_skills'] = df_obj_in_class.key_skills.apply(lambda x: [item.lower() for item in x.split("|")])
            if len(top_n_classes) < 3:
                top_n_classes.append((i, df_obj_in_class, size_cluster))
            else:
                min_item = min(top_n_classes, key=lambda t: t[2])
                if min_item[2] < size_cluster:
                    top_n_classes.remove(min_item)
                    top_n_classes.append((i, df_obj_in_class, size_cluster))
            line_top_skills = ','.join([f'{key}={val}' for key, val in get_top_skills(df_obj_in_class, 5, True)])
            output_text.append(f'Top skills: {line_top_skills}')
            mean_salary_from = int(df_obj_in_class.salary_from.mean())
            mean_salary_to = int(df_obj_in_class.salary_to.mean())
            output_text.append(f'Average salary: from={mean_salary_from} to={mean_salary_to}')
            for item in ('experience', 'employment', 'schedule'):
                top_val_feature = df_obj_in_class[item].value_counts()[:3]
                top_val_str = []
                for indx in top_val_feature.index:
                    top_val_str.append(f'{indx}={top_val_feature.loc[indx]}')
                output_text.append(f"{item}: {','.join(top_val_str)}")
            output_text.append('============================')
        with open(f'./clst_data/{name}/cluster_description.txt', 'w', encoding='utf8') as f:
            f.write('\n'.join(output_text))
        for indx, data_frame, _ in top_n_classes:
            data_frame['group_name'].value_counts().to_csv(f'./clst_data/{name}/clst_{indx}')
