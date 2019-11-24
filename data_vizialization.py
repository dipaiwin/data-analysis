import collections

import plotly.graph_objects as go
import plotly.express as exp
import pandas as pd
from wordcloud import WordCloud


def drop_columns(data_frame):
    return data_frame.drop(
        ['responsibility', 'conditions', 'requirement', 'description', 'name', 'city', 'salary_from', 'salary_to',
         'employer', 'published_at', 'experience', 'employment', 'schedule', 'key_skills', 'group_name', 'count_days',
         ], axis=1)


def extract_all_skills(data_frame):
    skls = data_frame.key_skills.apply(lambda x: [item.lower() for item in x.split("|")])
    c = collections.Counter()
    for row in skls:
        for item in row:
            c[item] += 1
    return c
    # return ['_'.join(item.replace(':', '_').replace('/', '_').split())
    #         for row in skls
    #         for item in row
    #         ]
    # return ['_'.join(item.split()) for row in skls for item in row]


def visualization_correlation_matrix(df):
    fg = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.keys(),
        y=df.keys()))
    fg.show()


def visualization_top_n(df):
    s = df.unstack()
    so = s.sort_values(kind="quicksort", ascending=False).to_frame()
    so.reset_index(inplace=True)
    so.columns = ['skill_1', 'skill_2', 'value']
    so = so.loc[so.skill_1 != so.skill_2][:20:2]
    so = so.sort_values(by=['skill_1'])
    fg = exp.line(so, x='skill_1', y='value', text='skill_2')
    fg.show()
    fg = exp.histogram(so, x='skill_1', y='value', histfunc='max', color='skill_2')
    fg.show()
    fg = exp.scatter_matrix(so, color='skill_2')
    fg.show()


def plotly_wordcloud(text):
    mfs = 100
    wc = WordCloud(max_words=200, max_font_size=mfs, collocations=False)
    wc.generate_from_frequencies(text)
    word_list = []
    freq_list = []
    fontsize_list = []
    position_list = []
    orientation_list = []
    color_list = []
    for (word, freq), fontsize, position, orientation, color in wc.layout_:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)
    x = []
    y = []
    for i in position_list:
        x.append(i[0])
        y.append(i[1])
    new_freq_list = []
    for i in freq_list:
        new_freq_list.append(i*mfs+1)
    trace = go.Scatter(x=x, y=y, textfont=dict(size=new_freq_list, color=color_list),
                       hoverinfo='text', hovertext=['{0} {1}'.format(w, f) for w, f in zip(word_list, freq_list)],
                       mode='text', text=word_list)

    layout = go.Layout({'xaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False},
                        'yaxis': {'showgrid': False, 'showticklabels': False, 'zeroline': False}})

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()


if __name__ == '__main__':
    dat_f = pd.read_csv('./data/train.csv').drop(['Unnamed: 0'], axis=1)
    skills = extract_all_skills(dat_f)
    train = drop_columns(dat_f)
    corr = train.corr().abs()
    visualization_correlation_matrix(corr)
    visualization_top_n(corr)
    plotly_wordcloud(skills)
