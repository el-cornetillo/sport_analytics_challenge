import pandas as pd

trs = ['BacktoCenter', 'BacktoLeft', 'BacktoRight',
       'CentertoBack', 'CentertoLeft', 'CentertoRight',
       'LefttoBack', 'LefttoCenter', 'LefttoRight',
       'RighttoBack', 'RighttoCenter', 'RighttoLeft']

sls = ['Back', 'Center', 'Left', 'Right']


def compute_graph(transitions):
    edges = {}
    nodes = {}

    for t in transitions:
        parts = t.split(' - ')
        parts = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
        for p in parts:
            if "nan" in p:
                continue
            if p[0] == p[1]:
                try:
                    nodes[p[0]] += 1
                except KeyError:
                    nodes[p[0]] = 1
                continue
            origin, destination = p[0], p[1]
            try:
                edges[origin + 'to' + destination] += 1
            except KeyError:
                edges[origin + 'to' + destination] = 1

    return {**edges, **nodes}


def compute_graph_team(df):
    dd = df.copy()
    dd['switch'] = dd.team_id.ne(dd.team_id.shift().bfill()).cumsum()
    dd = dd.groupby(['SPLIT', 'team_id', 'switch']).agg(
        {"Zone": lambda x: " - ".join([str(elt) for elt in x])})
    dd = dd.reset_index().groupby('team_id').agg(
        {"Zone": lambda x: compute_graph(list(x))}).Zone.apply(pd.Series)
    dd = dd.reindex(trs + sls, axis=1).fillna(0).astype(int)
    dd = dd.reset_index()
    return dd