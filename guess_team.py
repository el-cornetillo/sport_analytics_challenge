import os
import json
import pandas as pd


by_cols = ['Injured player id', 'Captain', 'Fantasy Assist Team']

maps = {}
for file in os.listdir('./maps/'):
    if file.endswith('json'):
        with open('./maps/' + file, 'r', encoding='utf8') as fp:
            maps[file.split('.')[0]] = json.load(fp)

teams = [l.strip() for l in open("./cols/teams.txt", 'r', encoding='utf8')]
played = set([l.strip()
              for l in open("./cols/played_games.txt", 'r', encoding='utf8')])


def candidates_team(pos, pos_jersey):
    return set([k for k, v in maps['team2players'].items() if (int(pos_jersey) in v[pos])])


def direct_guess(df):

    def helper(df, by_col):
        dd = df[~pd.isnull(df[by_col])][['team_id', by_col]].drop_duplicates()
        try:
            home = str(int(list(dd[dd.team_id == "1"][by_col])[0]))
        except:
            home = None
        try:
            away = str(int(list(dd[dd.team_id == "0"][by_col])[0]))
        except:
            away = None
        if by_col in ['Injured player id', 'Captain']:
            home, away = maps['player2team'].get(maps['id2player'].get(
                home)), maps['player2team'].get(maps['id2player'].get(away))
        if by_col == "Fantasy Assist Team":
            home, away = maps['id2team'].get(home), maps['id2team'].get(away)
        return home, away

    try:
        home, away = zip(*[helper(df, c) for c in by_cols])
        home = next((elt for elt in home if elt is not None), None)
        away = next((elt for elt in away if elt is not None), None)
        return home, away
    except:
        return None, None


def indirect_guess(df):
    dd = df[df.type_id.isin(['Player Off', "Player on"])][[
        'team_id', 'Player position', 'Jersey number']]
    try:
        conds = dd[dd.team_id == "1"].values[:, 1:]
        home = set.intersection(
            *[candidates_team(elt[0], int(elt[1])) for elt in conds])
    except:
        home = set()
    try:
        conds = dd[dd.team_id == "0"].values[:, 1:]
        away = set.intersection(
            *[candidates_team(elt[0], int(elt[1])) for elt in conds])
    except:
        away = set()

    return home, away


def possible_opponent(home=None, away=None):
    if home is not None:
        return [elt for elt in teams if "%s - %s" % (home, elt) not in played]
    if away is not None:
        return [elt for elt in teams if "%s - %s" % (elt, away) not in played]
    return teams


def whos_the_team(df, prod_mode=False):
    try:
        home, away = direct_guess(df)
        if ((home is not None) and (away is not None)):
            return [home], [away]
        if ((home is not None) and (away is None)):
            opponents = possible_opponent(home=home) if prod_mode else [
                t for t in teams if t != home]
            indirects = indirect_guess(df)[1]
            if indirects:
                return [home], list(indirects.intersection(set(opponents)))
            else:
                return [home], opponents
        if ((home is None) and (away is not None)):
            opponents = possible_opponent(away=away) if prod_mode else [
                t for t in teams if t != away]
            indirects = indirect_guess(df)[0]
            if indirects:
                return list(indirects.intersection(set(opponents))), [away]
            else:
                return opponents, [away]

        if ((home is None) and (away is None)):
            home, away = indirect_guess(df)
            if home and away:
                # if prod_mode:
                #     home = list(home)
                #     away = list(away)
                #     games = ["%s - %s" % (home[i], away[j]) for i in range(len(home))
                #              for j in range(len(away)) if home[i] != away[j]]
                #     games = [elt for elt in c if elt not in played]
                #     home, away = zip(*[elt.split(' - ') for elt in c])
                #     home = list(set(home))
                #     away = list(set(away))

                #     return home, away
                # else:
                return list(home), list(away)

            if home and (not away):
                # if prod_mode:
                #     opponents = list(
                #         set(sum([possible_opponent(home=h) for h in home], [])))
                #     return list(home), opponents
                # else:
                return list(home), teams

            if (not home) and away:
                # if prod_mode:
                #     opponents = list(
                #         set(sum([possible_opponent(away=a) for a in away], [])))
                #     return opponents, list(away)
                # else:
                return teams, list(away)

        return teams, teams
    except:
        return teams, teams