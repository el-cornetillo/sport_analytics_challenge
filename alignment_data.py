import pandas as pd
import numpy as np

rmv = ["Deleted event", "Start delay", "End delay",
       "Team set up", "Collection End", "Start",
       "Condition change", "Official change", "End",
       "Injury Time Announcement"]

statics_after_goals = ["Player Off", "Penalty faced", "Error",
                       "Card", "Contentions referee decision",
                       "Formation change", "Keeper Sweeper",
                       "Ball touch", "Goal"]

statics_after_fouls = ["Card", "Player Off", "Formation change",
                       "Contentious referee decision", "Foul",
                       "Error", "Attempt Saved", "Miss", "Penalty faced",
                       "Goal", "Save", "Post", "Clearance", "Claim",
                       "Punch", "Cross not claimed", "Smother",
                       "Keeper Sweeper", "Keeper pick-up", "Corner Awarded",
                       "Interception", "Blocked Pass"]


def first(it, cond):
    try:
        return next((ix for ix, elt in enumerate(it) if cond(elt)))
    except StopIteration:
        return 0


def align_events(df):
    df = df[df.period_id.isin(['First half', 'Second Half'])
            ].reset_index(drop=True)
    df = df[~df.type_id.isin(rmv)].reset_index(drop=True)

    df['time'] = 60 * df['min'].astype(int) + df['sec'].astype(int)

    try:
        df.drop(df.iloc[[i + 1 for ix, i in enumerate(df[df.type_id == 'Corner Awarded'].index) if ix %
                     2 == 0]].query('type_id != "Corner Awarded"').index, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
    except:
        pass

    try:
        times = df[df.type_id == "Corner Awarded"].groupby(["match_id", "time"], as_index=False).agg(
            {"sec": lambda x: len(x) != 2}).query("sec == True")[['match_id', 'time']].apply(
            lambda r: tuple(r) + ('Corner Awarded',), axis=1).tolist()

        swaps = df[df[['match_id', 'time', 'type_id']].apply(
            tuple, axis=1).isin(times)].index

        df.loc[swaps[1::2], ["min", "sec", "time"]
               ] = df.loc[swaps[::2], ["min", "sec", "time"]].values
    except:
        pass

    try:
        times = df[df.type_id == "Out"].groupby(['match_id', 'time'], as_index=False).agg(
        {"outcome": lambda x: "therefore" in list(x)[0]}).query('outcome == True')[['match_id', 'time']].apply(
        lambda r: tuple(r) + ('Out',), axis=1).tolist()
        swaps = df[df[['match_id', 'time', 'type_id']].apply(
            tuple, axis=1).isin(times)].index
        rows = list(df.index)
        for s in swaps[::2]:
            rows[s] = s + 1
            rows[s + 1] = s
        df = df.iloc[rows].reset_index(drop=True)
    except:
        pass

    try:
        times = df[df.type_id == "Foul"].groupby(['match_id', 'time'], as_index=False).agg(
        {"team_id": lambda x: list(x)[0] != df.team_id[list(x.index)[0] - 1]}).query(
        'team_id == True')[['match_id', 'time']].apply(
        lambda r: tuple(r) + ('Foul',), axis=1).tolist()
        swaps = df[df[['match_id', 'time', 'type_id']].apply(
            tuple, axis=1).isin(times)].index
        rows = list(df.index)
        for s in swaps[::2]:
            rows[s] = s + 1
            rows[s + 1] = s
        df = df.iloc[rows].reset_index(drop=True)
    except:
        pass

    try:
        times = df[df.type_id == "Corner Awarded"].groupby(['match_id', 'time'], as_index=False).agg(
        {"outcome": lambda x: "won" in list(x)[0]}).query('outcome == True')[['match_id', 'time']].apply(
        lambda r: tuple(r) + ('Corner Awarded',), axis=1).tolist()
        swaps = df[df[['match_id', 'time', 'type_id']].apply(
            tuple, axis=1).isin(times)].index
        rows = list(df.index)
        for s in swaps[::2]:
            rows[s] = s + 1
            rows[s + 1] = s
        df = df.iloc[rows].reset_index(drop=True)
    except:
        pass

    idxs_rdb = df[df.type_id == "Referee Drop Ball"].index.tolist()[::2]
    df['SPLIT_RDB'] = np.cumsum([int(i in idxs_rdb) for i in range(len(df))])
    df.drop(idxs_rdb + [i + 1 for i in idxs_rdb], axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    idxs_out = set(df[df.type_id == 'Out'].index[1::2])
    df['SPLIT_OUT'] = np.cumsum([int(i in idxs_out) for i in range(len(df))])

    idxs_period = set((df.period_id.ne(df.period_id.shift().bfill()))
                      .astype(int).nonzero()[0])

    idxs_foul = df[df.type_id == "Foul"].index.tolist()[1::2]

    try:
        idxs_foul = df.iloc[idxs_foul].apply(lambda x: (x.name, first(
        df.type_id.iloc[x.name:x.name + 15], lambda x: x not in
        statics_after_fouls)), axis=1).tolist()
    except AttributeError:
        idxs_foul = []

    idxs_foul = [ix + offset for ix, offset in idxs_foul
                 if not any((ix < p < ix + offset for p in idxs_period))]

    idxs_foul = set(idxs_foul)

    df['SPLIT_FOUL'] = np.cumsum([int(i in idxs_foul) for i in range(len(df))])

    try:
        idxs_goal = df[df.type_id == "Goal"].apply(lambda x: (x.name, first(
            df.type_id.iloc[x.name:x.name + 15], lambda x: x not in
            statics_after_goals)), axis=1).tolist()
    except AttributeError:
        idxs_goal = []

    idxs_goal = [ix + offset for ix, offset in idxs_goal
                 if not any((ix < p < ix + offset for p in idxs_period))]

    idxs_goal = set(idxs_goal)

    df['SPLIT_GOAL'] = np.cumsum([int(i in idxs_goal) for i in range(len(df))])

    df['SPLIT'] = df[[f for f in df.columns if f.startswith('SPLIT')]].apply(
        sum, axis=1, raw=True)

    return df