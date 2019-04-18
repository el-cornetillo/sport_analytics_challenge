import pandas as pd


def time_played(time_off, time_on, tmin, tmax):
    if not pd.isnull(time_off):
        return int(time_off - tmin)
    if not pd.isnull(time_on):
        return int(tmax - time_on)
    return 15 * 60

def compute_macro_player(dp, cols):

    dp = dp.copy()
    dp['in_back'] = dp.Zone == "Back"
    dp['in_center'] = dp.Zone == "Center"
    dp['in_right'] = dp.Zone == "Right"
    dp['in_left'] = dp.Zone == "Left"
    dd = dp[['player_id'] +
            cols + ['in_back', 'in_center', 'in_left', 'in_right']].groupby(['player_id']).sum()
    bb = dp[(dp.type_id == "Player Off") | (dp.type_id == "Player Off") | ((dp.type_id == "Card") & (~pd.isnull(
        dp["Red card"]))) | ((dp.type_id == "Card") & (~pd.isnull(dp["Second yellow"])))].groupby('player_id').agg({"time": max})
    bb.rename({'time': 'time_off'}, axis=1, inplace=True)
    dd = dd.combine_first(bb)

    bb = dp[(dp.type_id == "Player on") | (dp.type_id == "Player returns")].groupby('player_id').agg({"time": min})
    bb.rename({'time': 'time_on'}, axis=1, inplace=True)
    dd = dd.combine_first(bb)

    tmin, tmax = dp.time.min(), dp.time.max()

    dd['time_played'] = dd.apply(lambda x: time_played(
        x.time_off, x.time_on, tmin, tmax), axis=1)
    dd = dd.reset_index()
    dd.drop(['time_off', 'time_on'], axis=1, inplace=True)

    return dd