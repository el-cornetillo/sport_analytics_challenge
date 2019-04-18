import pandas as pd


def compute_macro_team(dp, cols):
    dd = dp[['team_id'] + cols].groupby('team_id').sum()
    dd = dd.reset_index()
    return dd