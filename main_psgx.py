import pandas as pd
import numpy as np
import os, json

#custom modules
import xml2csv           as x2c
import match_time_range  as mtr
import alignment_data    as ad
import feature_extractor as fe
import player_sequential as ps
import team_graphs       as tg
import guess_team        as gt
import macro_team        as mt
import macro_player      as mp
import next_event_seq    as nes
import adjust_position   as ap

from model_player import WhosThatNet
from model_next_event import NextMoveNet

maps = {}
for file in os.listdir('./maps/'):
    if file.endswith('json'):
        with open('./maps/' + file, 'r', encoding='utf8') as fp:
            maps[file.split('.')[0]] = json.load(fp)

rv_map_player = {v: k for k, v in maps['id2player'].items()}



clf_who = WhosThatNet()
clf_next = NextMoveNet()


def Result(path):
    df = x2c.translate_ids(x2c.get_match_events(path), test_mode=True)
    player, team, x, y = pred_from_dataframe(df)
    with open('res_psgx.csv', 'w', encoding='utf8') as fp:
    	fp.write("%s,%s,%s,%s" % (str(player), str(team), str(y), str(x)))
    
  
def pred_from_dataframe(df):
    ''' Function that takes a dataframe as input and compute predictions
    '''
    
    df1, df2 = df[:-10].reset_index(drop=True), df[-10:].reset_index(drop=True)
    is_start = df1.type_id.iloc[0] == "Start"
    is_end = df1.type_id.iloc[-1] == "End"
    

    df1 = ad.align_events(df1)
    tmin, tmax = df1.time.min(), df1.time.max()
    res = fe.extract_features(df1)
    id2cols = dict(enumerate(res.columns))
    hashes = res.apply(np.nonzero, axis=1).apply(sum)
    hashes = hashes.apply(lambda x: " ".join(sorted([id2cols[k] for k in x])))
    hashes = hashes.apply(ps.get_hash).tolist()

    teams = df1.team_id.tolist()
    player_events = df1[df1.player_id == "1"].index.tolist()
    n_interventions = len(player_events)
    
    actions, befores, afters, action_teams, before_teams, after_teams = ps.prepare_seqs(
        player_events, hashes, teams, is_start, is_end, tmin, tmax)
    cols = res.columns.tolist()
    res = pd.concat([df1, res], axis=1)

    macro_features_teams = mt.compute_macro_team(res, cols)
    graph_features = tg.compute_graph_team(df1)

    team_features = macro_features_teams.merge(
        graph_features, on="team_id").astype(int)
    team_features = team_features.reindex(
        list(team_features.columns[1:]) + [team_features.columns[0]], axis=1)

    home_team = team_features[team_features.team_id.astype(str) == "1"].values
    away_team = team_features[team_features.team_id.astype(str) == "0"].values

    try:
        player = mp.compute_macro_player(res[res.player_id == "1"], cols)
        player = list(player.values.squeeze()[1:])
        player = np.array([n_interventions] + player, dtype='int32')[np.newaxis, :]
    except ValueError:
        player = np.zeros((1, 284)).astype(int)

    timing = np.eye(len(mtr.map_ranges))[
        mtr.rv_map_ranges[mtr.overlap(tmin, tmax)]].astype(int)[np.newaxis, :]
    
    probas = clf_who.predict_proba([home_team, away_team, player, actions, action_teams,
             befores, before_teams, afters, after_teams, timing])
    probas = probas.squeeze().argsort()[::-1]
    
    candidates = gt.whos_the_team(df1, prod_mode=False)
    
    team_player = df1[df1.player_id=='1'].team_id.unique()[0]
    if team_player == "1":
        candidates = candidates[0]
    else:
        candidates = candidates[1]
        
    for i in range(len(probas)):
        pred = maps['cmap_player'][str(probas[i])]
        if maps['player2team'].get(pred) in candidates:
            break
    if i == len(probas) - 1:
        pred_player = str(probas[0])
    else:
        pred_player = str(probas[i])

    pred_player = rv_map_player.get(maps['cmap_player'][pred_player])
        
    actions, pos, refs, deltas = nes.prepare_seqs(df2)
    switch, next_pos = clf_next.predict([actions, pos, refs, deltas])

    last_team = str(df2.team_id.tolist()[-1])
    last_event, last_x, last_y = df2.iloc[-1][['type_id', 'x', 'y']]
    lbo_event = df2.iloc[-2].type_id
    last_event = last_event.replace(' ', '_').lower()
    lbo_event = lbo_event.replace(' ', '_').lower()
    if last_team == "0":
        last_x, last_y = 100 - last_x, 100 - last_y

    next_pos_x, next_pos_y = next_pos[0]
    next_pos_x, next_pos_y = ap.adjust_prediction(last_event, lbo_event, last_x, last_y, next_pos_x, next_pos_y)

    if switch:
        if last_team == '1':
            new_team = 0
            new_x, new_y = 100 - next_pos_x, 100 - next_pos_y
        if last_team == "0":
            new_team = 1
            new_x, new_y = next_pos_x, next_pos_y
    else:
        if last_team == '1':
            new_team = 1
            new_x, new_y = next_pos_x, next_pos_y
        if last_team == "0":
            new_team = 0
            new_x, new_y = 100 - next_pos_x, 100 - next_pos_y

    
    return pred_player, new_team, new_x, new_y