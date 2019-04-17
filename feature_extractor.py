import pandas as pd
import numpy as np

def open_file(file):
    return [l.strip() for l in open(file, 'r', encoding='utf8')]


body_parts = open_file("./cols/body_parts.txt")
card_kinds = open_file("./cols/card_kinds.txt")
foul_kinds = open_file("./cols/foul_kinds.txt")
goal_kinds = open_file("./cols/goal_kinds.txt")

pass_kinds = open_file("./cols/pass_kinds.txt")
pitch_zones = open_file("./cols/pitch_zones.txt")
role_kinds = open_file("./cols/role_kinds.txt")
save_kinds = open_file("./cols/save_kinds.txt")

shot_kinds = open_file("./cols/shot_kinds.txt")
shot_positions = open_file("./cols/shot_positions.txt")


def extract_features(df):

    r = pd.DataFrame()

    # ATTAQUE

    # goals

    r['goal'] = df.type_id == "Goal"

    for f in shot_positions + body_parts + goal_kinds:
        r['goal_' + f.replace('-', '_').replace(' ', '_').lower(
        )] = r['goal'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    # tirs

    r['shot'] = df.type_id.isin(['Miss', 'Post', 'Attempt Saved'])
    r['miss'] = df.type_id == "Miss"
    r['post'] = df.type_id == "Post"
    r['attempt_saved'] = df.type_id == "Attempt Saved"

    for f in shot_positions + body_parts + shot_kinds:

        r['shot_' + f.replace('-', '_').replace(' ', '_').lower(
        )] = r['shot'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    # DEFENSE

    # tacles

    r['tackle'] = df.type_id == "Tackle"
    r['tackle_success'] = r['tackle'] & df.outcome.str.contains('Success')
    r['tackle_fail'] = r['tackle'] & df.outcome.str.contains('Unsuccess')

    for f in ['Offensive', 'Defensive']:
        r['tackle_' + f[:3].lower()] = r['tackle'] & ~pd.isnull(df.get(f,
                                                                       pd.Series([np.nan] * len(df))))

    for pitch_zone in pitch_zones:
        r['tackle_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['tackle'] & (df.Zone == pitch_zone)

    # degagements

    r['clearance'] = df.type_id == "Clearance"
    r['clearance_head'] = r['clearance'] & ~pd.isnull(df["Head"])

    # interceptions

    r['interception'] = df.type_id == "Interception"
    r['interception_head'] = (
        df.type_id == "Interception") & ~pd.isnull(df.Head)
    r['blocked_pass'] = df.type_id == "Blocked Pass"

    for pitch_zone in pitch_zones:
        r['interception_zone_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = df.type_id.isin(
            ["Interception", "Blocked Pass"]) & (df.Zone == pitch_zone)

    # CIRCULATION DE LA BALLE

    # duels aériens

    r['aerial'] = df.type_id == "Aerial"
    r['aerial_success'] = r['aerial'] & df.outcome.str.contains('won')
    r['aerial_fail'] = r['aerial'] & df.outcome.str.contains('lost')
    for f in ['Offensive', 'Defensive']:
        r['aerial_' + f[:3].lower()] = r['aerial'] & ~pd.isnull(df.get(f,
                                                                       pd.Series([np.nan] * len(df))))

    for pitch_zone in pitch_zones:
        r['aerial_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['aerial'] & (df.Zone == pitch_zone)

    # takeons

    r['take_on'] = df.type_id == "Take On"
    r['take_on_success'] = r['take_on'] & df.outcome.str.contains('Success')
    r['take_on_fail'] = r['take_on'] & df.outcome.str.contains('Unsuccess')

    for pitch_zone in pitch_zones:
        r['take_on_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['take_on'] & (df.Zone == pitch_zone)

    # pertes de ballons (par dribbles)

    r['challenge'] = df.type_id == "Challenge"
    for f in ['Offensive', 'Defensive']:
        r['challenge_' + f[:3].lower()] = r['challenge'] & ~pd.isnull(df.get(f,
                                                                             pd.Series([np.nan] * len(df))))

    for pitch_zone in pitch_zones:
        r['challenge_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['challenge'] & (df.Zone == pitch_zone)

    # pertes de ballons (par tacles)

    r['dispossessed'] = df.type_id == "Dispossessed"

    for pitch_zone in pitch_zones:
        r['dispossessed_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['dispossessed'] & (df.Zone == pitch_zone)

    # recuperations de ballons

    r["recovery"] = df.type_id == "Ball recovery"

    # DISCIPLINE

    # fautes

    r['fouled'] = (df.type_id == "Foul") & df.outcome.str.contains('fouled')
    r['commit_foul'] = (
        df.type_id == "Foul") & df.outcome.str.contains('committed')

    for f in ['Offensive', 'Defensive']:
        r['foul_' + f[:3].lower()] = (df.type_id ==
                                      "Foul") & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    for f in foul_kinds:
        r['foul_' + f.replace('-', ' ').replace(' ', '_').lower(
        )] = (df.type_id == "Foul") & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    for pitch_zone in pitch_zones:
        r['foul_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = (df.type_id == "Foul") & (df.Zone == pitch_zone)

    # cartons

    r['card'] = df.type_id == 'Card'

    for f in card_kinds:
        r['card_' + f.replace('-', ' ').replace(' ', '_').lower(
        )] = r['card'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    # GARDIEN

    # interception sur centres

    r['claim'] = df.type_id == "Claim"

    # arrets du gardien / defenseurs

    r['save'] = df.type_id == "Save"

    for f in save_kinds:
        r['save_' + f.replace('-', ' ').replace(' ', '_').lower(
        )] = r['save'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    # punches gardien

    r['punch'] = df.type_id == "Punch"

    # centres mal recupérés

    r['cross_not_claimed'] = df.type_id == "Cross not claimed"

    # sorties gardien

    r['smother'] = df.type_id == "Smother"

    r['keeper_sweeper'] = df.type_id == "Keeper Sweeper"

    # reprises gardien

    r['gk_pickup'] = df.type_id == "Keeper pick-up"

    # corners gagnés

    r['won_corner'] = (
        df.type_id == "Corner Awarded") & df.outcome.str.contains('won')

    for f in ['Left', 'Right']:
        r['won_corner_side_' + f.replace('-', ' ').replace(' ', '_').lower(
        )] = r['won_corner'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    for pitch_zone in pitch_zones:
        r['won_corner_zone_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['won_corner'] & (df.Zone == pitch_zone)

    # corners concédés

    r['conceded_corner'] = (
        df.type_id == "Corner Awarded") & df.outcome.str.contains('conceded')

    for f in ['Left', 'Right']:
        r['conceded_corner_side_' + f.replace('-', ' ').replace(' ', '_').lower(
        )] = r['conceded_corner'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    for pitch_zone in pitch_zones:
        r['conceded_corner_zone_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['conceded_corner'] & (df.Zone == pitch_zone)

    # changements

    r['player_off'] = df.type_id == "Player Off"

    for f in role_kinds:
        r["player_off_" +
            f.replace('-', ' ').replace(' ',
                                        '_').lower()] = r['player_off'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    r['player_on'] = df.type_id == "Player on"

    for f in role_kinds:
        r["player_on_" +
            f.replace('-', ' ').replace(' ',
                                        '_').lower()] = r['player_on'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    r['formation_change'] = df.type_id == "Formation change"

    # SORTIES

    # sorties

    r['out_put'] = (df.type_id == "Out") & df.outcome.str.contains('put')

    for pitch_zone in pitch_zones:
        r['out_put_zone_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['out_put'] & (df.Zone == pitch_zone)

    r['out_gained'] = (df.type_id == "Out") & df.outcome.str.contains('gained')

    for pitch_zone in pitch_zones:
        r['out_gained_zone_' +
            pitch_zone.replace('-', ' ').replace(' ', '_').lower()] = r['out_gained'] & (df.Zone == pitch_zone)

    # PASSES

    # passes

    r['pass'] = df.type_id == "Pass"
    r['pass_success'] = r['pass'] & df.outcome.str.contains('Success')
    r['pass_fail'] = r['pass'] & df.outcome.str.contains('Unsuccess')

    for f in pass_kinds:
        r['pass_' +
            f.replace('-', ' ').replace(' ',
                                        '_').lower()] = r['pass'] & ~pd.isnull(df.get(f, pd.Series([np.nan] * len(df))))

    # hors-jeux

    r['offside'] = df.type_id == "Offside Pass"
    r['offside_provoked'] = df.type_id == "Offside provoked"

    # good skills

    r['good_skill'] = df.type_id == "Good skill"

    # grosses erreurs

    r['error'] = df.type_id == "Error"

    # penalties

    r['penalty_faced'] = df.type_id == "Penalty faced"

    # chances ratées

    r['chance_missed'] = df.type_id == "Chance missed"

    # shield ball opp

    r['shield_ball_opp'] = df.type_id == "Shield ball opp"

    # Ballons mal controlées

    r['control_fail'] = (
        df.type_id == "Ball touch") & df.outcome.str.contains('unsuccess')

    r['unintentional_touch'] = (
        df.type_id == "Ball touch") & df.outcome.str.contains('unintentionally')

    r['foul_throw_in_conceded'] = (
        df.type_id == "Foul throw-in") & df.outcome.str.contains('conceded')
    r['foul_throw_in_won'] = (
        df.type_id == "Foul throw-in") & df.outcome.str.contains('won')

    r['player_retired'] = df.type_id == "Player retired"

    r['contentious_referee_decision'] = df.type_id == "Contentious referee decision"

    return r