import os
import random
#import xml.etree.ElementTree as ET
import lxml.etree as ET
import pandas as pd
import numpy as np
import json


maps = {}
for file in os.listdir('./maps/'):
    if file.endswith('json'):
        with open('./maps/' + file, 'r', encoding='utf8') as fp:
            maps[file.split('.')[0]] = json.load(fp)


attrs_event = [l.strip() for l in open("./cols/attrs_event.txt", 'r', encoding='utf8')]

all_cols = [l.strip() for l in open('./cols/all_cols.txt',
                                    'r', encoding='utf8').readlines()]


def get_match_root(path):
    tree = ET.parse(path)
    root = tree.getroot()
    return root.getchildren()[0]


def get_qualifiers(event):
    return {"qualifier_id_" + q.get('qualifier_id'): q.get('value', '1')
            for q in event.getchildren() if q.get('qualifier_id')}


def get_match_events(path):
    root = get_match_root(path)

    events = pd.DataFrame({**event.attrib, **get_qualifiers(event)}
                          for event in root.getchildren())

    qualifiers = filter(lambda f: 'qualifier' in f, events.columns)
    qualifiers = sorted(qualifiers, key=lambda k: int(k.split('_')[-1]))

    events['match_id'] = path.split('.')[0]

    return events.reindex(attrs_event + qualifiers, axis=1)


def translate_ids(match, test_mode=False, extend=True):
    match = match.copy()
    if not test_mode:
        match['team_id'] = match.team_id.map(maps['id2team'])
        match['player_id'] = match.player_id.map(maps['id2player'])
    match['outcome'] = match.apply(
        lambda x: maps['id2outcomes'][x.type_id].get(x.outcome, ""), axis=1)
    match['type_id'] = match.type_id.map(maps['id2event'])
    match['period_id'] = match.period_id.map(maps['id2period'])
    match = match.rename({'qualifier_id_' + k: v for k,
                          v in maps['id2qualifiers'].items()}, axis=1)

    if extend:
        return match.reindex(all_cols, axis=1)
    return match