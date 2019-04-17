import numpy as np
import hashlib
import base64
import re
import pickle


with open('./tokenizers/tokenizer_player.pickle', 'rb') as fp:
    word_index_p = pickle.load(fp).word_index
    
rv_word_index_p = {v: k for k, v in word_index_p.items()}

w = 5
MAX_I = 30
N_HASH_CHARS = 11


def get_hash(s):
    _hash = hashlib.md5(s.encode('utf8')).digest()
    _hash = base64.b64encode(_hash).decode('utf8')
    return re.sub(r'[^a-zA-Z0-9]', '', _hash)[:N_HASH_CHARS]


def index_and_pad(seq, wmap=word_index_p, value=0, maxlen=5, reverse=False):

    def foo(w):
        try:
            return word_index_p.get(w, 0)
        except:
            return 0 

    if wmap is not None:
        seq = [[foo(w) for w in s.split()] for s in seq]
    else:
        seq = [[int(w) for w in s.split()] for s in seq]
    if reverse:
        seq = [[value] * (maxlen - len(s)) + s if len(s) < maxlen
               else s[:maxlen] for s in seq]
    else:
        seq = [s + [value] * (maxlen - len(s)) if len(s)
               < maxlen else s[:maxlen] for s in seq]
    return seq

def append_dim(arr):
    return arr[np.newaxis, :]

def pad_to_max_actions(actions, befores, afters, before_teams, after_teams):

    befores = np.array(index_and_pad(befores, reverse=True), dtype='int32')[:MAX_I]
    afters = np.array(index_and_pad(afters), dtype='int32')[:MAX_I]

    before_teams = index_and_pad(
        before_teams, wmap=None, value=-1, reverse=True)
    before_teams = np.array(before_teams, dtype='int32')[:MAX_I]

    after_teams = index_and_pad(after_teams, wmap=None, value=-1)
    after_teams = np.array(after_teams, dtype='int32')[:MAX_I]

    actions = np.array(index_and_pad(actions, maxlen=1), dtype='int32')[:MAX_I]

    actions = np.pad(actions, (0, MAX_I - len(actions)),
                     "constant", constant_values=0)[:, :1]
    action_teams = np.ones(actions.shape).astype(int)

    befores = np.pad(befores, (0, MAX_I - len(befores)),
                     "constant", constant_values=0)[:, :5]
    before_teams = np.pad(
        before_teams, (0, MAX_I - len(before_teams)), "constant", constant_values=0)[:, :5]

    afters = np.pad(afters, (0, MAX_I - len(afters)),
                    "constant", constant_values=0)[:, :5]
    after_teams = np.pad(after_teams, (0, MAX_I - len(after_teams)),
                         "constant", constant_values=0)[:, :5]

    return map(append_dim, (actions, befores, afters, action_teams, before_teams, after_teams))



def prepare_seqs(player_events, hashes, teams, is_start, is_end, tmin, tmax):
    padded_start = False
    padded_end = False

    if is_start:
        if tmin < 10:
            hashes = ["<START_PERIOD_1>"] + hashes
            teams = ["NO_TEAM"] + teams
            padded_start = True
        if tmin >= 45 * 60:
            hashes = ["<START_PERIOD_2>"] + hashes
            teams = ["NO_TEAM"] + teams
            padded_start = True
    if is_end:
        if tmax < 60 * 60:
            hashes = hashes + ["<END_PERIOD_1>"]
            teams = teams + ["NO_TEAM"]
            padded_end = True
        if tmax >= 90 * 60:
            hashes = hashes + ["<END_PERIOD_2>"]
            teams = teams + ["NO_TEAM"]
            padded_end = True

    hashes = [(hashes[i], " ".join(hashes[max(0, i - w):i]),
               " ".join(hashes[i + 1:i + w + 1])) for i in range(len(hashes))]

    teams = [(" ".join([str(int(teams[i] == t)) if t != "NO_TEAM" else str(-1) for t in teams[max(0, i - w):i]]),
              " ".join([str(int(teams[i] == t)) if t != "NO_TEAM" else str(-1) for t in teams[i + 1:i + w + 1]]))
             for i in range(len(teams))]

    if padded_start:
        hashes, teams = hashes[1:], teams[1:]
    if padded_end:
        hashes, teams = hashes[:-1], teams[:-1]

    interventions_player = [ix for ix in player_events if (
        (ix != 0) and (ix != (len(hashes) - 1)))]

    if not interventions_player:

        return pad_to_max_actions(("",), ("",), ("",), ("",), ("",))

    actions, befores, afters = zip(*[hashes[i] for i in interventions_player])
    before_teams, after_teams = zip(*[teams[i] for i in interventions_player])

    return pad_to_max_actions(actions, befores, afters, before_teams, after_teams)


