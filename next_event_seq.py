import numpy as np
import pickle



with open('./tokenizers/tokenizer_next_event.pickle', 'rb') as fp:
    word_index_t = pickle.load(fp).word_index

rv_word_index_t = {v: k for k, v in word_index_t.items()}



def prepare_seqs(df):
	df['time'] = df['min'].astype(int)*60 + df['sec'].astype(int)
	df.loc[df.team_id=="0", "x"] = 100 - df.loc[df.team_id=="0", "x"].astype(float)
	df.loc[df.team_id=="0", "y"] = 100 - df.loc[df.team_id=="0", "y"].astype(float)
	df['switch'] = (df.team_id.ne(df.team_id.shift().bfill())).astype(int)
	df['delta'] = df.time.diff().fillna(0).astype(int)
	df['type_id'] = df.type_id.apply(lambda x : x.replace(' ', '_').lower())

	def foo(w):
		try:
			return word_index_t.get(w, 0)
		except:
			return 0 

	events = [foo(w) for w in df.type_id]
	events = np.array(events, dtype='int32')[np.newaxis, :]

	pos = [[float(xx), float(yy)] for xx, yy in zip(df['x'], df['y'])]
	pos = np.array(pos)[np.newaxis, :]

	refs = df.team_id.tolist()
	refs = [int(elt == refs[-1]) for elt in refs]
	refs = np.array(refs)[np.newaxis, :]

	deltas = df.delta.astype(int).values[np.newaxis, :]

	return events, pos, refs, deltas
	          