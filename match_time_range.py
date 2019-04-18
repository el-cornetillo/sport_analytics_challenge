import numpy as np


map_ranges = {0: 'TIMING__0-15',
              1: 'TIMING__15-30',
              2: 'TIMING__22-37',
              3: 'TIMING__30-45',
              4: 'TIMING__45-60',
              5: 'TIMING__52-67',
              6: 'TIMING__60-75',
              7: 'TIMING__67-82',
              8: 'TIMING__7-22',
              9: 'TIMING__75-90',
              10: 'TIMING__prolong_1',
              11: 'TIMING__prolong_2'}

rv_map_ranges = {v: k for k, v in map_ranges.items()}


def overlap(tmin, tmax, refs = [0, 7, 15, 22, 30, 45, 52, 60, 67, 75]):
    tmin, tmax = tmin//60, tmax//60
    if tmax >90:
        return 'TIMING__prolong_2'
    if 45 < tmax < 60:
        return 'TIMING__prolong_1'
    
    t = refs[np.argmin([abs(tmin - ref) for ref in refs])]
    return "TIMING__%d-%d" % (t, t+15)