"""Strict JSON table audits and reproducible synthetic fixtures."""
from __future__ import annotations
import json
import math
from pathlib import Path
from .decode import DecodingConfig, beam_trace
from .filtering import sampling_trace
from .paths import enumerate_paths

DEFAULT = {'start': [60,39,1], 'a': [34,33,33], 'b': [1,1,98]}
NONMONOTONE = {'start': [50,50,9], 'a': [45,60,77], 'b': [54,47,5]}

class TableModel:
    """Original synthetic two-symbol Markov table. Integers are weights, not data."""
    def __init__(self, counts=None):
        counts = DEFAULT if counts is None else counts
        if not isinstance(counts,dict) or set(counts) != {'start','a','b'}:
            raise ValueError('counts require exactly start, a and b rows')
        self.counts = {}
        for state, row in counts.items():
            if not isinstance(row,list) or len(row)!=3 or any(type(n) is not int or not 0<=n<=1000000 for n in row) or sum(row)==0:
                raise ValueError('each row requires three nonnegative integer weights <= 1000000 with positive total')
            self.counts[state]=list(row)
        self.rows={s:dict(zip(['a','b','</s>'],[n/sum(row) for n in row])) for s,row in self.counts.items()}
    def distribution(self, context):
        if any(x not in ('a','b') for x in context):
            raise ValueError('table prompts use a and b only')
        return dict(self.rows[context[-1] if context else 'start'])
    def unigram_count(self,token):
        return 0


def build_audit(counts=None, config=None, prompt=None, widths=(1,2,3)):
    model=TableModel(counts);config=config or DecodingConfig(max_tokens=3)
    prompt=[] if prompt is None else prompt
    if not isinstance(widths,(list,tuple)) or not widths or len(widths)>8 or any(type(w) is not int or not 1<=w<=32 for w in widths):
        raise ValueError('widths require 1-8 integers in [1,32]')
    policy=enumerate_paths(model,prompt,config)
    oracle=enumerate_paths(model,prompt,DecodingConfig(max_tokens=config.max_tokens))
    best=oracle['outcomes'][0]
    beams=[]
    for width in widths:
        trace=beam_trace(model,prompt,width,config)
        winner=trace['beams'][0]
        beams.append({'width':width,'trace':trace,
                      'oracle_log_gap':best['model_log_probability']-winner['log_probability'],
                      'attains_oracle_score':abs(best['model_log_probability']-winner['log_probability'])<1e-12})
    return {'schema':'decodelab-audit-v1','counts':model.counts,'prompt':prompt,
            'config':{'max_tokens':config.max_tokens,'temperature':config.temperature,'top_k':config.top_k,'top_p':config.top_p},
            'stages':{s:sampling_trace(row,config) for s,row in model.rows.items()},
            'sampling_tree':policy,'unfiltered_oracle':oracle,'beams':beams}


def read_spec(path):
    path=Path(path)
    if path.stat().st_size>131072:raise ValueError('spec exceeds 128 KiB')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    def bad(value):raise ValueError('nonfinite JSON constant')
    value=json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=bad)
    if not isinstance(value,dict) or set(value)-{'counts','config','prompt','widths'}:raise ValueError('unknown spec fields')
    return build_audit(value.get('counts'),DecodingConfig(**value.get('config',{'max_tokens':3})),value.get('prompt'),value.get('widths',[1,2,3]))
