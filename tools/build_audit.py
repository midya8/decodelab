"""Regenerate checked fixtures; run from repository root with python -m tools.build_audit."""
import argparse
import json
import random
from pathlib import Path
from decodelab.audit import build_audit, DEFAULT, NONMONOTONE
from decodelab.decode import DecodingConfig
from decodelab.filtering import sampling_trace

def build():
    cases=[]
    for counts in [DEFAULT,NONMONOTONE]:
        for horizon in [1,2,3,4]:
            for t,k,p in [(1,None,None),(.5,None,.7),(1,2,.65),(2,2,.9)]:
                cases.append(build_audit(counts,DecodingConfig(max_tokens=horizon,temperature=t,top_k=k,top_p=p)))
    rng=random.Random(708);stages=[]
    for i in range(120):
        w=[rng.randint(1,1000) for _ in range(rng.randint(2,9))]
        dist={str(j):v/sum(w) for j,v in enumerate(w)}
        cfg=DecodingConfig(temperature=[.1,.5,.7,1,2,10][i%6],top_k=[None,1,3,20][i%4],top_p=[None,.4,.7,.95,1][i%5])
        stages.append({'distribution':dist,'config':{'temperature':cfg.temperature,'top_k':cfg.top_k,'top_p':cfg.top_p},'trace':sampling_trace(dist,cfg)})
    return {'schema':'decodelab-browser-fixtures-v1','cases':cases,'stages':stages}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    text=json.dumps(build(),sort_keys=True,indent=2,allow_nan=False)+'\n';path=Path('docs/audit.json')
    if args.check:
        assert json.loads(path.read_text())==json.loads(text),'fixture differs'
    else:path.write_text(text)
    print(json.dumps({'check':args.check,'cases':32,'stage_cases':120,'bytes':len(text)}))
