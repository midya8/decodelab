#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cd "$root"
uv build --wheel --out-dir "$tmp/wheel"
uv venv "$tmp/env"
uv pip install --python "$tmp/env/bin/python" "$tmp"/wheel/*.whl
cd "$tmp"
"$tmp/env/bin/python" - <<'PY'
import importlib,json,pathlib
from decodelab.audit import build_audit,NONMONOTONE
from decodelab.decode import DecodingConfig
from decodelab.filtering import sampling_trace
for name in ['decodelab.decode','decodelab.model','decodelab.filtering','decodelab.paths','decodelab.audit']:
    p=pathlib.Path(importlib.import_module(name).__file__)
    assert 'site-packages' in p.parts,p
for table in [None,NONMONOTONE]:
    for depth in [1,2,3,4]:
        for temperature,k,p in [(1,None,None),(.5,None,.7),(1,2,.65),(2,2,.9)]:
            a=build_audit(table,DecodingConfig(max_tokens=depth,temperature=temperature,top_k=k,top_p=p))
            assert abs(a['sampling_tree']['total_probability']-1)<1e-12
assert sampling_trace({'a':.6,'b':.3,'c':.1},DecodingConfig(temperature=.5,top_p=.7))['output']=={'a':1}
a=build_audit(NONMONOTONE)
assert [b['attains_oracle_score'] for b in a['beams']]==[True,False,True]
print(json.dumps({'installed_modules':5,'audit_cases':32,'nonmonotone_widths':[True,False,True]}))
PY
"$tmp/env/bin/python" -m decodelab audit --json "$tmp/audit.json" > "$tmp/stdout.json"
cmp "$tmp/audit.json" "$tmp/stdout.json"
printf 'a b. a a. b a.\n' > "$tmp/corpus.txt"
"$tmp/env/bin/python" -m decodelab generate --corpus "$tmp/corpus.txt" --strategy greedy --prompt a --max-tokens 4
