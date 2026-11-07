#!/usr/bin/env bash
# Fully offline DecodeLab demo: generate with every strategy, compare their
# generation-quality metrics, and exercise both sides of the quality gates.
set -u
cd "$(dirname "$0")/.."

OUT="$(mktemp -d /tmp/decodelab-demo-XXXX)"
trap 'echo "demo artifacts: $OUT"' EXIT

echo "== greedy =="
G=$(python3 -m decodelab generate --corpus examples/data/corpus.txt \
  --strategy greedy --prompt "the tokenizer" --max-tokens 12)
echo "$G"
[ "$G" = "the tokenizer splits text into tokens ." ] || { echo "FAIL greedy output changed"; exit 1; }

echo "== sampling (seeded) =="
S1=$(python3 -m decodelab generate --corpus examples/data/corpus.txt \
  --strategy sample --prompt "the" --max-tokens 10 --seed 9 --temperature 0.9 --top-p 0.9)
S2=$(python3 -m decodelab generate --corpus examples/data/corpus.txt \
  --strategy sample --prompt "the" --max-tokens 10 --seed 9 --temperature 0.9 --top-p 0.9)
echo "$S1"
[ "$S1" = "$S2" ] || { echo "FAIL seeded sampling not reproducible"; exit 1; }

echo "== beam =="
python3 -m decodelab generate --corpus examples/data/corpus.txt \
  --strategy beam --prompt "the tokenizer" --beam-width 3 --max-tokens 12

echo "== compare with reports (expect pass) =="
python3 -m decodelab compare --corpus examples/data/corpus.txt \
  --prompt "the tokenizer" --max-tokens 15 \
  --json "$OUT/report.json" --markdown "$OUT/report.md" \
  --max-repetition 0.5 || { echo "FAIL compare gate that should pass"; exit 1; }

python3 - "$OUT/report.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert len(report["strategies"]) == 5
for name, metrics in report["strategies"].items():
    assert metrics["distinct_2"] == 1.0, (name, metrics)
print("assertions ok: 5 strategies compared, no repeated bigrams anywhere")
PY

echo "== gate (expect fail) =="
if python3 -m decodelab compare --corpus examples/data/corpus.txt \
  --prompt "the tokenizer" --max-tokens 15 --min-distinct-2 1.5 >/dev/null 2>&1; then
  echo "FAIL gate that should fail"; exit 1
fi
echo "gate failed as expected"

echo "demo ok"
