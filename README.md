# DecodeLab

Inspectable decoding from scratch: stable staged sampling, complete small continuation trees, and beam pruning witnesses. Dependency-free Python, independent scalar JavaScript, no model API.

Temperature reshapes probability. Top-k and top-p remove mass. Beam search removes futures. DecodeLab makes those operations inspectable rather than calling the highest-probability text the best text.

## Probability audit

The sampler uses stable log-probability temperature scaling, then normalized top-k and normalized nucleus selection. `sampling_trace` exposes every distribution and retained-mass cutoff. This fixes the earlier unnormalized nucleus test and direct-power underflow. Short n-gram contexts now use the same start padding as training.

The finite-tree audit distinguishes EOS termination from generation-cap censoring, filtered sampling probability from original model score, and beam pruning from an exhaustive oracle with the same objective. One original synthetic fixture shows a lost better future. A second shows width2 losing a path preserved by widths1/3 under the stated shared finished/live beam.

```sh
python -m decodelab audit --json audit.json
python -m decodelab audit --spec spec.json --json audit.json
python -m tools.build_audit --check
```

The `docs` site runs independent scalar JavaScript. Edit transition weights, temperature, top-k, top-p, generation cap and prompt. Inspect every probability stage, paginate every finite outcome, follow every beam candidate and export the current audit. Nothing calls a model service.

## Original corpus-trained decoder

The original add-k n-gram model, tokenizer, generation/compare CLI, diversity metrics, reports and gate examples remain. All 36 original tests are unchanged. Sampling results intentionally change where corrected normalization or start padding changes the policy; the original exact-output greedy/demo assertions still pass.

```sh
python -m decodelab generate --corpus corpus.txt --strategy greedy --prompt "the tokenizer"
python -m decodelab generate --corpus corpus.txt --strategy sample --prompt "the" --temperature 0.9 --top-p 0.9 --seed 9
python -m decodelab generate --corpus corpus.txt --strategy beam --prompt "the" --beam-width 3
python -m decodelab compare --corpus corpus.txt --prompt "the tokenizer" --json report.json --markdown report.md --max-repetition 0.5
```

Distinct-n, repetition rate and unigram entropy describe output form, not factuality or coherence. A common seed makes each implementation reproducible; it does not isolate a causal effect or make one sample representative of a distribution. The n-gram model is an offline stand-in, not a modern language model.

## Verification

- 54 Python tests, including the 36 original tests; independent 90-digit Decimal sampling reference, rational path products and rational forward probability audits on 40 random tables.
- 32 retained full tree/beam audits and 120 staged-distribution cases.
- Three Node/Chrome tests: 7,927 independent numerical comparisons; 96 distribution views, 231 displayed outcome rows, 956 beam-candidate rows, export/invalidation checks and 320/390/1280 widths.
- Installed-wheel reproduction outside the checkout and Python3.10-3.13 CI.

```sh
python -m unittest discover -s tests -v
bash examples/demo.sh
python -m tools.build_audit --check
node --test tests/*.cjs
bash examples/verify_wheel.sh
```

Read [AUDIT.md](AUDIT.md) for exact API, filtering/tie rules, EOS/cap objective, bounds, counterexamples and prior sources. Exhaustive trees are exponential and bounded. Floating-point results are checked within tolerances, not exact real arithmetic. No new decoding algorithm or language-quality improvement is claimed.

MIT. Implementation and original synthetic audits by Midya Rahmani; established temperature, nucleus and beam methods are attributed in the contract and demo.
