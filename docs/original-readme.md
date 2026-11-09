# DecodeLab

Decoding strategies for language models, from scratch, with measured generation quality.

DecodeLab implements the standard decoding toolbox - greedy, temperature/top-k/top-p sampling, and beam search - in pure Python over an add-k smoothed n-gram model, then measures what each strategy actually produces: distinct-n diversity, repetition rate, and unigram entropy, with JSON/Markdown reports and CI gates. It is dependency-free and runs fully offline.

## Why

Decoding strategy is usually chosen by vibe. DecodeLab makes the tradeoffs measurable on any corpus:

- **an exact distribution to decode from** - the n-gram model's next-token distribution is computed exactly from counts, never approximated, so strategies are compared against ground truth;
- **reproducible sampling** - every sampler takes an explicit seed; the same seed and corpus produce the same text, so generation differences are attributable to the strategy, not luck;
- **quality as metrics, not impressions** - distinct-1/distinct-2, repetition rate and entropy for each strategy, side by side, in terminal, JSON and Markdown;
- **CI gates** - `--min-distinct-2` and `--max-repetition` fail the pipeline when a decoding change (temperature, top-k, top-p, beam width) pushes generation below policy.

## Usage

Generate with one strategy:

```bash
python3 -m decodelab generate --corpus corpus.txt --strategy greedy --prompt "the tokenizer"
python3 -m decodelab generate --corpus corpus.txt --strategy sample \
  --prompt "the" --temperature 0.9 --top-p 0.9 --seed 9
python3 -m decodelab generate --corpus corpus.txt --strategy beam --prompt "the" --beam-width 3
```

Compare all strategies with reports and gates:

```bash
python3 -m decodelab compare --corpus corpus.txt --prompt "the tokenizer" \
  --json report.json --markdown report.md --max-repetition 0.5
```

## Verification

- 36 unit tests cover tokenization, the n-gram model (smoothing, unseen contexts, distribution normalization), every strategy's determinism and filtering behavior, all metrics, all three report formats, and the CLI end to end including both sides of both gates.
- `bash examples/demo.sh` runs fully offline over a small AI-themed corpus: greedy output asserted exactly, seeded sampling asserted reproducible, a five-strategy comparison with JSON/Markdown reports, and both sides of the quality gate.
- CI runs the same suite across Python 3.10-3.13.

Bounds worth stating: the n-gram model is a stand-in that gives the strategies a real, exact distribution to work against - it demonstrates decoding behavior, not modern language-model quality. Metrics like distinct-n and repetition rate catch degenerate repetition, not factual correctness or coherence.

## License

MIT
