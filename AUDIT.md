# DecodeLab probability and search contract

Given a small transition table, DecodeLab records how filtering changes its probabilities and how beam search removes possible continuations. This note defines the sampling order, stopping events, beam objective and input bounds. The synthetic tables are original counterexamples, not a corpus evaluation or a new algorithm.

## Sampling pipeline

`sampling_trace(distribution, config)` returns every stage. Input probabilities must be finite, nonnegative, at most one and sum to one within 1e-10. Zero-probability events are never resurrected. Positive entries sort by decreasing original probability, then lexical token order.

1. Normalize the supplied distribution.
2. Compute `exp((log(p) - max(log(p))) / temperature)`, then normalize. Subtraction happens before division, keeping the largest exponential at one even for the smallest positive float temperature. Small tails can underflow to zero; this is not an arbitrary-precision sampler.
3. Keep the first k entries, if requested, then normalize their mass.
4. Keep the smallest prefix reaching top-p in that normalized distribution, then normalize again. Top-p=1 disables truncation. At least one positive event survives.

The witness includes base, temperature, top-k and final probabilities, retained temperature mass at top-k, retained top-k mass at the nucleus, and the prefix mass before its final event. These are explicit project conventions. Libraries differ at ties, thresholds, filter order and minimum-token rules. Floating-point cutoffs extremely close to a boundary can change membership across arithmetic implementations.

The old sampler decided the nucleus from unnormalized powers. For `(0.6,0.3,0.1)`, T=0.5 and p=0.7, it kept every event because the powers sum to 0.46. The normalized leading probability is 0.36/0.46, so the corrected nucleus contains only the first event. With top-k=2 at T=1, the top-k probabilities are `(2/3,1/3)`, so p=0.65 also retains only the first event. Direct power underflow is separately tested.

`DecodingConfig` rejects nonfinite temperatures/thresholds, booleans masquerading as numbers, and noninteger token/k limits. Greedy and beam decoding ignore sampling controls. Greedy ties retain the original corpus-frequency-then-lexical convention; sampling ties are lexical, so top-k=1 is not promised equal to greedy at a corpus-frequency tie.

## Conditional paths and stopping

`enumerate_paths(model, prompt, config, max_nodes=100000)` enumerates the full filtered continuation tree. It supports at most 12 generation decisions, 32 next-token entries, and the specified node budget (at most one million). Node counting includes EOS leaves and expanded/cap nodes. Exceeding a bound raises, rather than returning a partial certificate. Probability products may underflow for exceptionally tiny events; finite enumeration is not exact real arithmetic.

A prompt is conditioned on and excluded from continuation scores. An EOS outcome multiplies its EOS probability but does not append an EOS token to displayed text. A cap outcome has no EOS multiplier. They are distinct events even when displayed tokens are identical. Sampling probability multiplies filtered transitions; `model_log_probability` sums unfiltered transition logs. Total mass, EOS mass, cap mass and expected generated-token count are returned.

The n-gram model now pads short contexts with start symbols, matching how training counts were built and the model's stated contract. Full contexts and original fixed-output demos are unchanged. Empty/short-prompt generation intentionally changes to use the learned start distribution, not an unseen-context uniform distribution.

## Beam witnesses and oracle

`beam_trace(model, prompt, width, config)` records all candidates, parent indices, emitted events and survivors at every step, with a 100,000 candidate-record bound. Scores are unfiltered conditional continuation log probability. A finished path stays in the shared finished/live beam; ties sort by token sequence then EOS state. No length normalization, separate finished pool or length penalty is used.

`beam_search` preserves its original list-of-token-lists API and avoids retaining the full trace. Its per-parent candidate bound is retained: the global width winners cannot need more than width candidates from one parent. Use the trace to distinguish EOS termination from cap censoring.

Both beam fixtures use counts/weights in `[a,b,EOS]` order:

| Row | Pruning fixture | Nonmonotone-width fixture |
| --- | --- | --- |
| start | 60,39,1 | 50,50,9 |
| a | 34,33,33 | 45,60,77 |
| b | 1,1,98 | 54,47,5 |

At horizon 3, fixture one has oracle `b + EOS` with probability 0.3822. Width 1 ends at cap with `a a a` and log-score gap 1.7066336993338855. Widths 2 and 3 attain the oracle. In fixture two the oracle is `a + EOS` with probability 0.1940719830628088. Widths 1 and 3 attain it; width 2 prunes it and ends with capped `b b a`, log-score gap 0.6275452747266899. The width counterexample depends on this stated shared-pool algorithm, not all beam implementations. Higher model probability is not necessarily better text.

## Strict table CLI

```sh
python -m decodelab audit --json audit.json
python -m decodelab audit --spec spec.json --json audit.json
```

A spec has only `counts`, `config`, `prompt` and `widths`. Counts require exactly start/a/b rows, each three nonnegative integer weights at most one million and positive total. Prompts use a/b only. Widths are 1-8 values in 1-32. JSON is limited to 128 KiB and rejects duplicate keys and nonfinite constants. Errors return status 2. No partial audit is written on an invalid spec. A complete existing output is overwritten only on a valid audit; this CLI is not an atomic archival store.

Example spec:

```json
{"counts":{"start":[60,39,1],"a":[34,33,33],"b":[1,1,98]},"config":{"max_tokens":3,"temperature":0.5,"top_k":2,"top_p":0.7},"prompt":[],"widths":[1,2,3]}
```

## Reproduction

```sh
python -m unittest discover -s tests -v
bash examples/demo.sh
python -m tools.build_audit --check
node --test tests/*.cjs
bash examples/verify_wheel.sh
```

The original 36 tests remain byte-for-byte. New tests compare 120 staged distributions with independent unshifted 90-digit Decimal powers, all finite products on a rational Cartesian reference tree, rational forward mass/expected-length propagation on 40 random tables, and complete-width beams versus exhaustive oracles. Decimal uses a different computation from the production log/exp path. Rational references operate on supplied integer/decimal weights, not floating-point products.

`docs/audit.json` retains 32 complete audits (two tables, four horizons, four filter configurations) and 120 staged cases. Separate scalar JavaScript checks 7,927 numerical fields. Actual Chrome tests inspect 96 distribution views, every displayed outcome across 231 rows, 956 beam-candidate rows, one JSON export, stale-on-edit behavior and 320/390/1280 layout bounds. Near-tied displayed path ordering is checked as decreasing probability within numerical tolerance, then records are matched by tokens plus stop state, rather than demanding bitwise cross-language sorting.

An installed wheel is checked outside the checkout: five module origins, 32 table audits, the nucleus and nonmonotone-width controls, the JSON CLI and original generation CLI. CI runs Python 3.10-3.13, Chrome and isolated installation. These are tested numerical examples, not universal correctness proofs.

## Sources

- Holtzman et al., nucleus and the distinction between likelihood and text quality: https://arxiv.org/abs/1904.09751
- Nick Higham, shifted softmax numerical arithmetic: https://nhigham.com/2020/02/03/accurately-computing-the-softmax-function/
- Hugging Face, beam and generation strategies: https://huggingface.co/docs/transformers/generation_strategies
- OpenAI GPT-2 sampling implementation, including normalized cumulative probabilities: https://github.com/openai/gpt-2/blob/master/src/sample.py
- LabML, nucleus cumulative-prefix implementation: https://nn.labml.ai/sampling/nucleus.html
- Sebastian Raschka, temperature/top-k/top-p explanation and composition caveats: https://sebastianraschka.com/faq/docs/temperature-topk-topp-sampling.html
