# DecodeLab

DecodeLab shows how temperature, top-k, top-p and beam search change a small language-model continuation. It retains every sampling stage, finite outcome and beam-pruning decision so that probability, model score and stopping rules can be checked separately. Dependency-free Python and independent scalar JavaScript run offline, without a model API.

[Live inspector](https://midya8.github.io/decodelab/) · [Numerical rules and limits](AUDIT.md)

## Sampling: normalize before selecting a nucleus

The sampler applies stable log-probability temperature scaling, normalizes, applies top-k and normalizes again, then selects the smallest top-p prefix reaching the threshold. `sampling_trace` returns each distribution and its retained-mass cutoff. Equal sampling probabilities use lexical token order; other libraries can use different ties and filter rules.

The earlier sampler selected the nucleus from unnormalized powers. With probabilities `(0.6, 0.3, 0.1)`, temperature 0.5 and top-p 0.7, those powers sum to 0.46, so it kept every token. After normalization the first token alone has about 0.7826 mass and is the complete nucleus. The current log-space calculation also avoids all-event direct-power underflow, although tiny tails can still round to zero.

## Paths: distinguish EOS from a generation cap

The finite-tree audit enumerates all filtered continuations within explicit bounds. EOS contributes its probability and stops; a capped path does not invent an EOS event. Filtered sampling probability and unfiltered model log score are reported separately. The prompt is conditioned on, not included in continuation scores.

Beam traces show every candidate, parent and survivor. Finished and live paths share the same beam, scored by unfiltered conditional log probability, with no length penalty or separate finished pool. The exhaustive oracle uses the same EOS/cap objective.

One synthetic control loses a better future at width 1. Another retains the oracle at widths 1 and 3 but loses it at width 2. That width counterexample depends on the stated shared-pool algorithm, not every beam implementation. A higher model score does not establish better writing.

The browser lets you edit transition weights, temperature, top-k, top-p, generation cap and prompt. It shows each probability stage, paginates every outcome, lists beam candidates and exports the complete current audit.

## Original corpus-trained decoder

The add-k n-gram model, tokenizer, generation/compare commands, diversity metrics, reports and gate examples remain. All 36 original tests are unchanged. Short contexts now use the same start padding as training. Corrected normalization and padding intentionally change affected sampling or short-prompt outputs; the original fixed greedy/demo assertions still pass.

```sh
python -m pip install .
python -m decodelab audit --json audit.json
python -m decodelab audit --spec spec.json --json audit.json
python -m decodelab generate --corpus examples/data/corpus.txt --strategy greedy --prompt "the tokenizer"
python -m decodelab generate --corpus examples/data/corpus.txt --strategy sample --prompt "the" --temperature 0.9 --top-p 0.9 --seed 9
python -m decodelab generate --corpus examples/data/corpus.txt --strategy beam --prompt "the" --beam-width 3
python -m decodelab compare --corpus examples/data/corpus.txt --prompt "the tokenizer" --json report.json --markdown report.md --max-repetition 0.5
```

Distinct-n, repetition rate and unigram entropy describe output form, not factuality or coherence. A common seed makes an implementation reproducible; it does not isolate a causal effect or make one sample representative of a distribution. The n-gram model is an offline stand-in, not a modern language model.

## Checks

The suite has 54 Python tests, including the 36 original tests. Independent references use 90-digit Decimal sampling powers, rational path products and rational forward probability audits on 40 random tables. Checked-in evidence contains 32 complete tree/beam audits and 120 staged-distribution cases.

Three Node/actual-Chrome tests check 7,927 numerical comparisons, 96 distribution views, 231 displayed outcome rows and 956 beam-candidate rows, plus export, stale/invalid state and 320/390/1280 containment. These are floating-point comparisons within documented tolerances, not exact real-arithmetic proofs.

The installed-wheel check runs outside the checkout, verifies five module origins, reproduces 32 table audits and exercises the nucleus/width controls and both audit and original generation commands. CI covers Python 3.10-3.13.

```sh
python -m unittest discover -s tests -v
bash examples/demo.sh
python -m tools.build_audit --check
node --test tests/*.cjs
bash examples/verify_wheel.sh
python -m http.server 8000 --directory docs
```

Node 22+ and Chrome are needed for browser checks; the wheel script uses `uv`.

## Limits and prior work

Exhaustive trees are exponential. The Python enumerator has explicit depth, vocabulary and node limits and refuses overflow rather than returning partial evidence. The browser is limited to two symbols, six generation steps and beam widths 1-3. Very small probability products can underflow. Cutoffs close to a boundary can differ across arithmetic implementations.

[AUDIT.md](AUDIT.md) records the exact API, tie/filter rules, strict table specification, EOS/cap objective, counterexamples and prior sources. Temperature, nucleus sampling and beam search are established methods. This project does not claim a new decoder or improved language quality. Implementation and original synthetic audits: Midya Rahmani. MIT license.
