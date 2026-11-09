"""DecodeLab command line: generate text with each strategy, compare them."""

from __future__ import annotations

import argparse
import json
import random
import sys

from .decode import DecodingConfig, beam_search, greedy, sample
from .metrics import evaluate
from .model import NGramModel
from .report import build_report, to_json, to_markdown, to_terminal
from .tokenize import tokenize, tokenize_sentences


def _load_model(corpus_path: str, order: int, k: float) -> NGramModel:
    try:
        with open(corpus_path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise SystemExit(f"error: cannot read corpus: {exc}")
    sentences = tokenize_sentences(text)
    if not sentences:
        raise SystemExit("error: corpus produced no sentences")
    model = NGramModel(order, k=k)
    model.train(sentences)
    return model


def _config(args: argparse.Namespace) -> DecodingConfig:
    return DecodingConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )


def cmd_generate(args: argparse.Namespace) -> int:
    model = _load_model(args.corpus, args.order, args.k)
    prompt = tokenize(args.prompt) if args.prompt else []
    config = _config(args)
    if args.strategy == "greedy":
        out = greedy(model, prompt, config)
    elif args.strategy == "sample":
        out = sample(model, prompt, config, rng=random.Random(args.seed))
    else:
        out = beam_search(model, prompt, args.beam_width, config)[0]
    print(" ".join(out))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    model = _load_model(args.corpus, args.order, args.k)
    prompt = tokenize(args.prompt) if args.prompt else []
    base = DecodingConfig(max_tokens=args.max_tokens)
    strategies = {
        "greedy": greedy(model, prompt, base),
        "sample-t1.0": sample(model, prompt, base, rng=random.Random(args.seed)),
        "sample-t0.7-topk5": sample(
            model, prompt,
            DecodingConfig(max_tokens=args.max_tokens, temperature=0.7, top_k=5),
            rng=random.Random(args.seed),
        ),
        "sample-t1.0-topp0.9": sample(
            model, prompt,
            DecodingConfig(max_tokens=args.max_tokens, top_p=0.9),
            rng=random.Random(args.seed),
        ),
        f"beam-{args.beam_width}": beam_search(model, prompt, args.beam_width, base)[0],
    }
    report = build_report({name: evaluate(tokens) for name, tokens in strategies.items()})
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            fh.write(to_json(report))
    if args.markdown_path:
        with open(args.markdown_path, "w", encoding="utf-8") as fh:
            fh.write(to_markdown(report))
    print(to_terminal(report))

    failures: list[str] = []
    for name, metrics in report["strategies"].items():
        if args.min_distinct_2 is not None and metrics["distinct_2"] < args.min_distinct_2:
            failures.append(f"{name}: distinct_2 {metrics['distinct_2']:.4f} < {args.min_distinct_2}")
        if args.max_repetition is not None and metrics["repetition_rate"] > args.max_repetition:
            failures.append(f"{name}: repetition_rate {metrics['repetition_rate']:.4f} > {args.max_repetition}")
    for failure in failures:
        print(f"GATE FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="decodelab",
        description="Decoding strategies for language models, from scratch, with measured generation quality.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("generate", "generate text with one strategy"),
                            ("compare", "compare all strategies with quality metrics and gates")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--corpus", required=True, help="plain-text corpus file")
        p.add_argument("--order", type=int, default=3, help="n-gram order (default 3)")
        p.add_argument("--k", type=float, default=0.1, help="add-k smoothing (default 0.1)")
        p.add_argument("--prompt", default="", help="space-separated prompt text")
        p.add_argument("--max-tokens", type=int, default=30)
        p.add_argument("--temperature", type=float, default=1.0)
        p.add_argument("--top-k", type=int, default=None)
        p.add_argument("--top-p", type=float, default=None)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--beam-width", type=int, default=3)
        if name == "generate":
            p.add_argument("--strategy", choices=["greedy", "sample", "beam"], required=True)
        else:
            p.add_argument("--json", dest="json_path", default=None)
            p.add_argument("--markdown", dest="markdown_path", default=None)
            p.add_argument("--min-distinct-2", type=float, default=None)
            p.add_argument("--max-repetition", type=float, default=None)
    audit = sub.add_parser("audit", help="enumerate a bounded synthetic continuation tree and beam witnesses")
    audit.add_argument("--spec", help="strict JSON table/config spec; default is original synthetic table")
    audit.add_argument("--json", dest="audit_json", help="write full audit JSON")
    args = parser.parse_args(argv)
    if args.command == "audit":
        from .audit import build_audit, read_spec
        try:
            result = read_spec(args.spec) if args.spec else build_audit()
            body = json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n"
            if args.audit_json:
                with open(args.audit_json, "w", encoding="utf-8") as fh:
                    fh.write(body)
            print(body, end="")
            return 0
        except (ValueError, TypeError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    return cmd_generate(args) if args.command == "generate" else cmd_compare(args)


if __name__ == "__main__":
    sys.exit(main())
