import io
import json
import os
import random
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

from decodelab import (
    DecodingConfig,
    NGramModel,
    beam_search,
    distinct_n,
    greedy,
    repetition_rate,
    sample,
    token_entropy,
)
from decodelab.cli import main
from decodelab.metrics import evaluate
from decodelab.report import build_report, to_json, to_markdown, to_terminal
from decodelab.tokenize import tokenize, tokenize_sentences

CORPUS = (
    "The model learns from data. The data teaches the model. "
    "The tokenizer splits text into tokens. Tokens feed the model. "
    "Sampling adds variety to text. Greedy decoding picks the best token."
)


def trained_model(order=3, k=0.1):
    model = NGramModel(order, k=k)
    model.train(tokenize_sentences(CORPUS))
    return model


def write_corpus(tmp, text=CORPUS):
    path = os.path.join(tmp, "corpus.txt")
    with open(path, "w") as fh:
        fh.write(text)
    return path


class TestTokenize(unittest.TestCase):
    def test_lowercase_and_punctuation(self):
        self.assertEqual(tokenize("The Model, learns!"), ["the", "model", ",", "learns", "!"])

    def test_sentences_split_on_terminal_punctuation(self):
        sents = tokenize_sentences("One two. Three four! Five")
        self.assertEqual(sents, [["one", "two", "."], ["three", "four", "!"], ["five"]])


class TestModel(unittest.TestCase):
    def test_order_validation(self):
        with self.assertRaises(ValueError):
            NGramModel(0)

    def test_k_validation(self):
        with self.assertRaises(ValueError):
            NGramModel(2, k=0)

    def test_distribution_sums_to_one(self):
        model = trained_model()
        total = sum(model.distribution(["the", "model"]).values())
        self.assertAlmostEqual(total, 1.0, places=9)

    def test_unseen_context_gets_uniform_addk(self):
        model = trained_model()
        dist = model.distribution(["zzz", "qqq"])
        self.assertEqual(len(dist), len(model.vocab) + 1)
        self.assertAlmostEqual(min(dist.values()), max(dist.values()), places=12)

    def test_context_truncated_to_order(self):
        model = trained_model(order=2)
        d1 = model.distribution(["model"])
        d2 = model.distribution(["the", "model"])
        self.assertEqual(d1, d2)

    def test_untrained_model_raises(self):
        with self.assertRaises(ValueError):
            NGramModel(2).distribution(["x"])

    def test_train_replaces_counts(self):
        model = trained_model()
        model.train([["only", "this"]])
        self.assertEqual(model.vocab, frozenset({"only", "this"}))

    def test_unigram_counts(self):
        model = trained_model(order=2)
        self.assertEqual(model.unigram_count("the"), CORPUS.lower().split().count("the"))
        self.assertEqual(model.unigram_count("zzz"), 0)

    def test_log_probability(self):
        model = trained_model(order=2)
        lp = model.log_probability(["the", "model"])
        self.assertLess(lp, 0)
        with self.assertRaises(ValueError):
            model.log_probability([])


class TestDecodingConfig(unittest.TestCase):
    def test_defaults_valid(self):
        DecodingConfig()

    def test_invalid_values(self):
        with self.assertRaises(ValueError):
            DecodingConfig(max_tokens=0)
        with self.assertRaises(ValueError):
            DecodingConfig(temperature=0)
        with self.assertRaises(ValueError):
            DecodingConfig(top_k=0)
        with self.assertRaises(ValueError):
            DecodingConfig(top_p=0)
        with self.assertRaises(ValueError):
            DecodingConfig(top_p=1.5)


class TestDecoding(unittest.TestCase):
    def test_greedy_deterministic(self):
        model = trained_model()
        a = greedy(model, ["the", "tokenizer"], DecodingConfig(max_tokens=12))
        b = greedy(model, ["the", "tokenizer"], DecodingConfig(max_tokens=12))
        self.assertEqual(a, b)
        self.assertEqual(a, ["the", "tokenizer", "splits", "text", "into", "tokens", "."])

    def test_greedy_stops_at_end(self):
        model = trained_model(order=2)
        out = greedy(model, ["the", "model"], DecodingConfig(max_tokens=20))
        self.assertEqual(out, ["the", "model", "."])

    def test_sample_reproducible_by_seed(self):
        model = trained_model()
        a = sample(model, ["the"], DecodingConfig(max_tokens=15), rng=random.Random(42))
        b = sample(model, ["the"], DecodingConfig(max_tokens=15), rng=random.Random(42))
        self.assertEqual(a, b)

    def test_sample_top_k_restricts(self):
        model = trained_model(order=2)
        config = DecodingConfig(max_tokens=5, top_k=1)
        out = sample(model, ["the"], config, rng=random.Random(1))
        self.assertEqual(out, greedy(model, ["the"], DecodingConfig(max_tokens=5)))

    def test_temperature_sharpens(self):
        model = trained_model(order=2)
        low = sample(model, ["the"], DecodingConfig(max_tokens=5, temperature=0.1), rng=random.Random(5))
        high = greedy(model, ["the"], DecodingConfig(max_tokens=5))
        self.assertEqual(low, high)

    def test_top_p_keeps_at_least_one(self):
        model = trained_model(order=2)
        config = DecodingConfig(max_tokens=3, top_p=0.01)
        out = sample(model, ["the"], config, rng=random.Random(2))
        self.assertGreaterEqual(len(out), 2)

    def test_beam_width_validation(self):
        with self.assertRaises(ValueError):
            beam_search(trained_model(), ["the"], 0)

    def test_beam_includes_prompt_and_orders(self):
        model = trained_model(order=2)
        beams = beam_search(model, ["the"], 3, DecodingConfig(max_tokens=6))
        self.assertLessEqual(len(beams), 3)
        self.assertTrue(all(b[0] == "the" for b in beams))
        lps = [model.log_probability(b[1:]) for b in beams]
        self.assertEqual(lps, sorted(lps, reverse=True))

    def test_beam_best_matches_greedy_when_greedy_is_optimal(self):
        model = trained_model(order=2)
        config = DecodingConfig(max_tokens=4)
        self.assertEqual(beam_search(model, ["the", "tokenizer"], 1, config)[0],
                         greedy(model, ["the", "tokenizer"], config))


class TestMetrics(unittest.TestCase):
    def test_distinct_n(self):
        self.assertEqual(distinct_n(["a", "b", "a", "b"], 1), 0.5)
        self.assertEqual(distinct_n(["a", "b", "c"], 2), 1.0)
        self.assertEqual(distinct_n(["a"], 2), 0.0)
        with self.assertRaises(ValueError):
            distinct_n(["a"], 0)

    def test_repetition_rate(self):
        self.assertEqual(repetition_rate(["a", "a", "b"]), 0.5)
        self.assertEqual(repetition_rate(["a"]), 0.0)
        self.assertEqual(repetition_rate([]), 0.0)

    def test_token_entropy(self):
        self.assertEqual(token_entropy([]), 0.0)
        self.assertEqual(token_entropy(["a", "a"]), 0.0)
        self.assertAlmostEqual(token_entropy(["a", "b"]), 1.0)

    def test_evaluate_keys(self):
        metrics = evaluate(["a", "b", "a"])
        self.assertEqual(set(metrics), {"tokens", "distinct_1", "distinct_2", "repetition_rate", "entropy_bits"})


class TestReport(unittest.TestCase):
    def setUp(self):
        self.report = build_report({"greedy": evaluate(["a", "b", "a", "b"])})

    def test_json_roundtrip(self):
        parsed = json.loads(to_json(self.report))
        self.assertIn("greedy", parsed["strategies"])

    def test_markdown_table(self):
        md = to_markdown(self.report)
        self.assertIn("| Strategy |", md)
        self.assertIn("| greedy |", md)

    def test_terminal_row(self):
        self.assertIn("greedy", to_terminal(self.report))


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.corpus = write_corpus(self.tmp)

    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_generate_greedy(self):
        code, out, _ = self.run_cli(
            ["generate", "--corpus", self.corpus, "--strategy", "greedy",
             "--prompt", "the tokenizer", "--max-tokens", "12"])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "the tokenizer splits text into tokens .")

    def test_generate_sample_seed(self):
        argv = ["generate", "--corpus", self.corpus, "--strategy", "sample",
                "--prompt", "the", "--max-tokens", "10", "--seed", "9"]
        self.assertEqual(self.run_cli(argv)[1], self.run_cli(argv)[1])

    def test_generate_beam(self):
        code, out, _ = self.run_cli(
            ["generate", "--corpus", self.corpus, "--strategy", "beam",
             "--prompt", "the tokenizer", "--beam-width", "2", "--max-tokens", "12"])
        self.assertEqual(code, 0)
        self.assertTrue(out.strip().startswith("the tokenizer"))

    def test_compare_writes_reports(self):
        jp = os.path.join(self.tmp, "r.json")
        mp = os.path.join(self.tmp, "r.md")
        code, out, _ = self.run_cli(
            ["compare", "--corpus", self.corpus, "--prompt", "the tokenizer",
             "--max-tokens", "10", "--json", jp, "--markdown", mp])
        self.assertEqual(code, 0)
        report = json.load(open(jp))
        self.assertEqual(len(report["strategies"]), 5)
        self.assertIn("| Strategy |", open(mp).read())
        self.assertIn("greedy", out)

    def test_compare_gate_passes(self):
        code, _, _ = self.run_cli(
            ["compare", "--corpus", self.corpus, "--prompt", "the tokenizer",
             "--max-tokens", "10", "--max-repetition", "0.5"])
        self.assertEqual(code, 0)

    def test_compare_gate_fails(self):
        code, _, err = self.run_cli(
            ["compare", "--corpus", self.corpus, "--prompt", "the tokenizer",
             "--max-tokens", "10", "--min-distinct-2", "1.5"])
        self.assertEqual(code, 1)
        self.assertIn("GATE FAIL", err)

    def test_missing_corpus(self):
        with self.assertRaises(SystemExit):
            self.run_cli(["generate", "--corpus", os.path.join(self.tmp, "nope.txt"),
                          "--strategy", "greedy"])


if __name__ == "__main__":
    unittest.main()
