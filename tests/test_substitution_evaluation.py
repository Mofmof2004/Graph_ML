"""Protocol and numerical checks using a small, fully local recipe graph."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_substitutions as evaluation


class SubstitutionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.folder = Path(self.temp.name).resolve()
        # TemporaryDirectory cleanup is confined to this verified test workspace.
        self.assertTrue(self.folder.is_relative_to(ROOT))
        self.addCleanup(self.temp.cleanup)
        self.data_dir = self.folder / "processed"
        self.data_dir.mkdir()
        names = ["butter", "margarine", "oil", "sugar", "flour", "egg",
                 "honey", "vanilla", "salt", "rice"]
        lists = [[0, 3, 4], [1, 2, 5], [6, 7, 8], [9, 0, 6],
                 [0, 4, 5], [0, 3, 8], [3, 4, 9], [0, 5, 8]]
        all_edges = np.array([(r, i) for r, ids in enumerate(lists) for i in ids]).T
        nonheldout = all_edges[:, all_edges[0] < 4]
        splits = {"all": all_edges, "train": nonheldout[:, :-2],
                  "val": nonheldout[:, -2:-1], "test": nonheldout[:, -1:],
                  "heldout": all_edges[:, all_edges[0] >= 4]}
        shares = np.tile([0.2, 0.3, 0.5], 8)
        counts = np.bincount(splits["train"][1], minlength=len(names))
        idf = np.log((1 + len(np.unique(splits["train"][0]))) / (1 + counts)) + 1.0
        full_weights = idf[all_edges[1]] * np.sqrt(shares)
        split_weights = {name: full_weights[:10] if name == "train" else
                         full_weights[10:11] if name == "val" else
                         full_weights[11:12] if name == "test" else
                         full_weights[12:] if name == "heldout" else full_weights
                         for name in splits}
        rng = np.random.default_rng(123)
        features = rng.normal(size=(10, 5)).astype(np.float32)
        features[1] = features[0] + 0.1 * features[1]
        features[6] = features[3] + 0.5 * features[6]
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        pd.DataFrame({"recipe_id": range(8), "title": [f"Dish {i}" for i in range(8)],
                      "heldout": [False] * 4 + [True] * 4}).to_csv(self.data_dir / "recipes.csv", index=False)
        pd.DataFrame({"ingredient_id": range(10), "name": names, "idf": idf}).to_csv(
            self.data_dir / "ingredients.csv", index=False)
        np.savez(self.data_dir / "edges.npz", **splits)
        np.savez(self.data_dir / "edge_weights.npz", **split_weights, share=shares)
        np.save(self.data_dir / "ingredient_x.npy", features)
        (self.data_dir / "config.json").write_text(json.dumps(
            {"feature_model": "test-fixture-no-download", "share_exponent": 0.5}))
        self.subs = self.folder / "substitutions.csv"
        pd.DataFrame({"ingredient": ["butter", "butter", "sugar"],
                      "substitute": ["margarine", "oil", "honey"],
                      "source": ["manual", "manual", "manual"]}).to_csv(self.subs, index=False)
        self.data = evaluation.BenchmarkData.load(self.data_dir)

    def protocol(self, split="heldout"):
        return {"split": split, "seed": 0, "split_seed": 0, "validation_fraction": 0.5, "window": 50}

    def cases(self):
        return evaluation.freeze_cases(self.folder / "cases.json", self.data, self.subs,
                                       self.protocol())["cases"]

    def args(self, *extra):
        return evaluation.parse_args(["--data-dir", str(self.data_dir), "--substitutions", str(self.subs),
                                      "--output-dir", str(self.folder / "results"), "--split", "heldout",
                                      "--bootstrap-samples", "100", *extra])

    def test_split_disjointness_and_deterministic_cases(self):
        validation = evaluation.select_recipes(self.data, "validation", 0, 0.5)
        test = evaluation.select_recipes(self.data, "test", 0, 0.5)
        self.assertFalse(set(validation) & set(test))
        self.assertEqual(set(validation) | set(test), {4, 5, 6, 7})
        pairs, excluded, sources = evaluation.load_pairs(self.subs, self.data)
        first, _ = evaluation.generate_cases(self.data, pairs, excluded, sources, test, 0, 50)
        second, _ = evaluation.generate_cases(self.data, pairs, excluded, sources, test, 0, 50)
        self.assertEqual(first, second)
        self.assertEqual({c["recipe_id"] for c in first}, set(test))

    def test_known_substitutes_are_never_negatives_and_shares_are_preserved(self):
        cases = self.cases()
        _, excluded, _ = evaluation.load_pairs(self.subs, self.data)
        self.assertIn(0, excluded[1])  # A->B also excludes A as a negative when removing B.
        originals, plausible, random_, shares = evaluation.variants(cases)
        for c, original, a, b, sh in zip(cases, originals, plausible, random_, shares):
            self.assertNotIn(c["random_id"], excluded[c["removed_id"]])
            self.assertNotIn(c["random_id"], original)
            self.assertEqual(np.count_nonzero(original != a), 1)
            self.assertEqual(np.count_nonzero(original != b), 1)
            np.testing.assert_array_equal(sh, self.data.shares[c["recipe_id"]])
            w = evaluation.ingredient_weights(self.data, [a], [sh])[0]
            expected = self.data.idf[c["substitute_id"]] * np.sqrt(c["swapped_share"])
            self.assertAlmostEqual(w[c["position"]], expected)

    def test_weighted_mean_matches_same_mass_replacement_and_sentence_adapter(self):
        copies = evaluation.variants(self.cases())
        original, plausible, _, shares = copies
        z = evaluation.mean_embeddings(self.data, plausible, shares, weighted=True)
        ids, sh = plausible[0], shares[0]
        weights = self.data.idf[ids] * np.sqrt(sh)
        expected = sum(self.data.features[i] * w for i, w in zip(ids, weights)) / weights.sum()
        np.testing.assert_allclose(z[0], expected)
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, name, device):
                calls.append((name, device))

            def encode(self, texts, **kwargs):
                calls.extend(texts)
                return np.tile([1., 0.], (len(texts), 1))

        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeSentenceTransformer
        with patch.dict(sys.modules, {"sentence_transformers": module}):
            runs = list(evaluation.baseline_runs(self.data, ["sbert"], copies, 2, "cpu"))
        self.assertEqual(calls[0], ("test-fixture-no-download", "cpu"))
        self.assertEqual(len(calls) - 1, len(original) * 3)
        self.assertFalse(any("Dish" in text for text in calls[1:]))
        np.testing.assert_array_equal(runs[0][1], 1)

    def test_exact_jaccard_and_expected_ties(self):
        self.assertAlmostEqual(evaluation.jaccard_scores([[0, 1]], [[0, 2]])[0], 1 / 3)
        original, plausible, random_, _ = evaluation.variants(self.cases())
        a = evaluation.jaccard_scores(original, plausible)
        b = evaluation.jaccard_scores(original, random_)
        win, margin = evaluation.wins_from_scores(a, b, 1e-7)
        np.testing.assert_array_equal(win, 0.5)
        np.testing.assert_array_equal(margin, 0)

    def test_macro_weighting_paired_ci_and_empty_stratum(self):
        cases = [{"pair_id": pair, "pair_text_similarity": 0.2} for pair in ["a", "a", "a", "b"]]
        wins = np.array([1., 1., 1., 0.])
        summary = evaluation.summarize(cases, wins, np.zeros(4), 0.2, 0, 100)
        self.assertEqual(summary["all"]["micro_accuracy"], 0.75)
        self.assertEqual(summary["all"]["macro_accuracy"], 0.5)
        self.assertIsNone(summary["lower_text_similarity"]["macro_accuracy"])
        diff = evaluation.paired_comparison(cases, wins, wins, 0.2, 0, 100)
        self.assertEqual(diff["all"]["ci95"], [0., 0.])
        json.dumps(summary, allow_nan=False)

    def test_manifest_reuse_rejects_changed_settings_data_and_cases(self):
        path = self.folder / "cases.json"
        first = evaluation.freeze_cases(path, self.data, self.subs, self.protocol())
        content = path.read_bytes()
        self.assertEqual(first, evaluation.freeze_cases(path, self.data, self.subs, self.protocol()))
        self.assertEqual(content, path.read_bytes())
        with self.assertRaisesRegex(ValueError, "different data/settings"):
            evaluation.freeze_cases(path, self.data, self.subs, {**self.protocol(), "seed": 1})
        first["cases"][0]["random_id"] = -1
        path.write_text(json.dumps(first))
        with self.assertRaisesRegex(ValueError, "cases have changed"):
            evaluation.freeze_cases(path, self.data, self.subs, self.protocol())
        self.data.fingerprints["ingredient_x.npy"] = "changed"
        with self.assertRaisesRegex(ValueError, "different data/settings"):
            evaluation.freeze_cases(path, self.data, self.subs, self.protocol())

    def test_training_edges_cannot_include_heldout_recipes(self):
        with np.load(self.data_dir / "edges.npz") as archive:
            splits = {name: archive[name] for name in archive.files}
        splits["train"] = np.column_stack([splits["train"], [4, 0]])
        np.savez(self.data_dir / "edges.npz", **splits)
        with self.assertRaisesRegex(ValueError, "Training edges include held-out"):
            evaluation.BenchmarkData.load(self.data_dir)

    def test_offline_cli_runs_without_optional_models_and_reuses_cases(self):
        argv = ["--data-dir", str(self.data_dir), "--substitutions", str(self.subs),
                "--output-dir", str(self.folder / "cli"), "--bootstrap-samples", "100"]
        result = subprocess.run([sys.executable, str(ROOT / "evaluate_substitutions.py"), *argv],
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = self.folder / "cli"
        original_manifest = (output / "cases.json").read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluation.evaluate(evaluation.parse_args(argv))
        self.assertEqual(original_manifest, (output / "cases.json").read_bytes())
        self.assertEqual(set(report["methods"]), {"jaccard", "mean", "weighted_mean", "weighted_multihot"})
        self.assertEqual(report["methods"]["jaccard"]["all"]["macro_accuracy"], 0.5)
        scores = pd.read_csv(output / "scores.csv")
        self.assertEqual(len(scores), report["coverage"]["cases"] * 4)
        self.assertEqual(scores.groupby("method")["case_id"].apply(set).tolist(),
                         [set(scores["case_id"])] * 4)
        self.assertTrue((output / "summary.csv").exists())
        self.assertTrue((output / "comparison.md").exists())
        self.assertIsNone(report["methods"]["mean"]["all"]["seed_macro_std"])


    def test_inconsistent_training_weights_are_rejected(self):
        with np.load(self.data_dir / "edge_weights.npz") as archive:
            weights = {k: archive[k] for k in archive.files}
        weights["train"] = weights["train"] * 2
        np.savez(self.data_dir / "edge_weights.npz", **weights)
        with self.assertRaisesRegex(ValueError, "Training weights do not match"):
            evaluation.BenchmarkData.load(self.data_dir)


if __name__ == "__main__":
    unittest.main()
