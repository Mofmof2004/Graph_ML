"""Protocol and numerical checks using a small, fully local recipe graph."""

import importlib.util
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


    def test_features_keep_global_ids_and_normalize_small_total_weights(self):
        from graph_features import pooled_recipe_features
        edges = np.array([[0, 0, 2], [0, 1, 2]])
        features = np.eye(3, dtype=np.float32)
        pooled = pooled_recipe_features(edges, features, 4, np.array([0.1, 0.2, 0.1]))
        np.testing.assert_allclose(pooled[0], [1 / 3, 2 / 3, 0])
        np.testing.assert_array_equal(pooled[1], 0)
        np.testing.assert_array_equal(pooled[2], [0, 0, 1])


    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "PyTorch Geometric is not installed")
    def test_checkpoint_inference_has_no_query_to_query_influence(self):
        import torch
        from train_recipe_similarity import MODELS, train
        from embedders import GNNMethod, load_data
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        cases = self.cases()
        original, plausible, random_, shares = evaluation.variants(cases)
        weights = evaluation.ingredient_weights(self.data, original, shares)
        recipes = list(zip(original, weights))
        paths = []
        graph = load_data(self.data_dir)
        self.assertFalse((self.data_dir / "recipe_x_train_weighted.npy").exists())
        for name in MODELS:
            with self.subTest(model=name), contextlib.redirect_stdout(io.StringIO()):
                train(name, 2, 0, self.data_dir, self.folder, device="cpu", threads=1)
                path = self.folder / f"{name}_inductive_s0.pt"
                paths.append(path)
                adapter = GNNMethod(graph, name, 0, checkpoint_path=path, batch_size=1)
                separate = adapter.embed(recipes)
                adapter.batch_size = len(recipes)
                batch = adapter.embed(recipes)
                np.testing.assert_allclose(batch, separate, atol=1e-6)
                np.testing.assert_allclose(adapter.embed(recipes[::-1])[::-1], batch, atol=1e-6)
                self.assertTrue(np.isfinite(batch).all())
                with np.load(self.folder / f"{name}_inductive_s0_embeddings.npz") as exported:
                    np.testing.assert_allclose(adapter.recipes(), exported["embeddings"], atol=1e-6)
                base = adapter.model(adapter.node_x, adapter.edges, adapter.edge_weights).detach().clone()
                adapter.embed(recipes)
                torch.testing.assert_close(base, adapter.model(adapter.node_x, adapter.edges, adapter.edge_weights))
                checkpoint = torch.load(path, weights_only=True)
                checkpoint["config"]["data_sha256"]["edges.npz"] = "stale"
                stale = self.folder / f"{name}_stale.pt"
                torch.save(checkpoint, stale)
                with self.assertRaisesRegex(ValueError, "checkpoint data_sha256"):
                    GNNMethod(graph, name, 0, checkpoint_path=stale)
        with contextlib.redirect_stdout(io.StringIO()):
            report = evaluation.evaluate(self.args("--checkpoints", *map(str, paths)))
        for name in MODELS:
            self.assertIn(f"learned/{name}", report["paired_comparisons"])
            self.assertIsNone(report["methods"][f"learned/{name}"]["all"]["seed_macro_std"])

    @unittest.skipUnless(importlib.util.find_spec("torch_geometric"), "PyTorch Geometric is not installed")
    def test_checkpoint_seeds_share_cases_and_reject_mixed_configurations(self):
        import torch
        from train_recipe_similarity import train
        from embedders import GNNMethod, load_data

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        paths = [self.folder / f"gcn_inductive_s{seed}.pt" for seed in [0, 1]]
        with contextlib.redirect_stdout(io.StringIO()):
            for seed in [0, 1]:
                train("gcn", 2, seed, self.data_dir, self.folder, device="cpu", threads=1)
            report = evaluation.evaluate(self.args("--checkpoints", *map(str, paths)))
        learned = report["methods"]["learned/gcn"]["all"]
        per_seed = [run["metrics"]["all"]["macro_accuracy"] for run in report["runs"]
                    if run["method"] == "learned/gcn"]
        self.assertEqual(learned["runs"], 2)
        self.assertAlmostEqual(learned["macro_accuracy"], np.mean(per_seed))
        self.assertAlmostEqual(learned["seed_macro_std"], np.std(per_seed))
        scores = pd.read_csv(self.folder / "results/scores.csv")
        expected_cases = set(scores["case_id"])
        for _, rows in scores.groupby(["method", "seed"], dropna=False):
            self.assertEqual(set(rows["case_id"]), expected_cases)
        checkpoint = torch.load(paths[1], weights_only=True)
        checkpoint["config"]["steps"] = 123
        mismatched = self.folder / "different_budget.pt"
        torch.save(checkpoint, mismatched)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "different training configurations"):
            evaluation.evaluate(self.args("--checkpoints", str(paths[0]), str(mismatched)))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "Duplicate method/seed"):
            evaluation.evaluate(self.args("--checkpoints", str(paths[0]), str(paths[0])))
        del checkpoint["config"]["inference_policy"]
        legacy = self.folder / "legacy.pt"
        torch.save(checkpoint, legacy)
        with self.assertRaisesRegex(ValueError, "checkpoint inference_policy"):
            GNNMethod(load_data(self.data_dir), "gcn", 1, checkpoint_path=legacy)



if __name__ == "__main__":
    unittest.main()
