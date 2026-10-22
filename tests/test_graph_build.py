"""Check that graph construction works without pre-existing processed data."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import graph_features


class GraphBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.folder = Path(temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(ROOT))
        self.addCleanup(temporary.cleanup)

    def test_utilities_import_without_data_or_sentence_transformers(self):
        script = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import graph_features; "
                  "assert 'eval_common' not in sys.modules; "
                  "assert 'sentence_transformers' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", script], cwd=self.folder,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_clean_build_writes_config_without_language_download(self):
        raw = self.folder / "data/raw"
        raw.mkdir(parents=True)
        records = [{"title": f"Dish {chr(97 + i // 26)}{chr(97 + i % 26)}",
                    "ingredients": json.dumps(["100 g butter", "200 g flour", "1 egg",
                                               "50 g sugar", "1 g salt"]),
                    "NER": json.dumps(["butter", "flour", "egg", "sugar", "salt"]),
                    "link": f"local/{i}", "source": "fixture"} for i in range(40)]
        pd.DataFrame(records).to_csv(raw / "RecipeNLG_dataset.csv", index=False)
        (self.folder / "data/ingredient_properties.csv").write_bytes(
            (ROOT / "data/ingredient_properties.csv").read_bytes())
        output = self.folder / "data/processed"
        self.assertFalse(output.exists())
        source = (ROOT / "build_graph.py").read_text(encoding="utf-8")
        previous = Path.cwd()
        try:
            os.chdir(self.folder)
            with patch.object(graph_features, "encode_words",
                              side_effect=lambda words: np.eye(len(words), 5, dtype=np.float32)), \
                    contextlib.redirect_stdout(io.StringIO()):
                exec(compile(source, str(ROOT / "build_graph.py"), "exec"), {})
        finally:
            os.chdir(previous)
        config = json.loads((output / "config.json").read_text())
        self.assertEqual(config["feature_model"], graph_features.FEATURE_MODEL)
        self.assertEqual(np.load(output / "ingredient_x.npy").shape, (5, 5))
        self.assertEqual(len(pd.read_csv(output / "recipes.csv")), 40)
        self.assertTrue((output / "edges.npz").is_file())


if __name__ == "__main__":
    unittest.main()
