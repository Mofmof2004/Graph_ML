"""Graph-building utilities that can be imported before processed data exists.

The sentence encoder is imported only when encoding is requested.
"""

import hashlib
from pathlib import Path

import numpy as np


FEATURE_MODEL = "all-mpnet-base-v2"


DATA_FILES = ("recipes.csv", "ingredients.csv", "edges.npz", "edge_weights.npz",
              "ingredient_x.npy", "config.json")
INFERENCE_POLICY = "incoming_only_v1"
FEATURE_SCHEME = "train_edge_weighted_ingredient_mean_v1"


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def data_fingerprints(directory):
    return {name: file_digest(Path(directory) / name) for name in DATA_FILES}


def calculate_edge_weights(graph, num_ingredients, edge_share=1, idf_graph=None,
                           share_exponent=0.5):
    """Fit IDF on the supplied reference graph, then weight the requested edges."""
    reference = graph if idf_graph is None else idf_graph
    counts = np.bincount(reference[1], minlength=num_ingredients)
    idf = np.log((1 + len(np.unique(reference[0]))) / (1 + counts)) + 1.0
    values = idf[graph[1]] * np.asarray(edge_share) ** share_exponent
    return values.astype(np.float32), idf


def pooled_recipe_features(edges, ingredient_x, n_recipes, weights=None):
    """One row per global recipe ID; recipes without supplied edges stay zero."""
    weights = np.ones(edges.shape[1]) if weights is None else np.asarray(weights)
    totals = np.zeros((n_recipes, ingredient_x.shape[1]), dtype=np.float64)
    np.add.at(totals, edges[0], ingredient_x[edges[1]] * weights[:, None])
    denominator = np.bincount(edges[0], weights=weights, minlength=n_recipes)
    return (totals / np.maximum(denominator[:, None], 1e-12)).astype(np.float32)


def encode_words(words, model_name=FEATURE_MODEL):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(words, batch_size=256, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
