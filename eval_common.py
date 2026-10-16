"""Data and tests shared by evaluate_weights.py and evaluate_models.py.

Substitution test (proposal 3.1): for each recipe with an ingredient that has a known substitute,
make two copies, one with the substitute and one with a random ingredient of similar frequency.
A representation wins a case if the substitute copy is closer to the original.

Same-dish retrieval: recipes with the same title ("Banana Bread") should be each other's
nearest neighbours. Titles are not used by any representation.
"""

import json

import numpy as np
import pandas as pd

from normalisation import normalise
from sentence_transformers import SentenceTransformer

DATA_DIR = "data/processed"
SUBSTITUTIONS = "data/eval/substitutions_all.csv"   # from prepare_substitutions.py
WINDOW = 50              # random ingredients are drawn from this many frequency ranks around the target
MAX_GROUP = 30           # same-dish groups larger than this ("Fudge") are too generic to count
FEATURE_MODEL = "all-mpnet-base-v2"   # small, fast text model, 384 numbers per text

edges = np.load(f"{DATA_DIR}/edges.npz")
weights = np.load(f"{DATA_DIR}/edge_weights.npz")
recipes = pd.read_csv(f"{DATA_DIR}/recipes.csv")
ingredients = pd.read_csv(f"{DATA_DIR}/ingredients.csv")
ingredient_x = np.load(f"{DATA_DIR}/ingredient_x.npy")
with open(f"{DATA_DIR}/config.json") as f:
    SHARE_EXPONENT = json.load(f).get("share_exponent", 1.0)

edge_index, share = edges["all"], weights["share"]
idf = ingredients["idf"].to_numpy()
n_recipes, n_ingredients = len(recipes), len(ingredients)


# Frequency ranks, for drawing random ingredients as common as the true one
order = np.argsort(-ingredients["recipe_count"].to_numpy())
rank = np.empty(n_ingredients, dtype=int)
rank[order] = np.arange(n_ingredients)

edges_of_recipe = pd.Series(np.arange(edge_index.shape[1])).groupby(edge_index[0]).apply(np.array).to_dict()
ings_of_recipe = {r: set(edge_index[1, ks]) for r, ks in edges_of_recipe.items()}


def edge_weight(ids, shares, alpha=SHARE_EXPONENT):
    """Edge weight as in build_graph.py; alpha=None gives unit weights."""
    if alpha is None:
        return np.ones(len(ids))
    return idf[ids] * shares ** alpha


def matched_random(rng, target, exclude):
    lo, hi = max(rank[target] - WINDOW, 0), min(rank[target] + WINDOW, n_ingredients - 1)
    for _ in range(50):
        c = order[rng.integers(lo, hi + 1)]
        if c not in exclude and c != target:
            return c
    return None


def unit(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


def by_tercile(values, mask_fn):
    """Apply mask_fn to the lower, middle and upper third of `values`."""
    cuts = np.quantile(values, [1 / 3, 2 / 3])
    bucket = np.digitize(values, cuts)
    return [mask_fn(bucket == b) for b in range(3)], cuts


#----------Substitution cases----------#

def load_pairs(path=SUBSTITUTIONS):
    name2id = dict(zip(ingredients["name"], ingredients["ingredient_id"]))
    subs = pd.read_csv(path)
    pairs = {}
    for a, b in zip(subs["ingredient"], subs["substitute"]):
        a, b = normalise(a), normalise(b)
        if a in name2id and b in name2id and a != b:
            pairs.setdefault(name2id[a], []).append(name2id[b])
    return pairs


# Weight = IDF of the ingredient * its quantity share in the recipe, so rare ingredients
# that make up much of a recipe count most, and a pinch of salt counts very little.
# The share is compressed (square root by default): with the linear share, heavy but generic
# bulk such as 2 qt. water or 3 c. flour dominates the recipe, which made similarity worse.
# IDF uses the training edges only, so val/test edges do not leak into it.
def calculate_edge_weights(graph, num_ingredients, edge_share=1, idf_graph=None, share_exponent=SHARE_EXPONENT):
    """Calculate IDF-weighted edge weights for a graph.

    ``idf_graph`` is the graph used to fit the ingredient IDF values. When it
    is omitted, the graph being weighted is used, which is useful for
    standalone graphs. Passing the training graph keeps validation and test
    edges from affecting the IDF values.
    """
    if idf_graph is None:
        idf_graph = graph
    n_recipes = len(np.unique(idf_graph[0]))
    ingredient_df = np.bincount(idf_graph[1], minlength=num_ingredients)
    idf = np.log((1 + n_recipes) / (1 + ingredient_df)) + 1.0
    weights = (idf[graph[1]] * edge_share ** share_exponent).astype(np.float32)
    return weights, idf

def encode_words(words):
    model = SentenceTransformer(FEATURE_MODEL)   # downloads the model on first run
    return model.encode(words, batch_size=256,
                        normalize_embeddings=True, show_progress_bar=True).astype(np.float32)

#----------Same-dish retrieval----------#

titles = (recipes["title"].str.lower().str.replace(r"[^a-z ]", " ", regex=True)
          .str.split().str.join(" ").to_numpy())
group_size = pd.Series(titles).map(pd.Series(titles).value_counts()).to_numpy()
dish_queries = np.flatnonzero((group_size >= 2) & (group_size <= MAX_GROUP))


def same_dish(z):
    """MRR and hits@10 of finding another recipe with the same title. Recipes with an
    all-zero embedding (heldout recipes in Node2Vec) are left out."""
    z = unit(np.asarray(z, dtype=np.float32))
    valid = np.linalg.norm(z, axis=1) > 0
    q = [i for i in dish_queries if valid[i] and ((titles == titles[i]) & valid).sum() >= 2]
    sims = z[q] @ z.T
    sims[:, ~valid] = -np.inf
    sims[np.arange(len(q)), q] = -np.inf
    ranks = np.array([np.flatnonzero(titles[np.argsort(-sims[k])] == titles[i])[0] for k, i in enumerate(q)])
    return float(np.mean(1 / (ranks + 1))), float(np.mean(ranks < 10))
