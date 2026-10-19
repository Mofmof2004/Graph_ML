"""Recipe embedding methods behind one interface, so evaluations can treat them alike.

Every method has
    name        label used in tables and plots
    inductive   True if it can embed recipes that are not in the training graph
    recipes()   one embedding row per recipe (rows of unseen recipes are only
                meaningful when the method is inductive)
    embed(L)    embeddings for new recipes, given as a list of (ingredient ids, edge weights);
                only for inductive methods

To add a method, write a small class with these members and add it to build_methods().
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_recipe_similarity import MODELS, bidirectional_edges


DATA_DIR = Path("data/processed")
SIMILARITY_DIR = Path("data/similarity")
NODE2VEC_FILE = Path("data/node2vec/node2vec_train_weighted.npz")


#----------Data----------#

@dataclass
class RecipeData:
    n_recipes: int
    n_ingredients: int
    ingredients: pd.DataFrame
    ingredient_x: np.ndarray
    idf: np.ndarray
    share_exponent: float
    recipe_ingredients: list   # ingredient ids per recipe
    recipe_shares: list        # quantity share of each of those ingredients
    known: np.ndarray          # recipe has edges in the training graph
    train_edges: np.ndarray
    train_weights: np.ndarray

    def weights(self, ids, shares):
        """Edge weights as in build_graph.py: idf * share ** share_exponent."""
        return self.idf[ids] * shares ** self.share_exponent

    def recipe(self, recipe_id, replace=None):
        """(ingredient ids, edge weights) of a recipe, optionally with
        replace=(position, new ingredient) swapped in at the same quantity share."""
        ids = self.recipe_ingredients[recipe_id].copy()
        if replace is not None:
            ids[replace[0]] = replace[1]
        return ids, self.weights(ids, self.recipe_shares[recipe_id])


def load_data():
    edges = np.load(DATA_DIR / "edges.npz")
    weights = np.load(DATA_DIR / "edge_weights.npz")
    ingredients = pd.read_csv(DATA_DIR / "ingredients.csv")
    n_recipes = len(pd.read_csv(DATA_DIR / "recipes.csv"))
    n_ingredients = len(ingredients)
    with open(DATA_DIR / "config.json") as f:
        share_exponent = json.load(f)["share_exponent"]

    # IDF exactly as in build_graph.py (ingredients.csv only stores it rounded)
    train = edges["train"]
    df = np.bincount(train[1], minlength=n_ingredients)
    idf = np.log((1 + len(np.unique(train[0]))) / (1 + df)) + 1.0

    all_edges = edges["all"]
    order = np.argsort(all_edges[0], kind="stable")
    bounds = np.searchsorted(all_edges[0, order], np.arange(n_recipes + 1))
    recipe_ingredients = [all_edges[1, order[a:b]] for a, b in zip(bounds[:-1], bounds[1:])]
    recipe_shares = [weights["share"][order[a:b]] for a, b in zip(bounds[:-1], bounds[1:])]
    known = np.zeros(n_recipes, dtype=bool)
    known[np.unique(train[0])] = True

    return RecipeData(
        n_recipes, n_ingredients, ingredients,
        np.load(DATA_DIR / "ingredient_x.npy").astype(np.float32), idf, share_exponent,
        recipe_ingredients, recipe_shares, known, train, weights["train"],
    )


def all_recipes(data):
    return [data.recipe(r) for r in range(data.n_recipes)]


#----------Methods----------#

class MultiHot:
    """Ingredient weights as a sparse bag of ingredients; a sanity baseline that knows
    nothing about which ingredients are alike."""
    inductive = True

    def __init__(self, data, name="Multi-hot"):
        self.data, self.name = data, name

    def embed(self, recipes):
        out = np.zeros((len(recipes), self.data.n_ingredients), dtype=np.float32)
        for row, (ids, weights) in enumerate(recipes):
            out[row, ids] = weights
        return out

    def recipes(self):
        return self.embed(all_recipes(self.data))


class SentenceMean:
    """Edge-weighted mean of the sentence-encoder ingredient features. These are
    also the recipe input features of the GNNs, so the GNNs should beat this."""
    inductive = True

    def __init__(self, data, name="Sentence mean"):
        self.data, self.name = data, name

    def embed(self, recipes):
        x = self.data.ingredient_x
        return np.stack([(x[ids] * w[:, None]).sum(0) / w.sum() for ids, w in recipes])

    def recipes(self):
        return self.embed(all_recipes(self.data))


class Node2VecMethod:
    """Transductive: only recipes in the training graph have embeddings."""
    inductive = False

    def __init__(self, data, path=NODE2VEC_FILE, name="Node2Vec"):
        self.data, self.path, self.name = data, path, name

    def recipes(self):
        return np.load(self.path)["recipe_embeddings"]


class GNNMethod:
    """A trained GCNModel / GraphSAGEModel checkpoint. Recipes in the training graph
    are embedded in it; any other recipe is attached as a new node with edges from
    its ingredients only, so new recipes never influence each other or the graph."""
    inductive = True

    def __init__(self, data, model_name, seed, name=None):
        self.data, self.model_name, self.seed = data, model_name, seed
        self.name = name or model_name
        checkpoint = torch.load(SIMILARITY_DIR / f"{model_name}_inductive_s{seed}.pt",
                                map_location="cpu", weights_only=True)
        recipe_x = torch.from_numpy(np.load(DATA_DIR / "recipe_x_train_weighted.npy")).float()
        self.node_x = torch.cat([recipe_x, torch.from_numpy(data.ingredient_x)])
        self.edges = bidirectional_edges(torch.from_numpy(data.train_edges).long(), data.n_recipes)
        self.edge_weights = torch.from_numpy(np.tile(data.train_weights, 2)).float()
        self.model = MODELS[model_name](self.node_x.shape[1])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def embed(self, recipes):
        n_nodes, n_recipes = len(self.node_x), self.data.n_recipes
        source = np.concatenate([n_recipes + ids for ids, _ in recipes])
        target = np.concatenate([np.full(len(ids), n_nodes + row) for row, (ids, _) in enumerate(recipes)])
        weights = np.concatenate([w for _, w in recipes]).astype(np.float32)
        new_x = torch.from_numpy(SentenceMean(self.data).embed(recipes)).float()
        embeddings = self.model(
            torch.cat([self.node_x, new_x]),
            torch.cat([self.edges, torch.from_numpy(np.stack([source, target])).long()], dim=1),
            torch.cat([self.edge_weights, torch.from_numpy(weights)]),
        )
        return F.normalize(embeddings, dim=1)[n_nodes:].numpy()

    @torch.no_grad()
    def recipes(self):
        embeddings = F.normalize(self.model(self.node_x, self.edges, self.edge_weights), dim=1)
        embeddings = embeddings[:self.data.n_recipes].numpy().copy()
        unseen = np.flatnonzero(~self.data.known)
        embeddings[unseen] = self.embed([self.data.recipe(r) for r in unseen])
        return embeddings


def build_methods(data, seeds, include_multihot=False):
    """All methods to evaluate. GNNs get one entry per seed; results are grouped by name."""
    methods = [SentenceMean(data)]
    if include_multihot:
        methods.insert(0, MultiHot(data))
    if NODE2VEC_FILE.exists():
        methods.append(Node2VecMethod(data))
    for model_name in MODELS:
        methods += [GNNMethod(data, model_name, seed) for seed in seeds]
    return methods
