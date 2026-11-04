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
from scipy.sparse import csr_matrix

from torch_geometric.nn.conv.gcn_conv import gcn_norm

from train_recipe_similarity import MODELS, bidirectional_edges, embed_new_recipes
from graph_features import (FEATURE_SCHEME, INFERENCE_POLICY, data_fingerprints,
                            pooled_recipe_features)


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
    directory: Path = Path("data/processed")
    split: np.ndarray = None   # "train", "val", "test" or "heldout" per recipe, from build_graph.py

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


def load_data(directory=DATA_DIR):
    directory = Path(directory)
    edges = np.load(directory / "edges.npz")
    weights = np.load(directory / "edge_weights.npz")
    ingredients = pd.read_csv(directory / "ingredients.csv")
    n_recipes = len(pd.read_csv(directory / "recipes.csv"))
    n_ingredients = len(ingredients)
    with open(directory / "config.json") as f:
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
    split = np.full(n_recipes, "", dtype=object)
    for name in ["train", "val", "test", "heldout"]:
        if name in edges:
            split[np.unique(edges[name][0])] = name

    return RecipeData(
        n_recipes, n_ingredients, ingredients,
        np.load(directory / "ingredient_x.npy").astype(np.float32), idf, share_exponent,
        recipe_ingredients, recipe_shares, known, train, weights["train"], directory, split,
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
        self.features = data.ingredient_x

    def embed(self, recipes):
        x = self.features
        return np.stack([(x[ids] * w[:, None]).sum(0) / w.sum() for ids, w in recipes])

    def recipes(self):
        return self.embed(all_recipes(self.data))


class PPMIMean(SentenceMean):
    """Ingredients described by the ingredients they appear with: positive PMI of
    co-occurrence in training recipes, reduced by SVD to `dim` numbers. A recipe is the
    weighted mean of its ingredients' vectors, as in SentenceMean, so two ingredients are
    alike when they have the same partners (second-order co-occurrence), whatever their names."""

    def __init__(self, data, dim=256, name="PPMI"):
        super().__init__(data, name)
        recipes, ingredients = data.train_edges
        occurs = csr_matrix((np.ones(len(recipes)), (recipes, ingredients)),
                            shape=(data.n_recipes, data.n_ingredients))
        counts = (occurs.T @ occurs).toarray()
        np.fill_diagonal(counts, 0)
        totals = counts.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log(counts * counts.sum() / np.outer(totals, totals))
        ppmi = np.where(counts > 0, np.maximum(pmi, 0), 0)
        u, s, _ = np.linalg.svd(ppmi)
        dim = min(dim, len(s))
        self.features = (u[:, :dim] * np.sqrt(s[:dim])).astype(np.float32)


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


class Centred:
    """A method with its mean embedding over training recipes subtracted. Embeddings tend to
    share one direction, which compresses all cosines; centring removes it."""

    def __init__(self, method, data):
        self.method, self.data, self.mean = method, data, None
        self.name, self.seed, self.inductive = method.name, getattr(method, "seed", None), method.inductive

    def recipes(self):
        z = self.method.recipes()
        if self.mean is None:
            self.mean = z[self.data.known].mean(0)
        return z - self.mean

    def embed(self, recipes):
        if self.mean is None:
            self.recipes()
        return self.method.embed(recipes) - self.mean


class Fusion:
    """Sentence mean and a GNN side by side, each scaled to length 1 and then by the square
    root of its weight; no training involved. With mode "cosine", cosine on the result is
    weight * cos(sentence mean) + (1 - weight) * cos(GNN). With "zscore", each cosine is first
    divided by its spread over random pairs of training recipes, so a method whose cosines
    vary little (heavily smoothed ones) is not drowned out and the weight means the same for
    every partner. Constant offsets do not change any ranking or comparison."""
    inductive = True
    PAIRS = 20_000     # random training-recipe pairs used to measure the spread

    def __init__(self, sentence, gnn, weight, known, mode="cosine"):
        self.sentence, self.gnn, self.weight, self.known, self.mode = sentence, gnn, weight, known, mode
        self.name, self.seed = f"fusion {gnn.name} {weight:g}", getattr(gnn, "seed", None)
        self.scales = None

    def spread(self, x):
        a, b = np.random.default_rng(0).choice(np.flatnonzero(self.known), (2, self.PAIRS))
        x = unit(x)
        return float(np.std((x[a] * x[b]).sum(1)))

    def combine(self, x, z):
        """Fused embeddings from the parts' embeddings. The first call must pass embeddings of
        all recipes, which set the zscore scales."""
        if self.scales is None:
            self.scales = (self.spread(x), self.spread(z)) if self.mode == "zscore" else (1.0, 1.0)
        return np.hstack([np.sqrt(self.weight / self.scales[0]) * unit(x),
                          np.sqrt((1 - self.weight) / self.scales[1]) * unit(z)]).astype(np.float32)

    def embed(self, recipes):
        if self.scales is None:
            self.recipes()
        return self.combine(self.sentence.embed(recipes), self.gnn.embed(recipes))

    def recipes(self):
        return self.combine(self.sentence.recipes(), self.gnn.recipes())


class Node2VecMethod:
    """Transductive: only recipes in the training graph have embeddings."""
    inductive = False

    def __init__(self, data, path=NODE2VEC_FILE, name="Node2Vec"):
        self.data, self.path, self.name = data, path, name

    def recipes(self):
        return np.load(self.path)["recipe_embeddings"]


class GNNMethod:
    """A trained two-layer GCN or GraphSAGE checkpoint. Recipes in the training graph
    are embedded in it; any other recipe is attached as a new node with edges from
    its ingredients only, so new recipes never influence each other or the graph.
    That makes the batch size a speed setting only: every batch is a pass over the
    whole graph, so large batches are much faster."""
    inductive = True

    def __init__(self, data, model_name, seed, name=None, checkpoint_path=None,
                 device="cpu", batch_size=8192):
        self.data, self.model_name, self.seed = data, model_name, seed
        self.name = name or model_name
        self.device, self.batch_size = torch.device(device), batch_size
        self.checkpoint_path = Path(checkpoint_path or SIMILARITY_DIR / f"{model_name}_inductive_s{seed}.pt")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        config = checkpoint["config"]
        expected = {"model": model_name, "seed": seed,
                    "input_dim": int(data.ingredient_x.shape[1]),
                    "data_sha256": data_fingerprints(data.directory),
                    "training_feature_scheme": FEATURE_SCHEME,
                    "inference_policy": INFERENCE_POLICY}
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"{self.checkpoint_path}: checkpoint {key} does not match. Retrain for this data/protocol.")
        self.config = config
        self.load_graph()
        self.model = MODELS[model_name](self.node_x.shape[1], config["hidden"], config["out"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def load_graph(self):
        """Node features and bidirectional training edges, built as in training."""
        data = self.data
        recipe_x = pooled_recipe_features(data.train_edges, data.ingredient_x,
                                          data.n_recipes, data.train_weights)
        self.ingredient_x = torch.as_tensor(data.ingredient_x, device=self.device)
        self.node_x = torch.cat([torch.as_tensor(recipe_x, device=self.device), self.ingredient_x])
        self.edges = bidirectional_edges(torch.as_tensor(data.train_edges, dtype=torch.long, device=self.device),
                                         data.n_recipes)
        self.edge_weights = torch.as_tensor(np.tile(data.train_weights, 2), device=self.device).float()

    @torch.no_grad()
    def embed(self, recipes):
        return embed_new_recipes(self.model, self.node_x, self.edges, self.edge_weights,
                                 self.ingredient_x, self.data.n_recipes, recipes,
                                 self.batch_size).cpu().numpy()

    @torch.no_grad()
    def recipes(self):
        embeddings = F.normalize(self.model(self.node_x, self.edges, self.edge_weights), dim=1)
        embeddings = embeddings[:self.data.n_recipes].cpu().numpy().copy()
        unseen = np.flatnonzero(~self.data.known)
        embeddings[unseen] = self.embed([self.data.recipe(r) for r in unseen])
        return embeddings


class SGCPropagation(torch.nn.Module):
    """K rounds of GCN propagation (symmetric normalisation, self-loops, edge weights) with
    no weights to learn: SGC (Wu et al. 2019) without its classifier."""

    def __init__(self, k, embedding_dim, alpha=0.0):
        super().__init__()
        self.k, self.embedding_dim, self.alpha = k, embedding_dim, alpha

    def forward(self, node_x, edge_index, edge_weight):
        """With alpha > 0 every hop keeps that fraction of each node's own input,
        z <- alpha * x + (1 - alpha) * A z, which is APPNP propagation (Gasteiger et al. 2019)."""
        edge_index, norm = gcn_norm(edge_index, edge_weight, len(node_x))
        source, target = edge_index
        adjacency = torch.sparse_coo_tensor(torch.stack([target, source]), norm,
                                            (len(node_x), len(node_x))).coalesce()
        z = node_x
        for _ in range(self.k):
            z = self.alpha * node_x + (1 - self.alpha) * torch.sparse.mm(adjacency, z)
        return z


class PropagationMethod(GNNMethod):
    """The GNN input features smoothed over the training graph by SGCPropagation, with no
    training. Recipes are embedded exactly as in GNNMethod, new ones by incoming edges only."""

    def __init__(self, data, k, alpha=0.0, device="cpu", batch_size=8192):
        self.data, self.seed = data, None
        self.name = f"appnp{alpha:g}" if alpha else f"sgc{k}"
        self.device, self.batch_size = torch.device(device), batch_size
        self.load_graph()
        self.model = SGCPropagation(k, self.node_x.shape[1], alpha)


APPNP_HOPS = 10


def build_methods(data, seeds, include_multihot=False, model_names=None,
                  similarity_dir=SIMILARITY_DIR, node2vec_file=NODE2VEC_FILE, fusion_weights=(),
                  propagation_hops=(), appnp_alphas=(), ppmi=False, centre=False, fusion_mode="cosine"):
    """All methods to evaluate. GNNs get one entry per seed; results are grouped by name.
    model_names defaults to every model in MODELS. The untrained graph methods: an SGC
    propagation per entry of propagation_hops, an APPNP propagation (APPNP_HOPS hops) per entry
    of appnp_alphas, and PPMI. Each fusion weight adds a Fusion of the sentence mean with every
    GNN and untrained graph method, after the methods it combines. centre=True centres every
    method except the fusions, which are built from centred parts."""
    def maybe_centred(method):
        return Centred(method, data) if centre else method

    sentence = maybe_centred(SentenceMean(data))
    methods = [sentence]
    if include_multihot:
        methods.insert(0, MultiHot(data))
    if Path(node2vec_file).exists():
        methods.append(maybe_centred(Node2VecMethod(data, node2vec_file)))
    partners = [GNNMethod(data, model_name, seed,
                          checkpoint_path=Path(similarity_dir) / f"{model_name}_inductive_s{seed}.pt")
                for model_name in model_names or MODELS for seed in seeds]
    partners += [PropagationMethod(data, k) for k in propagation_hops]
    partners += [PropagationMethod(data, APPNP_HOPS, alpha) for alpha in appnp_alphas]
    if ppmi:
        partners.append(PPMIMean(data))
    partners = [maybe_centred(m) for m in partners]
    methods += partners
    methods += [Fusion(sentence, partner, weight, data.known, fusion_mode)
                for partner in partners for weight in fusion_weights]
    return methods
