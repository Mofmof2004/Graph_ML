"""Train inductive recipe-similarity models.

Similarity is defined by ingredient overlap, not recipe titles. Models train on
non-held-out recipes and embed held-out recipes from their available ingredient
lists at inference time. Evaluation retrieves similar training recipes for each
held-out query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/similarity")
SEED = 0
HIDDEN = 128
EMBEDDING_DIM = 128
LAYERS = 2
DEFAULT_STEPS = 1000
EVAL_EVERY = 50
POSITIVE_JACCARD = 0.30
NEGATIVE_JACCARD = 0.05
EVALUATION_K = 10


def load_data(device):
    edges = np.load(DATA_DIR / "edges.npz")
    recipes = pd.read_csv(DATA_DIR / "recipes.csv")
    ingredient_x = torch.from_numpy(
        np.load(DATA_DIR / "ingredient_x.npy")
    ).float().to(device)
    all_edges = edges["all"].astype(np.int64)
    all_weights = np.load(DATA_DIR / "edge_weights.npz")["all"].astype(np.float32)
    heldout = recipes["heldout"].to_numpy(dtype=bool)
    recipe_ingredients = [
        set(all_edges[1, all_edges[0] == recipe_id])
        for recipe_id in range(len(recipes))
    ]
    recipe_weights = []
    for recipe_id in range(len(recipes)):
        positions = np.flatnonzero(all_edges[0] == recipe_id)
        recipe_weights.append(dict(zip(all_edges[1, positions], all_weights[positions])))
    train_edges = torch.from_numpy(edges["train"]).long().to(device)
    return all_edges, train_edges, heldout, recipe_ingredients, recipe_weights, ingredient_x


def weighted_jaccard(first, second, weights):
    """Weighted Jaccard; rare, high-weight ingredients contribute more."""
    ingredients = first | second
    numerator = sum(min(weights[0].get(i, 0.0), weights[1].get(i, 0.0)) for i in ingredients)
    denominator = sum(max(weights[0].get(i, 0.0), weights[1].get(i, 0.0)) for i in ingredients)
    return numerator / denominator if denominator else 0.0


def build_training_pairs(recipe_ids, recipe_ingredients, recipe_weights, seed):
    rng = np.random.default_rng(seed)
    positives, negatives = [], []
    by_ingredient = {}
    for recipe_id in recipe_ids:
        for ingredient in recipe_ingredients[recipe_id]:
            by_ingredient.setdefault(ingredient, []).append(recipe_id)

    positive_candidates = set()
    for ids in by_ingredient.values():
        ids = np.asarray(ids)
        for _ in range(min(len(ids) * 3, 100)):
            if len(ids) < 2:
                break
            first, second = rng.choice(ids, 2, replace=False)
            similarity = weighted_jaccard(
                recipe_ingredients[first], recipe_ingredients[second],
                (recipe_weights[first], recipe_weights[second]),
            )
            if similarity >= POSITIVE_JACCARD:
                positive_candidates.add((min(first, second), max(first, second)))
    positives = list(positive_candidates)

    recipe_ids = np.asarray(recipe_ids)
    seen = set(positives)
    target = max(len(positives), 1)
    while len(negatives) < target:
        first, second = rng.choice(recipe_ids, 2, replace=False)
        pair = (min(first, second), max(first, second))
        similarity = weighted_jaccard(
            recipe_ingredients[first], recipe_ingredients[second],
            (recipe_weights[first], recipe_weights[second]),
        )
        if pair not in seen and similarity <= NEGATIVE_JACCARD:
            negatives.append(pair)
            seen.add(pair)
    return np.asarray(positives, dtype=np.int64), np.asarray(negatives, dtype=np.int64)


def recipe_features(edge_index, ingredient_x, n_recipes, edge_weights=None):
    features = torch.zeros(
        n_recipes, ingredient_x.shape[1], device=ingredient_x.device
    )
    values = ingredient_x[edge_index[1]]
    if edge_weights is not None:
        values = values * edge_weights.unsqueeze(1)
        degree = torch.zeros(n_recipes, device=ingredient_x.device)
        degree.index_add_(0, edge_index[0], edge_weights)
    else:
        degree = torch.bincount(edge_index[0], minlength=n_recipes)
    features.index_add_(0, edge_index[0], values)
    degree = degree.clamp(min=1)
    return features / degree.unsqueeze(1)


def bidirectional_edges(edge_index, n_recipes):
    src, dst = edge_index[0], edge_index[1] + n_recipes
    return torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)


class FeatureLightGCN(nn.Module):
    """LightGCN-style propagation whose recipe embeddings are feature-derived."""

    def __init__(self, recipe_x, ingredient_x, graph_edges, n_recipes):
        super().__init__()
        self.n_recipes = n_recipes
        self.register_buffer("recipe_x", recipe_x)
        self.register_buffer("ingredient_x", ingredient_x)
        self.register_buffer("graph_edges", graph_edges)
        self.projection = nn.Linear(recipe_x.shape[1], EMBEDDING_DIM)

    def _propagate(self, x, edge_index):
        src, dst = edge_index
        degree = torch.bincount(
            torch.cat([src, dst]), minlength=x.shape[0]
        ).float().clamp(min=1)
        norm = (degree[src] * degree[dst]).rsqrt()
        states = [x]
        for _ in range(LAYERS):
            message = torch.zeros_like(x)
            message.index_add_(0, dst, x[src] * norm.unsqueeze(1))
            message.index_add_(0, src, x[dst] * norm.unsqueeze(1))
            x = message
            states.append(x)
        return torch.stack(states).mean(0)

    def forward(self):
        x = self.projection(torch.cat([self.recipe_x, self.ingredient_x]))
        return F.normalize(self._propagate(x, self.graph_edges)[:self.n_recipes], dim=1)

    def embed_new(self, recipe_x, ingredient_ids, ingredient_weights):
        ingredient_z = self.projection(self.ingredient_x)
        recipe_z = self.projection(recipe_x)
        messages = []
        for row, ids in enumerate(ingredient_ids):
            if ids:
                ids = list(ids)
                weights = torch.tensor(
                    [ingredient_weights[row][ingredient_id] for ingredient_id in ids],
                    device=recipe_x.device,
                )
                messages.append(
                    (ingredient_z[ids] * weights.unsqueeze(1)).sum(0)
                    / weights.sum().clamp(min=1e-12)
                )
            else:
                messages.append(torch.zeros_like(recipe_z[row]))
        return F.normalize(recipe_z + torch.stack(messages), dim=1)


class GraphSAGEPairModel(nn.Module):
    def __init__(self, node_x, graph_edges, n_recipes):
        super().__init__()
        self.n_recipes = n_recipes
        self.register_buffer("node_x", node_x)
        self.register_buffer("graph_edges", graph_edges)
        self.conv1 = SAGEConv(node_x.shape[1], HIDDEN)
        self.conv2 = SAGEConv(HIDDEN, EMBEDDING_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(EMBEDDING_DIM * 4, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN, 1),
        )

    def encode(self, node_x=None, edge_index=None):
        node_x = self.node_x if node_x is None else node_x
        edge_index = self.graph_edges if edge_index is None else edge_index
        hidden = F.relu(self.conv1(node_x, edge_index))
        return F.normalize(self.conv2(hidden, edge_index), dim=1)

    def forward(self):
        return self.encode()

    def score(self, embeddings, pairs):
        first, second = embeddings[pairs[:, 0]], embeddings[pairs[:, 1]]
        features = torch.cat(
            [first, second, first * second, (first - second).abs()], dim=1
        )
        return self.decoder(features).squeeze(1)

    def embed_new(self, recipe_x, ingredient_ids, train_embeddings):
        recipe_x = recipe_x.to(self.node_x.device)
        n_existing = self.node_x.shape[0]
        n_new = recipe_x.shape[0]
        rows = []
        cols = []
        for row, ids in enumerate(ingredient_ids):
            rows.extend([self.n_recipes + ingredient_id
                         for ingredient_id in ids])
            cols.extend([n_existing + row] * len(ids))
        if rows:
            new_edges = torch.tensor(
                [rows, cols], dtype=torch.long, device=recipe_x.device
            )
        else:
            new_edges = torch.empty((2, 0), dtype=torch.long, device=recipe_x.device)
        x = torch.cat([self.node_x, recipe_x])
        edges = torch.cat([self.graph_edges, new_edges], dim=1)
        z = self.encode(x, edges)
        return z[n_existing:]


def pair_loss(model, embeddings, positives, negatives, lightgcn):
    positive_scores = (
        (embeddings[positives[:, 0]] * embeddings[positives[:, 1]]).sum(1)
        if lightgcn else model.score(embeddings, positives)
    )
    negative_scores = (
        (embeddings[negatives[:, 0]] * embeddings[negatives[:, 1]]).sum(1)
        if lightgcn else model.score(embeddings, negatives)
    )
    if lightgcn:
        return F.relu(0.2 - positive_scores + negative_scores).mean()
    return F.binary_cross_entropy_with_logits(
        torch.cat([positive_scores, negative_scores]),
        torch.cat([torch.ones_like(positive_scores), torch.zeros_like(negative_scores)]),
    )


@torch.no_grad()
def evaluate(
    query_embeddings, candidate_embeddings, query_ids, candidate_ids,
    recipe_ingredients, recipe_weights
):
    hits, reciprocal_ranks, ndcgs = [], [], []
    candidate_ids = np.asarray(candidate_ids)
    for row, query_id in enumerate(query_ids):
        similarities = query_embeddings[row] @ candidate_embeddings.T
        order = np.argsort(-similarities)
        relevant = np.array([
            weighted_jaccard(
                recipe_ingredients[query_id], recipe_ingredients[candidate_id],
                (recipe_weights[query_id], recipe_weights[candidate_id]),
            )
            >= POSITIVE_JACCARD
            for candidate_id in candidate_ids[order]
        ])
        relevant_count = int(relevant.sum())
        if relevant_count == 0:
            continue
        hits.append(float(relevant[:EVALUATION_K].sum() > 0))
        first = np.flatnonzero(relevant)
        reciprocal_ranks.append(1.0 / (first[0] + 1))
        gains = relevant[:EVALUATION_K] / np.log2(np.arange(2, EVALUATION_K + 2))
        ideal = np.sort(relevant)[::-1][:EVALUATION_K] / np.log2(
            np.arange(2, EVALUATION_K + 2)
        )
        ndcgs.append(float(gains.sum() / max(ideal.sum(), 1e-12)))
    return {
        "queries": len(hits),
        "hits_at_10": float(np.mean(hits)) if hits else None,
        "mrr": float(np.mean(reciprocal_ranks)) if hits else None,
        "ndcg_at_10": float(np.mean(ndcgs)) if hits else None,
    }


def train(model_name, steps, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_edges, train_edge_index, heldout, recipe_ingredients, recipe_weights, ingredient_x = load_data(device)
    train_ids = np.flatnonzero(~heldout)
    test_ids = np.flatnonzero(heldout)
    n_recipes = len(recipe_ingredients)
    train_weight_values = torch.from_numpy(
        np.load(DATA_DIR / "edge_weights.npz")["train"]
    ).float().to(device)
    recipe_x = recipe_features(train_edge_index, ingredient_x, n_recipes, train_weight_values)
    graph_edges = bidirectional_edges(train_edge_index, n_recipes)
    train_pairs, negatives = build_training_pairs(
        train_ids, recipe_ingredients, recipe_weights, seed
    )
    if len(train_pairs) == 0:
        raise RuntimeError("No positive ingredient-similarity pairs were found.")

    if model_name == "lightgcn":
        model = FeatureLightGCN(recipe_x, ingredient_x, graph_edges, n_recipes).to(device)
        lightgcn = True
    elif model_name == "graphsage":
        model = GraphSAGEPairModel(
            torch.cat([recipe_x, ingredient_x]), graph_edges, n_recipes
        ).to(device)
        lightgcn = False
    else:
        raise ValueError("model must be lightgcn or graphsage")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    positive_tensor = torch.from_numpy(train_pairs).long().to(device)
    negative_tensor = torch.from_numpy(negatives).long().to(device)
    for step in range(1, steps + 1):
        model.train()
        embeddings = model()
        loss = pair_loss(model, embeddings, positive_tensor, negative_tensor, lightgcn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == steps:
            print(f"step {step:4d} loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_embeddings = model().cpu().numpy()
        heldout_features = recipe_features(
            torch.from_numpy(all_edges[:, heldout[all_edges[0]]]).long().to(device),
            ingredient_x,
            n_recipes,
            torch.from_numpy(
                np.load(DATA_DIR / "edge_weights.npz")["heldout"]
            ).float().to(device),
        )[test_ids]
        heldout_lists = [recipe_ingredients[recipe_id] for recipe_id in test_ids]
        heldout_weights = [recipe_weights[recipe_id] for recipe_id in test_ids]
        if lightgcn:
            test_embeddings = model.embed_new(
                heldout_features, heldout_lists, heldout_weights
            ).cpu().numpy()
        else:
            test_embeddings = model.embed_new(
                heldout_features, heldout_lists, train_embeddings
            ).cpu().numpy()
    metrics = evaluate(
        test_embeddings,
        train_embeddings[train_ids],
        test_ids,
        train_ids,
        recipe_ingredients,
        recipe_weights,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model_name,
        "seed": seed,
        "steps": steps,
        "positive_weighted_jaccard": POSITIVE_JACCARD,
        "negative_weighted_jaccard": NEGATIVE_JACCARD,
        "heldout_queries": len(test_ids),
        "metrics": metrics,
    }
    torch.save({"state_dict": model.state_dict(), "config": config},
               OUTPUT_DIR / f"{model_name}_inductive_s{seed}.pt")
    (OUTPUT_DIR / f"{model_name}_inductive_s{seed}.json").write_text(
        json.dumps(config, indent=2)
    )
    print(f"inductive test: {metrics}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["lightgcn", "graphsage"])
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.model, args.steps, args.seed)


if __name__ == "__main__":
    main()
