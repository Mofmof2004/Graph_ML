"""Train inductive recipe/ingredient embeddings with an unsupervised random-walk loss.

The objective is the one from GraphSAGE (Hamilton et al., 2017): nodes that
co-occur on short random walks should have similar embeddings, and nodes drawn
from a degree-based noise distribution should not. Walks follow edges with
probability proportional to the IDF * quantity-share edge weights. Models train
on the non-held-out graph and embed held-out recipes from their ingredient
lists at inference time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import scatter

from graph_features import (FEATURE_SCHEME, INFERENCE_POLICY, data_fingerprints,
                            pooled_recipe_features)


DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("data/similarity")
SEED = 0
HIDDEN = 128
EMBEDDING_DIM = 128
DEFAULT_STEPS = 1000
EVAL_EVERY = 50
BATCH_NODES = 512        
WALK_LENGTH = 5          
NEGATIVES = 20           
NEGATIVE_POWER = 0.75    


def load_data(device, data_dir=DATA_DIR):
    data_dir = Path(data_dir)
    edges = np.load(data_dir / "edges.npz")
    recipes = pd.read_csv(data_dir / "recipes.csv")
    ingredient_x = torch.from_numpy(
        np.load(data_dir / "ingredient_x.npy")
    ).float().to(device)
    all_edges = edges["all"].astype(np.int64)
    all_weights = np.load(data_dir / "edge_weights.npz")["all"].astype(np.float32)
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
    train_edge_weights = torch.from_numpy(
        np.load(data_dir / "edge_weights.npz")["train"]
    ).float().to(device)
    return (
        all_edges, train_edges, train_edge_weights, heldout,
        recipe_ingredients, recipe_weights, ingredient_x,
    )


class WeightedWalker:
    """Weighted random walks: the next node is drawn with probability proportional
    to the edge weight. Nodes without edges stay where they are."""

    def __init__(self, edge_index, edge_weight, n_nodes):
        order = torch.argsort(edge_index[0], stable=True)
        
        self.targets = edge_index[1, order]
        
        self.degree = torch.bincount(edge_index[0], minlength=n_nodes)
        self.row_start = torch.cat([self.degree.new_zeros(1), self.degree.cumsum(0)])
        self.cumulative = torch.cat([
            edge_weight.new_zeros(1, dtype=torch.float64),
            edge_weight[order].double().cumsum(0),
        
        ])

    def walk(self, start, length, generator=None):
        steps = [start]
        current = start
        for _ in range(length):
            first, last = self.row_start[current], self.row_start[current + 1]
            low, high = self.cumulative[first], self.cumulative[last]
            draw = low + torch.rand(len(current), dtype=torch.float64,
                                    device=current.device, generator=generator) * (high - low)
            position = torch.searchsorted(self.cumulative, draw, right=True) - 1
            position = torch.minimum(torch.maximum(position, first), last - 1)
            has_edges = last > first
            current = torch.where(has_edges, self.targets[position.clamp(min=0)], current)
            steps.append(current)
        return torch.stack(steps, dim=1)


def sample_walk_pairs(walker, start_nodes, negative_distribution, generator=None):
    """Positive (anchor, context) pairs from walks and NEGATIVES noise nodes per pair."""
    start = start_nodes[torch.randint(len(start_nodes), (BATCH_NODES,),
                                      device=start_nodes.device, generator=generator)]
    walks = walker.walk(start, WALK_LENGTH, generator)
    anchors = walks[:, :1].expand(-1, WALK_LENGTH).reshape(-1)
    contexts = walks[:, 1:].reshape(-1)
    keep = anchors != contexts
    anchors, contexts = anchors[keep], contexts[keep]
    negatives = torch.multinomial(negative_distribution, len(anchors) * NEGATIVES,
                                  replacement=True, generator=generator)
    return anchors, contexts, negatives.view(len(anchors), NEGATIVES)


def random_walk_loss(embeddings, anchors, contexts, negatives):
    anchor = embeddings[anchors]
    positive_scores = (anchor * embeddings[contexts]).sum(-1)
    negative_scores = (anchor.unsqueeze(1) * embeddings[negatives]).sum(-1)
    return -(F.logsigmoid(positive_scores).mean()
             + F.logsigmoid(-negative_scores).mean())


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
    degree = degree.clamp(min=1e-12)
    return features / degree.unsqueeze(1)


def bidirectional_edges(edge_index, n_recipes):
    src, dst = edge_index[0], edge_index[1] + n_recipes
    return torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)


def merge_graphs(
    node_x,
    edge_index,
    edge_weight,
    new_node_x,
    new_edge_index,
    new_edge_weight,
):
    return (
        torch.cat([node_x, new_node_x]),
        torch.cat([edge_index, new_edge_index], dim=1),
        torch.cat([edge_weight, new_edge_weight]),
    )


def embed_new_recipes(model, node_x, graph_edges, graph_weights, ingredient_x,
                      n_recipes, recipes, batch_size=256):
    """Embed independent new recipes without changing the reference graph.

    Each recipe is (ingredient IDs, edge weights). The function is differentiable;
    callers use no_grad for evaluation. Only ingredient -> query edges are added.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if not recipes:
        return node_x.new_empty((0, model.embedding_dim))
    outputs = []
    n_nodes = len(node_x)
    for start in range(0, len(recipes), batch_size):
        chunk = recipes[start:start + batch_size]
        sources, targets, values, features = [], [], [], []
        for row, (ids, weights) in enumerate(chunk):
            ids = torch.as_tensor(ids, dtype=torch.long, device=node_x.device)
            weights = torch.as_tensor(weights, dtype=node_x.dtype, device=node_x.device)
            if (not len(ids) or weights.shape != ids.shape or not torch.isfinite(weights).all()
                    or (weights <= 0).any() or (ids < 0).any() or (ids >= len(ingredient_x)).any()):
                raise ValueError("Recipes need valid ingredient IDs and aligned positive weights.")
            sources.append(ids + n_recipes)
            targets.append(torch.full_like(ids, n_nodes + row))
            values.append(weights)
            features.append((ingredient_x[ids] * weights[:, None]).sum(0) / weights.sum())
        extra_edges = torch.stack([torch.cat(sources), torch.cat(targets)])
        embeddings = model(torch.cat([node_x, torch.stack(features)]),
                           torch.cat([graph_edges, extra_edges], dim=1),
                           torch.cat([graph_weights, torch.cat(values)]))
        outputs.append(F.normalize(embeddings[n_nodes:], dim=1))
    return torch.cat(outputs)


class WeightedSAGEConv(nn.Module):
    """GraphSAGE layer with a mean aggregator that weights neighbours by edge weight."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.self_linear = nn.Linear(input_dim, output_dim)
        self.neighbour_linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x, edge_index, edge_weight):
        source, target = edge_index
        # Projecting before aggregating gives the same result (the weighted mean commutes
        # with the bias-free neighbour_linear), with output_dim rather than input_dim per edge
        messages = self.neighbour_linear(x)
        total = scatter(messages[source] * edge_weight.unsqueeze(1), target, 0,
                        dim_size=len(x), reduce="sum")
        weight = scatter(edge_weight, target, 0, dim_size=len(x), reduce="sum")
        neighbours = total / weight.clamp(min=1e-12).unsqueeze(1)
        return self.self_linear(x) + neighbours


class GCNModel(nn.Module):
    """Two GCN layers."""

    def __init__(self, input_dim, hidden=HIDDEN, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, embedding_dim)

    def forward(self, node_x, edge_index, edge_weight):
        """Raw embeddings, used by the loss. Normalise them for retrieval."""
        hidden = F.relu(self.conv1(node_x, edge_index, edge_weight))
        return self.conv2(hidden, edge_index, edge_weight)


class GraphSAGEModel(nn.Module):
    """Two GraphSAGE layers with edge-weighted mean aggregation."""

    def __init__(self, input_dim, hidden=HIDDEN, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.conv1 = WeightedSAGEConv(input_dim, hidden)
        self.conv2 = WeightedSAGEConv(hidden, embedding_dim)

    def forward(self, node_x, edge_index, edge_weight):
        """Raw embeddings, used by the loss. Normalise them for retrieval."""
        hidden = F.relu(self.conv1(node_x, edge_index, edge_weight))
        return self.conv2(hidden, edge_index, edge_weight)


MODELS = {"gcn": GCNModel, "graphsage": GraphSAGEModel}


def train(model_name, steps, seed, data_dir=DATA_DIR, output_dir=OUTPUT_DIR,
          device=None, threads=None):
    if steps < 1:
        raise ValueError("steps must be positive.")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if threads is not None:
        if threads < 1:
            raise ValueError("threads must be positive.")
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fingerprints = data_fingerprints(data_dir)
    (
        all_edges, train_edge_index, train_edge_weights, heldout,
        recipe_ingredients, recipe_weights, ingredient_x,
    ) = load_data(device, data_dir)
    train_ids = np.unique(train_edge_index[0].cpu().numpy())
    if not len(train_ids) or heldout[train_ids].any():
        raise ValueError("Training graph must be nonempty and exclude reserved held-out recipes.")
    known = np.zeros(len(heldout), dtype=bool)
    known[train_ids] = True
    test_ids = np.flatnonzero(~known)
    n_recipes = len(recipe_ingredients)

    # Derive inputs from the actual training graph, never a stale feature cache.
    recipe_x = torch.from_numpy(pooled_recipe_features(
        train_edge_index.cpu().numpy(), ingredient_x.cpu().numpy(), n_recipes,
        train_edge_weights.cpu().numpy())).to(device)
    graph_edges = bidirectional_edges(train_edge_index, n_recipes)
    graph_edge_weights = torch.cat([train_edge_weights, train_edge_weights])
    node_x = torch.cat([recipe_x, ingredient_x])

    walker = WeightedWalker(graph_edges, graph_edge_weights, len(node_x))
    start_nodes = torch.nonzero(walker.degree > 0).squeeze(1)
    negative_distribution = walker.degree.float() ** NEGATIVE_POWER
    generator = torch.Generator(device=device).manual_seed(seed)

    model = MODELS[model_name](node_x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []
    training_started = time.perf_counter()
    for step in range(1, steps + 1):
        model.train()
        anchors, contexts, negatives = sample_walk_pairs(
            walker, start_nodes, negative_distribution, generator
        )
        embeddings = model(node_x, graph_edges, graph_edge_weights)
        loss = random_walk_loss(embeddings, anchors, contexts, negatives)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step}.")
        losses.append(float(loss.detach()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % EVAL_EVERY == 0 or step == steps:
            print(f"step {step:4d} loss {loss.item():.4f}")
    training_seconds = time.perf_counter() - training_started

    model.eval()
    with torch.no_grad():
        train_node_x = node_x
        train_embeddings = F.normalize(
            model(train_node_x, graph_edges, graph_edge_weights), dim=1
        ).cpu().numpy()

        query_recipes = []
        for recipe_id in test_ids:
            ids = np.array(sorted(recipe_ingredients[recipe_id]), dtype=np.int64)
            query_recipes.append((ids, np.array([recipe_weights[recipe_id][i] for i in ids])))
        test_embeddings = embed_new_recipes(
            model, node_x, graph_edges, graph_edge_weights, ingredient_x,
            n_recipes, query_recipes).cpu().numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_recipe_ids = np.concatenate([train_ids, test_ids])
    embedding_values = np.concatenate(
        [train_embeddings[train_ids], test_embeddings], axis=0
    )
    embedding_order = np.argsort(embedding_recipe_ids)
    np.savez(
        output_dir / f"{model_name}_inductive_s{seed}_embeddings.npz",
        embeddings=embedding_values[embedding_order],
        recipe_id=embedding_recipe_ids[embedding_order],
        split=np.where(
            known[embedding_recipe_ids[embedding_order]], "train", "unseen"
        ),
    )
    
    config = {
        "model": model_name,
        "seed": seed,
        "steps": steps,
        "loss": "graphsage random walk",
        "batch_nodes": BATCH_NODES,
        "walk_length": WALK_LENGTH,
        "negatives": NEGATIVES,
        "negative_power": NEGATIVE_POWER,
        "heldout_queries": int(heldout.sum()),
        "unseen_queries": len(test_ids),
        "data_sha256": fingerprints,
        "training_feature_scheme": FEATURE_SCHEME,
        "inference_policy": INFERENCE_POLICY,
        "input_dim": int(ingredient_x.shape[1]), "hidden": HIDDEN, "out": EMBEDDING_DIM,
        "learning_rate": 1e-3, "weight_decay": 1e-5,
        "negative_loss_weight": 1.0,
        "substitution_supervision": False,
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_loss_first": losses[0], "training_loss_last": losses[-1],
        "device": str(device),
        "training_seconds": training_seconds,
    }

    torch.save({"state_dict": model.state_dict(), "config": config},
               output_dir / f"{model_name}_inductive_s{seed}.pt")
    (output_dir / f"{model_name}_inductive_s{seed}.json").write_text(
        json.dumps(config, indent=2)
    )
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=list(MODELS), default="gcn", nargs="?")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, help="CPU threads; useful for reproducible pilot runs.")
    args = parser.parse_args()
    train(args.model, args.steps, args.seed, args.data_dir, args.output_dir, args.device, args.threads)


if __name__ == "__main__":
    main()
