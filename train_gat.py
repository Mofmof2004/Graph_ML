"""Train the GAT + GraphSAGE encoder from the proposal on the recipe-ingredient graph.

Layer 1 learns how much each neighbour matters (attention, GATv2). With USE_EDGE_WEIGHTS the
IDF * quantity-share weight from build_graph.py is an edge feature, so attention starts from
the prior instead of learning that salt matters little from scratch. Layer 2 is GraphSAGE.

Trained without labels by link prediction: each step a fraction of the training edges is hidden,
the model passes messages over the rest, and must score the hidden (recipe, ingredient) pairs
above recipes paired with random ingredients.

Ablations from the proposal: LAYER1 = "mean" replaces attention with mean aggregation,
LAYER1 = "none" removes all edges (an MLP on the node features alone).
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GATv2Conv, SAGEConv

DATA_DIR = "data/processed"
OUTPUT = "data/gat"
LAYER1 = "gat"            # gat | mean | none
USE_EDGE_WEIGHTS = True   # give the attention layer the edge weights as an edge feature
HIDDEN = 128
OUT = 128
HEADS = 4
DROPOUT = 0.1
STEPS = 3000              # full-graph training steps; val AUC plateaus around here
LR = 1e-3
SUP_FRAC = 0.1            # fraction of training edges hidden and predicted each step
NEG_POWER = 0.75          # negatives are drawn proportional to ingredient degree ** NEG_POWER
EVAL_EVERY = 50
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, in_dim, layer1=LAYER1, edge_weights=USE_EDGE_WEIGHTS):
        super().__init__()
        self.layer1 = layer1
        self.edge_weights = edge_weights and layer1 == "gat"
        if layer1 == "gat":
            self.l1 = GATv2Conv(in_dim, HIDDEN, heads=HEADS, concat=False, dropout=DROPOUT,
                                edge_dim=1 if self.edge_weights else None, add_self_loops=False)
        elif layer1 == "mean":
            self.l1 = SAGEConv(in_dim, HIDDEN, aggr="mean")
        elif layer1 == "none":
            self.l1 = nn.Linear(in_dim, HIDDEN)
        else:
            raise ValueError(layer1)
        self.l2 = nn.Linear(HIDDEN, OUT) if layer1 == "none" else SAGEConv(HIDDEN, OUT, aggr="mean")
        self.scale = nn.Parameter(torch.tensor(10.0))   # temperature for the link scores

    def forward(self, x, edge_index, edge_weight=None, return_attention=False):
        attention = None
        if self.layer1 == "gat":
            edge_attr = edge_weight.unsqueeze(1) if self.edge_weights else None
            h, attention = self.l1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        elif self.layer1 == "mean":
            h = self.l1(x, edge_index)
        else:
            h = self.l1(x)
        h = F.dropout(F.relu(h), DROPOUT, self.training)
        z = self.l2(h) if self.layer1 == "none" else self.l2(h, edge_index)
        z = F.normalize(z, dim=-1)
        return (z, attention) if return_attention else z


class RecipeGraph:
    """Node ids: recipes 0..R-1, ingredients R..R+I-1. Message passing goes both ways along
    each recipe-ingredient edge; both directions carry the edge's weight."""

    def __init__(self, data_dir=DATA_DIR, device=DEVICE):
        edges = np.load(f"{data_dir}/edges.npz")
        weights = np.load(f"{data_dir}/edge_weights.npz")
        self.device = device
        self.ingredient_x = torch.from_numpy(np.load(f"{data_dir}/ingredient_x.npy")).to(device)
        self.R = int(edges["all"][0].max()) + 1
        self.I = self.ingredient_x.shape[0]
        self.weight_scale = float(weights["train"].mean())   # edge features are weight / mean training weight
        self.split = {k: torch.from_numpy(edges[k]).long().to(device) for k in ["train", "val", "test"]}
        self.split_w = {k: torch.from_numpy(weights[k]).float().to(device) / self.weight_scale
                        for k in ["train", "val", "test"]}

    def recipe_x(self, rec_ing, n_recipes):
        """Recipe features: mean of their ingredients' features over the given edges only,
        so hidden supervision edges do not leak into the recipe's own input."""
        x = torch.zeros(n_recipes, self.ingredient_x.shape[1], device=self.device)
        x.index_add_(0, rec_ing[0], self.ingredient_x[rec_ing[1]])
        deg = torch.bincount(rec_ing[0], minlength=n_recipes).clamp(min=1).unsqueeze(1)
        return x / deg

    def inputs(self, rec_ing, w):
        """Node features, both-direction edge index and weights for message passing over rec_ing."""
        x = torch.cat([self.recipe_x(rec_ing, self.R), self.ingredient_x])
        src, dst = rec_ing[0], rec_ing[1] + self.R
        edge_index = torch.cat([torch.stack([src, dst]), torch.stack([dst, src])], dim=1)
        return x, edge_index, torch.cat([w, w])


def link_scores(model, z, recipe, ingredient, R):
    return model.scale * (z[recipe] * z[ingredient + R]).sum(-1)


@torch.no_grad()
def validate(model, g, rng, neg_p):
    """ROC AUC of val edges against the same recipes paired with degree-sampled ingredients,
    passing messages over all training edges."""
    model.eval()
    x, ei, w = g.inputs(g.split["train"], g.split_w["train"])
    z = model(x, ei, w)
    pos = g.split["val"]
    neg_ing = torch.from_numpy(rng.choice(g.I, size=pos.shape[1], p=neg_p)).to(g.device)
    s_pos = link_scores(model, z, pos[0], pos[1], g.R).cpu().numpy()
    s_neg = link_scores(model, z, pos[0], neg_ing, g.R).cpu().numpy()
    return roc_auc_score(np.r_[np.ones(len(s_pos)), np.zeros(len(s_neg))], np.r_[s_pos, s_neg])


def train(name, layer1=LAYER1, edge_weights=USE_EDGE_WEIGHTS, seed=SEED):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    g = RecipeGraph()
    train_edges, train_w = g.split["train"], g.split_w["train"]
    E = train_edges.shape[1]
    deg = torch.bincount(train_edges[1], minlength=g.I).double().cpu().numpy() ** NEG_POWER
    neg_p = deg / deg.sum()

    model = Encoder(g.ingredient_x.shape[1], layer1, edge_weights).to(g.device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best = {"val_auc": -1.0}
    for step in range(1, STEPS + 1):
        model.train()
        perm = torch.from_numpy(rng.permutation(E)).to(g.device)
        n_sup = int(E * SUP_FRAC)
        sup, mp = perm[:n_sup], perm[n_sup:]
        x, ei, w = g.inputs(train_edges[:, mp], train_w[mp])
        z = model(x, ei, w)

        pos = train_edges[:, sup]
        neg_ing = torch.from_numpy(rng.choice(g.I, size=n_sup, p=neg_p)).to(g.device)
        logits = torch.cat([link_scores(model, z, pos[0], pos[1], g.R),
                            link_scores(model, z, pos[0], neg_ing, g.R)])
        labels = torch.cat([torch.ones(n_sup), torch.zeros(n_sup)]).to(g.device)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % EVAL_EVERY == 0:
            auc = validate(model, g, np.random.default_rng(1000), neg_p)
            print(f"step {step:4d} loss {loss.item():.4f} val AUC {auc:.4f}")
            if auc > best["val_auc"]:
                best = {"val_auc": auc, "step": step, "state": {k: v.clone() for k, v in model.state_dict().items()}}

    os.makedirs(OUTPUT, exist_ok=True)
    config = {"name": name, "layer1": layer1, "edge_weights": edge_weights and layer1 == "gat",
              "hidden": HIDDEN, "out": OUT, "heads": HEADS, "dropout": DROPOUT, "steps": STEPS, "lr": LR,
              "sup_frac": SUP_FRAC, "neg_power": NEG_POWER, "seed": seed,
              "weight_scale": g.weight_scale, "best_step": best["step"], "val_auc": best["val_auc"]}
    torch.save({"state_dict": best["state"], "config": config}, f"{OUTPUT}/{name}.pt")
    with open(f"{OUTPUT}/{name}.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"best val AUC {best['val_auc']:.4f} at step {best['step']} -> {OUTPUT}/{name}.pt")
    return config


def load(name, device=DEVICE):
    ckpt = torch.load(f"{OUTPUT}/{name}.pt", map_location=device, weights_only=True)
    c = ckpt["config"]
    model = Encoder(np.load(f"{DATA_DIR}/ingredient_x.npy").shape[1], c["layer1"], c["edge_weights"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, c


@torch.no_grad()
def embed_recipes(model, g, lists, weights, batch=20000):
    """Embed recipes given as ingredient-id lists, as new nodes on top of the training graph.
    Each new node only receives messages from its ingredients, so new nodes never affect each
    other or the existing graph, and the same procedure works for unseen recipes."""
    model.eval()
    x0, ei0, w0 = g.inputs(g.split["train"], g.split_w["train"])
    N = x0.shape[0]
    out = []
    for start in range(0, len(lists), batch):
        chunk, wchunk = lists[start:start + batch], weights[start:start + batch]
        n = len(chunk)
        src = torch.from_numpy(np.concatenate(chunk)).long().to(g.device) + g.R
        dst = torch.from_numpy(np.repeat(np.arange(n), [len(l) for l in chunk])).long().to(g.device)
        w = torch.from_numpy(np.concatenate(wchunk)).float().to(g.device) / g.weight_scale
        x_new = torch.zeros(n, x0.shape[1], device=g.device)
        x_new.index_add_(0, dst, g.ingredient_x[src - g.R])
        x_new /= torch.bincount(dst, minlength=n).clamp(min=1).unsqueeze(1)
        x = torch.cat([x0, x_new])
        ei = torch.cat([ei0, torch.stack([src, dst + N])], dim=1)
        z = model(x, ei, torch.cat([w0, w]))
        out.append(z[N:].cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    # usage: python train_gat.py [name layer1 edge_weights seed], e.g. gat_w gat 1 0
    if len(sys.argv) > 1:
        name, layer1, ew, seed = sys.argv[1], sys.argv[2], bool(int(sys.argv[3])), int(sys.argv[4])
        train(name, layer1, ew, seed)
    else:
        train(f"{LAYER1}{'_w' if USE_EDGE_WEIGHTS else ''}_s{SEED}")
